import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset, DatasetDict
from transformers import set_seed, AutoImageProcessor
from PIL import Image
import argparse
import random
from easydict import EasyDict
import numpy as np
import shutil
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
from functools import partial


from flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval

class RetrievalModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        self.checkpoint_path = self.args.checkpoint_path
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(self.checkpoint_path, subfolder="query_tokenizer")
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(self.checkpoint_path, subfolder="context_tokenizer")
        self.image_processor = AutoImageProcessor.from_pretrained(self.args.image_processor_name)
        
        # Load and prepare datasets
        self.prepare_datasets()

        self.train_dataloader()

        self.model = FLMRModelForRetrieval.from_pretrained(self.checkpoint_path,
                                                           query_tokenizer=self.query_tokenizer,
                                                           context_tokenizer=self.context_tokenizer)
        
        if self.args.freeze_vit:
            # freeze parameters of query_encoder and context_encoder
            for name, param in self.model.query_vision_encoder.named_parameters():
                param.requires_grad = False
            for name, param in self.model.context_vision_encoder.named_parameters():
                param.requires_grad = False

        
        

    def prepare_datasets(self):
        self.dataset = load_dataset(self.args.dataset_hf_path, self.args.dataset + "_data")
        self.passage_ds = load_dataset(self.args.dataset_hf_path, self.args.dataset + "_passages")
        
        def add_path_prefix_in_img_path(example, prefix):
            example["img_path"] = os.path.join(prefix, example["img_path"])
            return example

        self.dataset = self.dataset.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": self.args.image_root_dir})

        instructions = [
            "Using the provided image, obtain documents that address the subsequent question: ",
            "Retrieve documents that provide an answer to the question alongside the image: ",
            "Extract documents linked to the question provided in conjunction with the image: ",
            "Utilizing the given image, obtain documents that respond to the following question: ",
            "Using the given image, access documents that provide insights into the following question: ",
            "Obtain documents that correspond to the inquiry alongside the provided image: ",
            "With the provided image, gather documents that offer a solution to the question: ",
            "Utilizing the given image, obtain documents that respond to the following question: ",
        ]
        

        def prepare_inputs(sample):
            sample = EasyDict(sample)

            random_instruction = random.choice(instructions)
            input_text_sequence = " ".join(
                [random_instruction]
                + [sample.question]
            )

            sample["input_text_sequence"] = input_text_sequence

            return sample
        
        self.dataset = self.dataset.map(prepare_inputs)

        print(self.dataset['train'][0])
        
        

        # Tokenize and prepare image pixels for input
        # ds = ds.map(
        #     tokenize_inputs,
        #     fn_kwargs={"query_tokenizer": self.query_tokenizer, "context_tokenizer": self.context_tokenizer, "image_processor": self.image_processor},
        #     batched=True,
        #     batch_size=8,
        #     num_proc=16,
        # )

    def collate_fn(self, batch, passage_split="train_passages"):
        
        batch_data = {}

        input_text_sequences = [example['input_text_sequence'] for example in batch]
        encoding = self.query_tokenizer(input_text_sequences)
        query_input_ids = encoding["input_ids"]
        query_attention_mask = encoding["attention_mask"]
        
        img_paths = [example['img_path'] for example in batch]
        pixel_values = []
        for img_path in img_paths:
            image = Image.open(img_path).convert("RGB")
            encoded = self.image_processor(image, return_tensors="pt")
            pixel_values.append(encoded.pixel_values)

        pixel_values = torch.stack(pixel_values, dim=0)

        num_negative_examples = self.args.num_negative_examples

        
        def negative_sampling(pos_item_ids, num_samples=1):
            """Generate negative samples for a query. ONLY used in training
            Args:
                user_item (int tensor): user id
                num_samples (int, optional): number of samples. Defaults to 1.
            Returns:
                neg_items: list of negative item ids.
            """
            neg_items = []
            
            while len(neg_items) < num_samples:
                # sample num_samples negative items for the user
                while True:
                    neg_item = np.random.randint(low=0, high=len(self.passage_ds), size=1)[0]
                    
                    VALID = True
                    neg_item = self.passage_ds[passage_split][int(neg_item)]
                    if neg_item['passage_id'] in pos_item_ids:
                        VALID = False
                    
                    if VALID == True:
                        break
                neg_items.append(neg_item)
            return neg_items
        
        batched_context_input_sequences = []

        for example in batch:
            select_pos_item_index = random.sample(range(len(example['pos_item_ids'])), k=1)[0]
            pos_item_id = example['pos_item_ids'][select_pos_item_index]
            pos_item_content = example['pos_item_contents'][select_pos_item_index]

            batched_context_input_sequences.append(pos_item_content)

            neg_items = negative_sampling(pos_item_id, num_samples=num_negative_examples)
            neg_item_ids = [item['passage_id'] for item in neg_items]
            neg_item_contents = [item['passage_content'] for item in neg_items]

            batched_context_input_sequences.extend(neg_item_contents)
        
        context_encoding = self.context_tokenizer(batched_context_input_sequences)
        context_input_ids = context_encoding["input_ids"]
        context_attention_mask = context_encoding["attention_mask"]

        batch_data.update(dict(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            query_pixel_values=pixel_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
        ))
        # print(query_input_ids.shape)
        # print(query_attention_mask.shape)
        # print(pixel_values.shape)
        # print(context_input_ids.shape)
        # print(context_attention_mask.shape)
        return batch_data


    def train_dataloader(self):
        # Create a partial function with parameters
        parametrized_collate_fn = partial(self.collate_fn, passage_split="train_passages")

        dataloader = DataLoader(
            self.dataset['train'], 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            collate_fn=parametrized_collate_fn,
            num_workers=4,
        )
        return dataloader
    
    def val_dataloader(self):
        # Create a partial function with parameters
        parametrized_collate_fn = partial(self.collate_fn, passage_split="valid_passages")

        dataloader = DataLoader(
            self.dataset['valid'], 
            batch_size=self.args.valid_batch_size, 
            collate_fn=parametrized_collate_fn,
            num_workers=2,
        )
        return dataloader

    def test_dataloader(self):
        # Create a partial function with parameters
        parametrized_collate_fn = partial(self.collate_fn, passage_split="test_passages")

        dataloader = DataLoader(
            self.dataset['test'], 
            batch_size=self.args.test_batch_size, 
            collate_fn=parametrized_collate_fn,
            num_workers=2,
        )
        return dataloader

    def forward(self, batch):
        batch = {
            k: v.to(self.device) for k,v in batch.items()
        }
        # Prepare inputs for model
        inputs = {
            'query_input_ids': batch['query_input_ids'],
            'query_attention_mask': batch['query_attention_mask'],
            'query_pixel_values': batch['query_pixel_values'],
            'context_input_ids': batch['context_input_ids'],
            'context_attention_mask': batch['context_attention_mask'],
            'use_in_batch_negatives': True,
            "num_negative_examples": self.args.num_negative_examples,
        }

        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        current_lrs = self.scheduler.get_last_lr()
        for index, current_lr in enumerate(current_lrs):
            self.log(f"train/lr[{index}]", current_lr, prog_bar=True, on_step=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('valid/loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('test/loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)

        from transformers import get_scheduler
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


class ModelSaveCallback(Callback):
    def __init__(self, save_path, save_top_k=3):
        self.save_path = save_path
        self.best_models = []
        self.save_top_k = save_top_k
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.state.stage in ['sanity_check']:
            return
        current_loss = trainer.callback_metrics["valid/loss"].item()
        current_step = trainer.global_step
        model_name = f"model_step_{current_step}"
        model_path = os.path.join(self.save_path, model_name)

        if self.save_top_k == -1:
            # save all models
            pl_module.model.save_pretrained(model_path)
            pl_module.query_tokenizer.save_pretrained(os.path.join(model_path, "query_tokenizer"))
            pl_module.context_tokenizer.save_pretrained(os.path.join(model_path, "context_tokenizer"))
            print(f"\nThe metric is {current_loss}, save_top_k=-1. Saving {model_path}")
            return
        
        if len(self.best_models) < self.save_top_k or current_loss < max(self.best_models, key=lambda x: x[1])[1]:
            print(f"\nThe metric is {current_loss}, at the top {self.save_top_k}. Saving {model_path}")
            
            self.best_models.append((model_path, current_loss))
            self.best_models.sort(key=lambda x: x[1])

            if len(self.best_models) > self.save_top_k:
                removed_model = self.best_models.pop()
                print("\nRemoving", removed_model[0])
                try:
                    shutil.rmtree(removed_model[0], ignore_errors=True)
                except Exception as e:
                    print(f"\nRemove failed. The file may have been removed. \nError: {e}")
            
            pl_module.model.save_pretrained(model_path)
            pl_module.query_tokenizer.save_pretrained(os.path.join(model_path, "query_tokenizer"))
            pl_module.context_tokenizer.save_pretrained(os.path.join(model_path, "context_tokenizer"))
            
        else:
            print(f"\nThe current metric is {current_loss}, not at the top {self.save_top_k}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--save_top_k", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dataset_hf_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--log_with_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="finetune_preflmr")
    parser.add_argument("--image_root_dir", type=str, required=True)
    parser.add_argument("--image_processor_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--checkpoint_path", type=str, default="LinWeizheDragon/PreFLMR_ViT-L", required=True)
    parser.add_argument("--num_negative_examples", type=int, default=4)
    parser.add_argument("--freeze_vit", action="store_true")
    parser.add_argument("--model_save_path", type=str, default="saved_models")

    # Parse known and unknown arguments
    args, unknown_args = parser.parse_known_args()
    # Convert unknown args to kwargs for Trainer
    trainer_kwargs = {}
    it = iter(unknown_args)
    for key in it:
        if key.startswith('--'):
            key = key.lstrip('--')
            try:
                value = next(it)
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
            except StopIteration:
                raise ValueError(f"Argument {key} lacks a corresponding value.")
            trainer_kwargs[key] = value

    set_seed(42)  # Set seeds for reproducibility

    model = RetrievalModel(args)
    print("trainer_kwargs", trainer_kwargs)

    # checkpoint_callback = ModelCheckpoint(monitor="valid/loss", mode="min", save_top_k=0, save_last=True)
    save_pretrained_callback = ModelSaveCallback(save_path=args.model_save_path, save_top_k=args.save_top_k)

    if args.log_with_wandb:
        wandb_logger = WandbLogger(project=args.wandb_project)

    trainer = Trainer(
        default_root_dir=args.model_save_path,
        callbacks=[save_pretrained_callback],
        enable_checkpointing=False,
        logger=wandb_logger if args.log_with_wandb else None,
        **trainer_kwargs)
    
    if args.mode == 'train':
        trainer.fit(model)
    else:
        trainer.test(model)

if __name__ == "__main__":
    main()

"""
Example Use:

# Start training
python example_finetune_preflmr.py \
--image_root_dir /path/to/EVQA/images/ \
--dataset_hf_path BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
--dataset EVQA \
--freeze_vit \
--log_with_wandb \
--model_save_path saved_models \
--checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
--image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
--batch_size 8 \
--accumulate_grad_batches 8 \
--valid_batch_size 16 \
--test_batch_size 64 \
--mode train \
--max_epochs 99999999 \
--learning_rate 0.000005 \
--warmup_steps 100 \
--accelerator auto \
--devices auto \
--strategy ddp_find_unused_parameters_true \
--num_sanity_val_steps 2 \
--precision bf16 \
--val_check_interval 2000 \
--save_top_k -1 \

# Run Testing
python example_use_preflmr.py \
    --use_gpu --run_indexing \
    --index_root_path "." \
    --index_name EVQA_PreFLMR_ViT-G_finetuned_model_step_10156 \
    --experiment_name EVQA \
    --indexing_batch_size 64 \
    --image_root_dir /path/to/EVQA/images/ \
    --dataset_hf_path BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
    --dataset EVQA \
    --use_split test \
    --nbits 8 \
    --num_gpus 1 \
    --Ks 1 5 10 20 50 100 500 \
    --checkpoint_path saved_models/model_step_10156 \
    --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --query_batch_size 8 \
"""