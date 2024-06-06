# FLMR
The huggingface-transformers implementation of Fine-grained Late-interaction Multi-modal Retriever.

The official implementation is at [here](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering).

The details of the model and checkpoints can be found [here](#models-and-benchmark-results).

The details for reproducing the datasets and evaluation in the paper can be found [here](docs/Datasets.md).

## Updates

- [05/06/2024] 🔥🔥🔥We made some updates to the implementation
  - Added an evaluation script that reproduces the results in the PreFLMR paper [here](#new-evaluate-the-preflmr-models-on-all-m2kr-benchmarks)
  - Added the updated benchmark results with the transformer implementation [here](#models-and-benchmark-results)
  - Added an example script to fine-tune PreFLMR on a custom retrieval dataset [here](#new-finetune-the-preflmr-model-on-downstream-datasets)
  - **IMPORTANT**: fixed the OVEN data splits in the M2KR benchmark, and updated each entry with a fixed instruction to ensure the evaluation result is not affected by random sampling of instructions. Please delete your local cache and download the dataset again.

## Table of Contents
- [FLMR](#flmr)
  - [Updates](#updates)
  - [Table of Contents](#table-of-contents)
  - [Models and Benchmark Results](#models-and-benchmark-results)
  - [How to use this package](#how-to-use-this-package)
    - [Environment](#environment)
    - [Index a custom document collection](#index-a-custom-document-collection)
    - [Search a custom document collection](#search-a-custom-document-collection)
    - [Training with contrastive learning](#training-with-contrastive-learning)
  - [Alternative: use transformers.AutoModel to load pre-trained models](#alternative-use-transformersautomodel-to-load-pre-trained-models)
  - [Use example scripts](#use-example-scripts)
    - [Use FLMR](#use-flmr)
    - [\[NEW!\] Use PreFLMR](#new-use-preflmr)
    - [\[NEW!\] Evaluate the PreFLMR models on all M2KR benchmarks](#new-evaluate-the-preflmr-models-on-all-m2kr-benchmarks)
    - [\[NEW!\] Finetune the PreFLMR model on downstream datasets](#new-finetune-the-preflmr-model-on-downstream-datasets)
      - [Run finetuning](#run-finetuning)
      - [Run Testing](#run-testing)
      - [Example finetuning results](#example-finetuning-results)
  - [Note](#note)
  - [Citation](#citation)

## Models and Benchmark Results

| Model         | WIT Recall@10 | IGLUE Recall@1 | KVQA Recall@5 | MSMARCO Recall@5 | OVEN Recall@5 | LLaVA Recall@1 | EVQA Recall@5 | EVQA Pseudo Recall@5 | OKVQA Recall@5 | OKVQA Pseudo Recall@5 | Infoseek Recall@5 | Infoseek Pseudo Recall@5 |
|---------------|---------------|----------------|---------------|------------------|---------------|----------------|---------------|----------------------|----------------|-----------------------|-------------------|--------------------------|
| [LinWeizheDragon/PreFLMR_ViT-G🤗](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-G) | 0.619         | 0.718          | 0.419         | 0.783            | 0.643         | 0.726          | 0.625         | 0.721                | 0.302          | 0.674                 | 0.392             | 0.577                    |
| [LinWeizheDragon/PreFLMR_ViT-L🤗](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L) | 0.605         | 0.699          | 0.440         | 0.779            | 0.608         | 0.729          | 0.609         | 0.708                | 0.314          | 0.690                 | 0.374             | 0.578                    |
| [LinWeizheDragon/PreFLMR_ViT-B🤗](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-B) | 0.427         | 0.574          | 0.294         | 0.786            | 0.468         | 0.673          | 0.550         | 0.663                | 0.272          | 0.658                 | 0.260             | 0.496                    |

**Note:** We converted the checkpoints from PyTorch to Huggingface-transformers, whose benchmark results differ from the numbers reported in the original paper slightly. You can reproduce the results in the above paper by referring to the instructions in [this document](docs/Datasets.md).

## How to use this package

### Environment

Create virtualenv:
```
conda create -n FLMR python=3.10 -y
conda activate FLMR
```
Install Pytorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install faiss

```
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```

Test if faiss generate error
```
python -c "import faiss"
```

Install FLMR
```
git clone https://github.com/LinWeizheDragon/FLMR.git
cd FLMR
pip install -e .
```

Install ColBERT engine
```
cd third_party/ColBERT
pip install -e .
```

Install other dependencies
```
pip install ujson gitpython easydict ninja datasets transformers
```

### Index a custom document collection
1. Load pre-trained models
    ```python
    import os
    import torch
    import pandas as pd
    import numpy as np
    from torchvision.transforms import ToPILImage
    from transformers import AutoImageProcessor

    from flmr import index_custom_collection
    from flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval

    # load models
    checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-G"
    image_processor_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
        checkpoint_path, subfolder="context_tokenizer"
    )

    model = FLMRModelForRetrieval.from_pretrained(
        checkpoint_path,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer,
    )
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
    ```

2. Create document collections
    ```python
    num_items = 100
    feature_dim = 1664
    passage_contents = [f"This is test sentence {i}" for i in range(num_items)]
    # Option 1. text-only documents
    custom_collection = passage_contents
    # Option 2. multi-modal documents with pre-extracted image features
    # passage_image_features = np.random.rand(num_items, feature_dim)
    # custom_collection = [
    #     (passage_content, passage_image_feature, None) for passage_content, passage_image_feature in zip(passage_contents, passage_image_features)
    # ]
    # Option 3. multi-modal documents with images
    # random_images = torch.randn(num_items, 3, 224, 224)
    # to_img = ToPILImage()
    # if not os.path.exists("./test_images"):
    #     os.makedirs("./test_images")
    # for i, image in enumerate(random_images):
    #     image = to_img(image)
    #     image.save(os.path.join("./test_images", "{}.jpg".format(i)))

    # image_paths = [os.path.join("./test_images", "{}.jpg".format(i)) for i in range(num_items)]

    # custom_collection = [
    #     (passage_content, None, image_path)
    #     for passage_content, image_path in zip(passage_contents, image_paths)
    # ]
    ```

3. Run indexing on the custom collection
    ```python
    index_custom_collection(
        custom_collection=custom_collection,
        model=model,
        index_root_path=".",
        index_experiment_name="test_experiment",
        index_name="test_index",
        nbits=8, # number of bits in compression
        doc_maxlen=512, # maximum allowed document length
        overwrite=True, # whether to overwrite existing indices
        use_gpu=False, # whether to enable GPU indexing
        indexing_batch_size=64,
        model_temp_folder="tmp",
        nranks=1, # number of GPUs used in indexing
    )
    ```
### Search a custom document collection
1. Create toy query data
    ```python
    num_queries = 2

    query_instructions = [f"instruction {i}" for i in range(num_queries)]
    query_texts = [f"{query_instructions[i]} : query {i}" for i in range(num_queries)]
    query_images = torch.zeros(num_queries, 3, 224, 224)
    query_encoding = query_tokenizer(query_texts)
    query_pixel_values = image_processor(query_images, return_tensors="pt")['pixel_values']
    ```

2. Obtain query embeddings with model
    ```python
    inputs = dict(
        input_ids=query_encoding['input_ids'],
        attention_mask=query_encoding['attention_mask'],
        pixel_values=query_pixel_values,
    )

    # Run model query encoding
    res = model.query(**inputs)

    queries = {i: query_texts[i] for i in range(num_queries)}
    query_embeddings = res.late_interaction_output
    ```

3. Search the collection
    ```python
    from flmr import search_custom_collection, create_searcher

    # initiate a searcher
    searcher = create_searcher(
        index_root_path=".",
        index_experiment_name="test_experiment",
        index_name="test_index",
        nbits=8, # number of bits in compression
        use_gpu=True, # whether to enable GPU searching
    )
    # Search the custom collection
    ranking = search_custom_collection(
        searcher=searcher,
        queries=queries,
        query_embeddings=query_embeddings,
        num_document_to_retrieve=5, # how many documents to retrieve for each query
    )

    # Analyse retrieved documents
    ranking_dict = ranking.todict()
    for i in range(num_queries):
        print(f"Query {i} retrieved documents:")
        retrieved_docs = ranking_dict[i]
        retrieved_docs_indices = [doc[0] for doc in retrieved_docs]
        retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
        retrieved_doc_texts = [passage_contents[doc_idx] for doc_idx in retrieved_docs_indices]

        data = {
            "Confidence": retrieved_doc_scores,
            "Content": retrieved_doc_texts,
        }

        df = pd.DataFrame.from_dict(data)

        print(df)
    ```

### Training with contrastive learning
```python
import torch
from flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval

checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-L"
image_processor_name = "openai/clip-vit-large-patch14"
query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="context_tokenizer")

model = FLMRModelForRetrieval.from_pretrained(checkpoint_path,
                                                query_tokenizer=query_tokenizer,
                                                context_tokenizer=context_tokenizer,
                                                )

Q_encoding = query_tokenizer(["Using the provided image, obtain documents that address the subsequent question: What is the capital of France?", "Extract documents linked to the question provided in conjunction with the image: What is the capital of China?"])
D_encoding = context_tokenizer(["Paris is the capital of France.", "Beijing is the capital of China.",
                            "Paris is the capital of France.", "Beijing is the capital of China."])
Q_pixel_values = torch.zeros(2, 3, 224, 224)
inputs = dict(
    query_input_ids=Q_encoding['input_ids'],
    query_attention_mask=Q_encoding['attention_mask'],
    query_pixel_values=Q_pixel_values,
    context_input_ids=D_encoding['input_ids'],
    context_attention_mask=D_encoding['attention_mask'],
    use_in_batch_negatives=True,
)

res = model.forward(**inputs)
print(res)
```

**Note** that the examples in this code block are only for demonstration purposes. They show that the pre-trained model gives higher scores to correct documents. In real training, you always need to pass in the documents in the order "positive doc for query1, negative doc1 for query1, negative doc2 for query1, ..., positive doc for query2, negative doc1 for query2, negative doc2 for query2, ...".  You may want to read the later section which provides an example finetuning script.

## Alternative: use transformers.AutoModel to load pre-trained models
```
pip install transformers
```

```python
from transformers import AutoConfig, AutoModel, AutoImageProcessor, AutoTokenizer
import torch

checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-L"
image_processor_name = "openai/clip-vit-large-patch14"
query_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer", trust_remote_code=True)
context_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, subfolder="context_tokenizer", trust_remote_code=True)

model = AutoModel.from_pretrained(checkpoint_path,
                                query_tokenizer=query_tokenizer,
                                context_tokenizer=context_tokenizer,
                                trust_remote_code=True,
                                )
image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
```


## Use example scripts
We provide two scripts to show how the pretrained models can be used in evaluation:
1. `examples/example_use_flmr.py`: an example script to evaluate FLMR (with 10 ROIs) on [OK-VQA](https://okvqa.allenai.org/).
2. `examples/example_use_preflmr.py`: an example script to evaluate PreFLMR on [E-VQA](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa).

### Use FLMR
```bash
cd examples/
```

Download `KBVQA_data` from [here](https://huggingface.co/datasets/BByrneLab/RAVQAV2Data) and unzip the image folders. The ROI/captioning/object detection results have been included.

Run the following command (remove `--run_indexing` if you have already run indexing once):

```bash
python example_use_flmr.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name OKVQA_GS\
            --experiment_name OKVQA_GS \
            --indexing_batch_size 64 \
            --image_root_dir /path/to/KBVQA_data/ok-vqa/ \
            --dataset_path BByrneLab/OKVQA_FLMR_preprocessed_data \
            --passage_dataset_path BByrneLab/OKVQA_FLMR_preprocessed_GoogleSearch_passages \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 \
            --checkpoint_path LinWeizheDragon/FLMR \
            --image_processor_name openai/clip-vit-base-patch32 \
            --query_batch_size 8 \
            --num_ROIs 9 \
```

### [NEW!] Use PreFLMR
You can download the E-VQA images from https://github.com/google-research/google-research/tree/master/encyclopedic_vqa. We will add a dataset link here soon.

```bash
cd examples/
```

Run the following command (remove `--run_indexing` if you have already run indexing once):

```bash
python example_use_preflmr.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name EVQA_PreFLMR_ViT-G \
            --experiment_name EVQA \
            --indexing_batch_size 64 \
            --image_root_dir /rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/EVQA/eval_image/ \
            --dataset_hf_path BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
            --dataset EVQA \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 500 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
            --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
            --query_batch_size 8 \
            --compute_pseudo_recall \
```
Here, we upload all the M2KR datasets into one HF dataset `BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR` with different datasets as subset.
To reproduce results of the other datasets in the paper, you can change the `--dataset` to `OKVQA`, `KVQA`, `LLaVA`, `OVEN`, `Infoseek`, `WIT`, `IGLUE` and `EVQA`.

**Updates**:

- Enable `--compute_pseudo_recall` to compute pseudo recall for datasets like EVQA/OKVQA/Infoseek
- Enable `--Ks 1 5 10 20 50 100 500`: max(Ks) needs to be 500 to match the performance reported in the PreFLMR paper.

### [NEW!] Evaluate the PreFLMR models on all M2KR benchmarks

Change the image root paths in `examples/evaluate_all.sh` and execute:

```bash
cd examples
bash evaluate_all.sh
```

Obtain the report by:

```bash
python report.py
```

###  [NEW!] Finetune the PreFLMR model on downstream datasets

You will need to install pytorch-lightning:

```
pip install pytorch-lightning==2.1.0
```

#### Run finetuning
```bash
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
```
#### Run Testing
```bash
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
```

#### Example finetuning results

By running the above script, we are able to obtain the following finetuning performance:

| Step  | Pseudo Recall@5 on EVQA |
|-------|-------------------------|
| 2500  | 73.6                    |
| 10000 | 73.55                   |
| 12000 | 74.21                   |
| 14000 | 73.73                   |

(Checkpoints with low validation losses were picked and tested, run on 2 A100 GPUs)

![Screenshot 2024-06-05 171340](https://github.com/LinWeizheDragon/FLMR/assets/33350454/13da0d3e-b0b7-45d2-9c61-466ea07c9032)

## Note
The FLMR model is implemented following the documentation style of `transformers`. You can find detailed documentation in the modeling files. 


## Citation
If our work helped your research, please kindly cite our paper for FLMR and PreFLMR. 
```
@inproceedings{
    lin2023finegrained,
    title={Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering},
    author={Weizhe Lin and Jinghong Chen and Jingbiao Mei and Alexandru Coca and Bill Byrne},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=IWWWulAX7g}
        }
        
@article{Lin_Mei_Chen_Byrne_2024, 
        title={PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers}, 
        url={http://arxiv.org/abs/2402.08327}, 
        number={arXiv:2402.08327}, 
        publisher={arXiv}, 
        author={Lin, Weizhe and Mei, Jingbiao and Chen, Jinghong and Byrne, Bill}, 
        year={2024}}

```

