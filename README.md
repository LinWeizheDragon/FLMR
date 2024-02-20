# FLMR
The huggingface-transformers implementation of Fine-grained Late-interaction Multi-modal Retriever.

The official implementation is at [here](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering).

## Use with AutoModel
```
pip install transformers
```
```
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


## Example scripts to run indexing and retrieval with FLMR or PreFLMR

### Environment

Create virtualenv:
```
conda create -n FLMR_new python=3.10 -y
conda activate FLMR_new
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
cd examples/research_projects/flmr-retrieval/third_party/ColBERT
pip install -e .
```

Install other dependencies
```
pip install ujson gitpython easydict ninja datasets
```

### Use the model directly
```
import torch
from flmr.models.flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval

checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-L"
image_processor_name = "openai/clip-vit-large-patch14"
query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="context_tokenizer")

model = FLMRModelForRetrieval.from_pretrained(checkpoint_path,
                                                query_tokenizer=query_tokenizer,
                                                context_tokenizer=context_tokenizer,
                                                )
```

### Use FLMR
```
cd examples/
```

Download `KBVQA_data` from [here](https://huggingface.co/datasets/BByrneLab/RAVQAV2Data) and unzip the image folders.

Run the following command (remove `--run_indexing` if you have already run indexing once):

```
python example_use_flmr.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name OKVQA_GS\
            --experiment_name OKVQA_GS \
            --indexing_batch_size 64 \
            --image_root_dir /path/to/KBVQA_data/ok-vqa/ \
            --dataset_path LinWeizheDragon/OKVQA_FLMR_preprocessed_data \
            --passage_dataset_path LinWeizheDragon/OKVQA_FLMR_preprocessed_GoogleSearch_passages \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 \
            --checkpoint_path LinWeizheDragon/FLMR \
            --image_processor_name openai/clip-vit-base-patch32 \
            --query_batch_size 8 \
            --num_ROIs 9 \
```

### Use PreFLMR
```
cd examples/
```

Run the following command (remove `--run_indexing` if you have already run indexing once):

```
python example_use_preflmr.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name EVQA_PreFLMR_ViT-G \
            --experiment_name EVQA \
            --indexing_batch_size 64 \
            --image_root_dir /rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/EVQA/eval_image/ \
            --dataset_path LinWeizheDragon/EVQA_PreFLMR_preprocessed_data \
            --passage_dataset_path LinWeizheDragon/EVQA_PreFLMR_preprocessed_passages \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
            --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
            --query_batch_size 8 \
            --run_indexing
```

## Note
The FLMR model is implemented following the documentation style of `transformers`. You can find detailed documentation in the modeling files. 
