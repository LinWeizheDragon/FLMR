# FLMR
The huggingface-transformers implementation of Fine-grained Late-interaction Multi-modal Retriever.

The official implementation is at [here](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering).

The details of the model and checkpoints can be found [here](docs/MODEL_ZOO.md).

The details for reproducing the datasets and evaluation in the paper will be released soon.

## Table of Contents
- [FLMR](#flmr)
  - [Table of Contents](#table-of-contents)
  - [How to use this package](#how-to-use-this-package)
    - [Environment](#environment)
    - [Index a custom document collection](#index-a-custom-document-collection)
    - [Search a custom document collection](#search-a-custom-document-collection)
    - [Training with contrastive learning](#training-with-contrastive-learning)
  - [Alternative: use transformers.AutoModel to load pre-trained models](#alternative-use-transformersautomodel-to-load-pre-trained-models)
  - [Use example scripts](#use-example-scripts)
    - [Use FLMR](#use-flmr)
    - [Use PreFLMR](#use-preflmr)
  - [Note](#note)
  - [Citation](#citation)


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
```
cd examples/
```

Download `KBVQA_data` from [here](https://huggingface.co/datasets/BByrneLab/RAVQAV2Data) and unzip the image folders. The ROI/captioning/object detection results have been included.

Run the following command (remove `--run_indexing` if you have already run indexing once):

```
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

### Use PreFLMR
You can download the E-VQA images from https://github.com/google-research/google-research/tree/master/encyclopedic_vqa. We will add a dataset link here soon.

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
            --dataset_hf_path BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
            --dataset EVQA \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
            --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
            --query_batch_size 8 
```
Here, we upload all the M2KR datasets into one HF dataset `BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR` with different datasets as subset.
To reproduce results of the other datasets in the paper, you can change the `--dataset` to `OKVQA`, `KVQA`, `LLaVA`, `OVEN`, `Infoseek`, `WIT`, `IGLUE` and `E-VQA`.


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

