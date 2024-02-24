# Model Zoo

For the huggingface implementation, we provide four pretrained checkpoints of FLMR and PreFLMR models. 

# PreFLMR
PreFLMR is pretrained on WIT, CC3M, MSMARCO, KVQA, LLaVA, OVEN, OKVQA, Infoseek and E-VQA datasets. For details on the split for training and evaluation, hyperparameter used for training the PreFLMR models, please refer to the paper [PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers](https://arxiv.org/abs/2402.08327). Here, we release three pretrained checkpoints:

| Model   | Vision Encoder | Text Encoder | Checkpoint Name   | No. Param. | WIT   | LLaVA  | OVEN  | KVQA  | IGLUE | Infoseek | E-VQA | OKVQA |
|---------|----------------|--------------|-------------------------------------------------------------|-------|-------|--------|-------|-------|-------|----------|-------|--------|
| PreFLMR | ViT-B          | Base-v2      | [LinWeizheDragon/PreFLMR_ViT-B](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-B) | 327M |34.2  | 50.9   | 46.1  | 28.9  | 60.5  | 42.5     | 32.7  | 46.5   |
| PreFLMR | ViT-L          | Base-v2      | [LinWeizheDragon/PreFLMR_ViT-L](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L) | 543M |49.6  | 51.2   | 54.8  | 40.5  | 69.5  | 48.7     | 45.0  | 50.9   |0
| PreFLMR | ViT-G          | Base-v2      | [LinWeizheDragon/PreFLMR_ViT-G](https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-G) | 2.1B | 49.5  | 51.8   | 59.6  | 38.7  | 69.3  | 50.9     | 42.4  | 52.1   |

For the evaluation metrics, WIT uses Recall@10, IGLUE uses Recall@1, all the rest datasets use Recall@5.


The ViT-B is initialized from `openai/clip-vit-base-patch32`, ViT-L is initialized from `openai/clip-vit-large-patch14-336` and ViT-G is initialized from `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`. The ViT is frozen during the pretraining of PreFLMR. Base-v2 text encoder refer to the base-sized BERT model from [ColBERTv2](https://github.com/stanford-futuredata/ColBERT). 

<p align="center">
  <img src="./radar_plot.jpg" width="500px"> <br>
  PreFLMR outperforms baseline on all nine benchmarks except for the text-only MSMARCO task.
</p>

# FLMR

We provide FLMR checkpoints in addition to the PreFLMR checkpoints. The FLMR is from the paper [Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering](https://arxiv.org/abs/2309.17133). The mapping network of FLMR is pretrained with the subset of WIT which contains about 200K image-text pairs. FLMR uses `openai/clip-vit-base-patch32` as the vision encoder and the same Base-v2 ColBERTv2 as the text encoder. The FLMR checkpoint is: [LinWeizheDragon/FLMR](https://huggingface.co/LinWeizheDragon/FLMR).
