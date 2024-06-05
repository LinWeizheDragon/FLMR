#!/bin/bash

declare -A datasets
datasets=(
  ["WIT"]='/root/rds/data/WIT/images'
  ["IGLUE"]='/root/rds/data/WIT/images'
  ["KVQA"]='/root/rds/data/KVQA/KVQAimgs'
  ["OVEN"]='/root/rds/data/OVEN'
  ["LLaVA"]='/root/rds/data/OKVQA'
  ["Infoseek"]='/root/rds/data/Infoseek/val'
  ["EVQA"]='/root/rds/data/EVQA/images'
  ["OKVQA"]='/root/rds/data/OKVQA'
  ["MSMARCO"]=''
)

declare -A processor_names
processor_names=(["ViT-G"]='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
                 ["ViT-L"]='openai/clip-vit-large-patch14'
                 ["ViT-B"]='openai/clip-vit-base-patch32')

declare -A additional_args
additional_args=(["WIT"]=''
          ["IGLUE"]=''
          ["KVQA"]=''
          ["OVEN"]=''
          ["LLaVA"]=''
          ["Infoseek"]='--compute_pseudo_recall'
          ["EVQA"]='--compute_pseudo_recall'
          ["OKVQA"]='--compute_pseudo_recall'
          ["MSMARCO"]=''
          )

for dataset in "${!datasets[@]}"; do
  for model in ViT-G ViT-L ViT-B; do
    python example_use_preflmr.py \
      --use_gpu --run_indexing \
      --num_gpus 1 \
      --index_root_path "./Index" \
      --index_name "${dataset}_PreFLMR_${model}" \
      --experiment_name "${dataset}" \
      --indexing_batch_size 64 \
      --image_root_dir "${datasets[$dataset]}" \
      --dataset_hf_path BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
      --dataset "${dataset}" \
      --use_split test \
      --nbits 8 \
      --Ks 1 5 10 20 50 100 500 \
      --checkpoint_path "LinWeizheDragon/PreFLMR_${model}" \
      --image_processor_name "${processor_names[$model]}" \
      --query_batch_size 8 \
      ${additional_args[$dataset]}
  done
done

python report.py