import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union

from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig



def index_custom_collection(
    custom_collection: List, 
    model: Union[str, nn.Module],
    index_root_path: str = ".", 
    index_experiment_name: str = "default_experiment", 
    index_name: str = "index",
    nbits: int = 8,
    doc_maxlen: int = 512,
    overwrite: bool = False,
    use_gpu: bool = True,
    indexing_batch_size: int = 64,
    model_temp_folder: str = "tmp",
    nranks: int = 1,
) -> str:

    # Launch indexer
    with Run().context(RunConfig(nranks=nranks, root=index_root_path, experiment=index_experiment_name)):

        # check if the index already exists
        index_name = f"{index_name}.nbits={nbits}"
        index_path = os.path.join(index_root_path, index_experiment_name, "indexes", index_name)
        
        if os.path.exists(index_path) and not overwrite:
            print(f"The index {index_path} exists. Skipping...\n Set `overwrite=True` to re-generate the index.")
            return index_path
        
        config = ColBERTConfig(
            nbits=nbits,
            doc_maxlen=doc_maxlen,
            total_visible_gpus=nranks if use_gpu else 0,
        )
        print("indexing with", nbits, "bits")

        if isinstance(model, str):
            # The input model is already a checkpoint
            checkpoint_path = model
        else:
            # Limited by the original ColBERT engine, we save the checkpoint and pass it to the Indexer
            checkpoint_path = os.path.join(model_temp_folder, "temp_model")
            print(f"Limited by the original ColBERT engine, we save the checkpoint at {checkpoint_path} and pass it to the Indexer \n You can set the temp folder by passing `model_temp_folder`. \n You can also pass in an existing model checkpoint like `LinWeizheDragon/PreFLMR_ViT-G` or other models stored locally.")
            model.save_pretrained(checkpoint_path)
            if getattr(model, "query_tokenizer", None) is not None:
                model.query_tokenizer.save_pretrained(os.path.join(checkpoint_path, "query_tokenizer"))
            if getattr(model, "context_tokenizer", None) is not None:
                model.context_tokenizer.save_pretrained(os.path.join(checkpoint_path, "context_tokenizer"))

        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(
            name=index_name,
            collection=custom_collection,
            batch_size=indexing_batch_size,
            overwrite=True,
        )
        index_path = indexer.get_index()

        return index_path