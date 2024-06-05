import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union

from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
from easydict import EasyDict
from PIL import Image

def create_searcher(
        index_root_path: str = ".", 
        index_experiment_name: str = "default_experiment", 
        index_name: str = "index",
        use_gpu: bool = True,
        nbits: int = 8,
    ) -> Searcher:
    # Search documents
    with Run().context(RunConfig(nranks=1, rank=1, root=index_root_path, experiment=index_experiment_name)):
        if use_gpu:
            total_visible_gpus = torch.cuda.device_count()
        else:
            total_visible_gpus = 0

        config = ColBERTConfig(
            total_visible_gpus=total_visible_gpus,
        )
        searcher = Searcher(
            index=f"{index_name}.nbits={nbits}", checkpoint=None, config=config
        )

        return searcher

def search_custom_collection(
        searcher: Searcher,
        queries: Dict[int, str],
        query_embeddings: torch.tensor,
        num_document_to_retrieve: int = 100,
        remove_zero_tensors: bool = True,
        centroid_search_batch_size: int = None,
        **kwargs,
    ) -> Dict: 

        queries = Queries(data=queries)

        search_results = searcher._search_all_Q(
            queries,
            query_embeddings,
            progress=False,
            batch_size=centroid_search_batch_size,
            k=num_document_to_retrieve,
            remove_zero_tensors=remove_zero_tensors,  # For PreFLMR, this is needed
            **kwargs, # other arguments
        )

        return search_results