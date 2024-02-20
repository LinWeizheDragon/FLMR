import torch
import shutil

from transformers import AutoConfig, AutoModel
from transformers import AutoImageProcessor, AutoTokenizer
from models.flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRConfig


if __name__ == '__main__':
    
    AutoConfig.register("flmr", FLMRConfig)
    AutoModel.register(FLMRConfig, FLMRModelForRetrieval)

    FLMRConfig.register_for_auto_class()
    FLMRModelForRetrieval.register_for_auto_class("AutoModel")
    FLMRQueryEncoderTokenizer.register_for_auto_class("AutoTokenizer")
    FLMRContextEncoderTokenizer.register_for_auto_class("AutoTokenizer")

    checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-G"
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="context_tokenizer")

    model = FLMRModelForRetrieval.from_pretrained(checkpoint_path,
                                                    query_tokenizer=query_tokenizer,
                                                    context_tokenizer=context_tokenizer,
                                                    )
                                                
    local_path = "./PreFLMR_ViT-G"
    model.save_pretrained(local_path)
    query_tokenizer.save_pretrained(f"{local_path}/query_tokenizer")
    context_tokenizer.save_pretrained(f"{local_path}/context_tokenizer")

    # Copy cpp extensions
    src = "./models/flmr/segmented_maxsim.cpp"
    dst = f"{local_path}/segmented_maxsim.cpp"

    shutil.copyfile(src, dst)