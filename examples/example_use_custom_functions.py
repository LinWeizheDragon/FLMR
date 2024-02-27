import os
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import ToPILImage
from transformers import AutoImageProcessor

from flmr import index_custom_collection
from flmr import create_searcher, search_custom_collection
from flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval


if __name__ == "__main__":
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

    # Create a toy document collection
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

    

    # Run indexing
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

    # Create toy query data
    num_queries = 2
    
    query_instructions = [f"instruction {i}" for i in range(num_queries)]
    query_texts = [f"{query_instructions[i]} : query {i}" for i in range(num_queries)]
    query_images = torch.zeros(num_queries, 3, 224, 224)
    query_encoding = query_tokenizer(query_texts)
    query_pixel_values = image_processor(query_images, return_tensors="pt")['pixel_values']
    
    inputs = dict(
        input_ids=query_encoding['input_ids'],
        attention_mask=query_encoding['attention_mask'],
        pixel_values=query_pixel_values,
    )

    # Run model query encoding
    res = model.query(**inputs)

    query_embeddings = res.late_interaction_output

    queries = {i: query_texts[i] for i in range(num_queries)}

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