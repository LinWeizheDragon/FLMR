from transformers import AutoConfig, AutoModel, AutoImageProcessor, AutoTokenizer
import torch

if __name__ == '__main__':
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