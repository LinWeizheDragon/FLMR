from datasets import DatasetDict
from huggingface_hub import HfApi

if __name__ == '__main__':
    api = HfApi()

    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo
    repo_id="LinWeizheDragon/PreFLMR_ViT-L"
    folder_path="./PreFLMR_ViT-L"

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
    )
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
    )

    
