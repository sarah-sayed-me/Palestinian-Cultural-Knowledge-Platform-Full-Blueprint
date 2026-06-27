from huggingface_hub import HfApi

def publish(local_dir: str, path_in_repo: str, commit_message: str):
    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id="palestinian-kg/palestinian-cultural-knowledge",
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )

if __name__ == "__main__":
    publish(
        local_dir="data/processed/hf/wikipedia_ar",
        path_in_repo="data/wikipedia_ar",
        commit_message="Add Arabic Wikipedia corpus v0.1.0 (484 docs, ~444K words)",
    )