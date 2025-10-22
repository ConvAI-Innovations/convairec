"""
Upload model and code to Hugging Face Hub
"""

from huggingface_hub import HfApi, create_repo, upload_folder
import os

# Configuration
HF_TOKEN = "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"
USERNAME = "convaiinnovations"
REPO_NAME = "convrec-face-recognition"
REPO_TYPE = "model"  # Can be "model", "dataset", or "space"

def upload_to_huggingface():
    """Upload the entire folder to Hugging Face"""

    # Initialize API
    api = HfApi()

    # Create repository ID
    repo_id = f"{USERNAME}/{REPO_NAME}"

    try:
        # Create repository if it doesn't exist
        create_repo(
            repo_id=repo_id,
            token=HF_TOKEN,
            repo_type=REPO_TYPE,
            exist_ok=True,
            private=False  # Set to True for private repo
        )
        print(f"Repository created/verified: {repo_id}")

    except Exception as e:
        print(f"Repository creation info: {e}")

    # Upload the entire folder
    try:
        print(f"Uploading files to {repo_id}...")

        # Get current directory
        folder_path = os.path.dirname(os.path.abspath(__file__))

        # Upload folder
        upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
            ignore_patterns=["*.pyc", "__pycache__", ".git*", "*.log", "upload_to_hf.py"]
        )

        print(f"Successfully uploaded to: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"Error uploading: {e}")

if __name__ == "__main__":
    upload_to_huggingface()