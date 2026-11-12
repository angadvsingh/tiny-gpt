"""
Download the TinyStories dataset (V2, the GPT-4-generated version).

    python scripts/download_data.py

Files go into data/raw/:
  - TinyStoriesV2-GPT4-valid.txt   (~22 MB)  held out, used to measure progress
  - TinyStoriesV2-GPT4-train.txt   (~2.2 GB) used to train

Stories are separated by the line "<|endoftext|>".

What is TinyStories? About 2 million short stories written by GPT-4 using only
words a 3-4 year old would know. The researchers who made it showed that even
very small models (a few million parameters) can learn to write coherent,
grammatical stories from it -- which is exactly what we want on a laptop.
"""

from pathlib import Path

# Hugging Face Hub hosts most public ML datasets and models. hf_hub_download
# fetches one file and skips it if you already downloaded it.
from huggingface_hub import hf_hub_download

REPO = "roneneldan/TinyStories"
FILES = ["TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-train.txt"]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    # parents=True creates data/ too if needed; exist_ok=True: no error if it exists.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        print(f"Downloading {name} ...")
        path = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset", local_dir=OUT_DIR)
        size_mb = Path(path).stat().st_size / 1e6  # file size in megabytes
        print(f"  -> {path} ({size_mb:,.0f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
