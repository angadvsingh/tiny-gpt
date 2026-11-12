# tiny-gpt

A small GPT-style language model built from scratch in Python/PyTorch and trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

## Progress

- [x] **Step 1: Data + tokenizer.** Byte-level BPE tokenizer written from scratch, with tests; dataset tokenized to binary files.
- [ ] Step 2: Transformer model
- [ ] Step 3: Training loop
- [ ] Step 4: Evaluation + sampling
- [ ] Step 5: Profiling, optimization, C++ tokenizer
- [ ] Step 6: Experiments + write-up

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Step 1: Data and tokenizer

```bash
pytest -q                              # run tokenizer tests
python scripts/download_data.py        # download TinyStories (~2.2 GB)
python scripts/train_tokenizer.py      # learn a 4096-token BPE vocab
python scripts/prepare_data.py         # tokenize text into data/train.bin, data/val.bin
```

## Layout

```
tiny_gpt/        library code (tokenizer, later the model and training)
scripts/         command-line entry points
tests/           unit tests
data/            downloaded and generated data (not committed)
```
