"""
Turn the raw text into one long array of token ids, saved as a binary file.

    python scripts/prepare_data.py                    # 400 MB of training text
    python scripts/prepare_data.py --train-mb 0       # 0 = use the whole file

Output:
  data/train.bin, data/val.bin   token ids as uint16 (2 bytes each)
  data/meta.json                 vocab size + token counts

Why a binary file?
------------------
During training we grab random windows of, say, 256 tokens millions of times.
Re-tokenizing text each time would be painfully slow. Instead we tokenize
ONCE and store the ids in a flat file. Later we "memory-map" it: numpy treats
the file on disk like an array, so grabbing a window is basically free, even
if the file is bigger than your RAM.

Why uint16?
-----------
uint16 = "unsigned 16-bit integer", which holds whole numbers 0..65535. That's
enough for our vocab (4096), and it uses half the space of a normal 32-bit int.

Why train AND val files?
------------------------
The model learns from train.bin. val.bin ("validation") is text the model
never trains on. If loss keeps improving on train but gets worse on val, the
model is memorizing instead of learning general patterns ("overfitting").

Encoding is the slow part (pure Python), so we split the stories across all
CPU cores with multiprocessing.
"""

from __future__ import annotations

import argparse  # parses command-line options like --train-mb 100
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np  # fast arrays of numbers; the foundation of most ML code

from tiny_gpt.tokenizer import EOT, BPETokenizer

# Path(__file__) is this script's own location. .resolve().parent.parent walks
# up from scripts/prepare_data.py to the project root, so the script works no
# matter which folder you run it from.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"  # the `/` operator joins Path pieces
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Multiprocessing setup
#
# A Pool starts several separate Python processes ("workers"), one per CPU
# core, and hands each a batch of stories. Separate processes don't share
# memory, so each worker loads its own copy of the tokenizer once, in
# _init_worker, and keeps it in this global variable.
# ---------------------------------------------------------------------------
_tok: BPETokenizer | None = None


def _init_worker(tok_path: str) -> None:
    global _tok  # `global` = assign to the module-level _tok, not a new local
    _tok = BPETokenizer.load(tok_path)


def _encode_batch(stories: list[str]) -> np.ndarray:
    """Runs inside a worker: encode a batch of stories into one numpy array."""
    ids: list[int] = []
    for s in stories:
        ids.extend(_tok.encode(s))
        ids.append(_tok.eot_id)  # mark the end of each story
    # Convert the Python list into a compact numpy array of 2-byte integers.
    return np.array(ids, dtype=np.uint16)


def encode_file(src: Path, dst: Path, tok_path: Path, max_mb: float) -> int:
    """Tokenize the stories in `src` and write their ids to `dst`. Returns token count."""
    # `with open(...) as f:` opens the file and closes it automatically after.
    with open(src, encoding="utf-8") as f:
        # f.read(n) reads n characters; f.read() reads everything.
        text = f.read(int(max_mb * 1e6)) if max_mb > 0 else f.read()
    if max_mb > 0:
        # We probably cut off mid-story; drop everything after the last EOT.
        text = text[: text.rfind(EOT)]

    # Split into individual stories, trim whitespace, and drop empty ones.
    stories = [s.strip() for s in text.split(EOT)]
    stories = [s for s in stories if s]
    print(f"{src.name}: {len(stories):,} stories, {len(text) / 1e6:,.0f} MB")

    # Group stories into batches of 2000. Sending each worker a batch, rather
    # than one story at a time, cuts the overhead of passing data between processes.
    batch = 2000
    batches = [stories[i : i + batch] for i in range(0, len(stories), batch)]

    total = 0
    t0 = time.time()
    # "wb" = write binary. Pool(n_workers, init_function, init_args).
    with open(dst, "wb") as out, Pool(os.cpu_count(), _init_worker, (str(tok_path),)) as pool:
        # pool.imap sends batches to workers and gives results back IN ORDER,
        # so the stories end up in the file in their original order.
        for i, arr in enumerate(pool.imap(_encode_batch, batches)):
            out.write(arr.tobytes())  # raw bytes: 2 per token, nothing else
            total += len(arr)
            if i % 20 == 0 or i == len(batches) - 1:
                print(f"  {i + 1}/{len(batches)} batches, {total:,} tokens, {time.time() - t0:.0f}s")

    # chars/token tells you how well the tokenizer compresses (~4 is typical).
    print(f"  -> {dst} ({total:,} tokens, {len(text) / max(total, 1):.2f} chars/token)")
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-mb", type=float, default=400, help="MB of training text to use (0 = all)")
    args = p.parse_args()  # args.train_mb now holds the value (dashes become underscores)

    tok_path = DATA / "tokenizer.json"
    tok = BPETokenizer.load(tok_path)
    assert tok.vocab_size <= 2**16, "vocab too large for uint16"

    # Validation file is small, so use all of it (max_mb=0).
    n_val = encode_file(RAW / "TinyStoriesV2-GPT4-valid.txt", DATA / "val.bin", tok_path, 0)
    n_train = encode_file(RAW / "TinyStoriesV2-GPT4-train.txt", DATA / "train.bin", tok_path, args.train_mb)

    # Save a small summary the training code will read later.
    meta = {"vocab_size": tok.vocab_size, "eot_id": tok.eot_id, "train_tokens": n_train, "val_tokens": n_val}
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))

    # Sanity check: read back the start of val.bin and decode it. If this
    # prints normal-looking text, the whole pipeline works end to end.
    # np.memmap opens the file as an array WITHOUT loading it all into memory.
    val = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    print("\nFirst 60 tokens of val.bin, decoded:")
    print(repr(tok.decode(val[:60].tolist())))


# This block only runs when you execute the file directly
# (`python scripts/prepare_data.py`), not when another file imports it.
# It's REQUIRED here: on macOS, multiprocessing re-imports this file in every
# worker, and without this guard each worker would start its own Pool forever.
if __name__ == "__main__":
    main()
