"""
Train our BPE tokenizer on a sample of TinyStories.

    python scripts/train_tokenizer.py                 # defaults
    python scripts/train_tokenizer.py --vocab-size 8192 --sample-mb 50

We don't need the full 2.2 GB to learn good merges: word statistics stabilize
quickly, and a 20 MB sample (~25k stories) is plenty for a 4096-token vocab.
Output: data/tokenizer.json

How big should the vocab be? It's a tradeoff:
  - Bigger vocab  -> each token covers more text, so sequences are shorter
                     (the model sees more story per step), BUT the model has a
                     bigger table of token embeddings to learn, and rare
                     tokens get few training examples each.
  - Smaller vocab -> more tokens per story, but each one is well-trained.
4096 is a sensible size for small models on simple text like TinyStories.
"""

import argparse
import time
from pathlib import Path

from tiny_gpt.tokenizer import EOT, BPETokenizer

ROOT = Path(__file__).resolve().parent.parent  # project root (see prepare_data.py)
TRAIN_FILE = ROOT / "data" / "raw" / "TinyStoriesV2-GPT4-train.txt"
OUT_FILE = ROOT / "data" / "tokenizer.json"


def main() -> None:
    # Command-line options, so you can experiment without editing code.
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--sample-mb", type=float, default=20)
    args = p.parse_args()

    # Read only the first N megabytes of the training file.
    with open(TRAIN_FILE, encoding="utf-8") as f:
        text = f.read(int(args.sample_mb * 1e6))
    text = text[: text.rfind(EOT)]  # don't cut a story in half
    print(f"Training on {len(text) / 1e6:.1f} MB of text, vocab size {args.vocab_size}")

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(text, vocab_size=args.vocab_size, verbose=True)
    print(f"Trained in {time.time() - t0:.1f}s")

    tok.save(OUT_FILE)
    print(f"Saved to {OUT_FILE}")

    # A quick look at what it learned: split one sentence and show each token.
    example = "Once upon a time, there was a little girl named Lily."
    ids = tok.encode(example)
    pieces = [tok.decode([i]) for i in ids]  # decode each id on its own
    print(f"\nExample: {example!r}")
    print(f"{len(example)} characters -> {len(ids)} tokens: {pieces}")

    # The longest tokens are usually whole common words or phrases.
    # sorted(..., key=len, reverse=True) sorts longest first; [:10] takes the top 10.
    longest = sorted(tok.vocab.values(), key=len, reverse=True)[:10]
    print("Some of the longest tokens:", [t.decode("utf-8", "replace") for t in longest])


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
