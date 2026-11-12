"""
A byte-level BPE (Byte Pair Encoding) tokenizer, written from scratch.

The big idea
------------
Models only understand numbers, so we need a way to turn text into a list of
integers ("token ids") and back again.

1. Start with the 256 possible byte values as our base vocabulary. Any text,
   in any language, is just a sequence of UTF-8 bytes, so with these 256
   tokens we can already represent *everything*. Nothing is ever "unknown".

2. Look at a big pile of training text and find the pair of adjacent tokens
   that appears most often (e.g. "t" followed by "h"). Create a brand new
   token for that pair ("th") and replace every occurrence of the pair with it.

3. Repeat. Next round "th" + "e" might become "the". After a few thousand
   rounds, common words are single tokens and rare words are split into a
   few common pieces.

Each round is called a "merge". The ordered list of merges IS the tokenizer:
to encode new text, you replay the same merges in the same order.

This is the same family of algorithm GPT-2/3/4 use. Ours is simpler and slower
(pure Python), but the logic is the same.

A tiny worked example
---------------------
Text: "aaab aab"
  bytes:            a a a b _ a a b       (ids 97 97 97 98 32 97 97 98)
  most common pair: (a, a)  -> new token 256 = "aa"
  after merge 1:    aa a b _ aa b
  most common pair: (aa, b) -> new token 257 = "aab"   (appears once... etc.)
Encoding later replays: first merge (a,a), then (aa,b), in that order.
"""

# Lets us write type hints like `list[int] | None` and refer to a class inside
# its own definition. Type hints (the `: int`, `-> str` bits) don't change how
# the code runs -- they're documentation that editors and tools can check.
from __future__ import annotations

import json
import re  # regular expressions: patterns for finding/splitting text
from collections import Counter, defaultdict
from pathlib import Path  # a nicer way to handle file paths than plain strings

# The special token that marks "end of one story, start of the next".
# It never gets split into bytes: it always becomes exactly one id.
# Why have it at all? When the model trains on a stream of many stories glued
# together, this token tells it "a new, unrelated story starts here". Later,
# when the model *generates* this token, we know it thinks the story is over.
EOT = "<|endoftext|>"

# Pre-tokenization: before running BPE, we chop text into word-like chunks and
# only allow merges *inside* a chunk. Without this, BPE would happily create
# tokens like "dog." or "the cat" that glue punctuation and words together,
# which wastes vocabulary on combinations instead of reusable pieces.
#
# Reading the pattern (the `|` means "or"; the first alternative that matches wins):
#   's|'t|'re|...    contractions: "don't" -> "don", "'t"
#    ?[A-Za-z]+      an optional space, then letters:  " dog"
#    ?[0-9]+         an optional space, then digits:   " 42"
#    ?[^\sA-Za-z0-9]+  an optional space, then punctuation/other symbols: ",", " 🐶"
#   \s+(?!\S)        whitespace, but leave the last space to attach to the next word
#   \s+              any remaining whitespace (e.g. "\n\n")
#
# The leading optional space means a word keeps its preceding space, so " dog"
# and "dog" are different tokens. That's how GPT-2 does it, and it means
# decode() can rebuild the original spacing perfectly.
#
# re.compile() turns the pattern into an object once, so we don't re-parse it
# every time we use it. The r"""...""" is a "raw string": backslashes like \s
# are kept as-is instead of being treated as escape characters.
CHUNK_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    Replace every occurrence of `pair` in `ids` with `new_id`.

    Example: _merge([1, 2, 3, 1, 2], pair=(1, 2), new_id=99) -> [99, 3, 99]

    (A leading underscore in a name is a Python convention for "internal
    helper, not meant to be used from outside this file".)
    """
    out = []
    i = 0
    while i < len(ids):
        # If ids[i] and ids[i+1] form the pair, emit the new id and skip both.
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        # __init__ runs when you create a tokenizer: `tok = BPETokenizer()`.
        # `self` is the tokenizer object itself; `self.x = ...` stores data on it.

        # merges[(a, b)] = new_id. Python dicts remember insertion order, so
        # this also records the ORDER merges were learned, which matters a lot.
        self.merges: dict[tuple[int, int], int] = {}

        # vocab[id] = the raw bytes that token stands for.
        # `{i: bytes([i]) for i in range(256)}` is a "dict comprehension": a
        # compact loop that builds {0: b'\x00', 1: b'\x01', ..., 97: b'a', ...}.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        self.eot_id: int = 256  # placeholder; set properly after training

        # Cache: once we've encoded a chunk like " the", remember the answer.
        # Stories reuse the same words constantly, so this makes encoding
        # many times faster.
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Learn merges from `text` until the vocabulary has `vocab_size` tokens
        (256 bytes + merges + 1 special EOT token).
        """
        num_merges = vocab_size - 256 - 1
        assert num_merges >= 0, "vocab_size must be at least 257"

        # Step 1: count how often each chunk appears.
        # "the" appears millions of times in TinyStories, but we only need to
        # process it once and remember its count. This is what makes training
        # fast enough. A Counter is a dict that counts things:
        #   Counter(["a", "b", "a"]) -> {"a": 2, "b": 1}
        chunk_counts: Counter[str] = Counter()
        for story in text.split(EOT):
            chunk_counts.update(CHUNK_PATTERN.findall(story))

        # Each unique chunk becomes a list of byte ids, plus its count.
        # " dog".encode("utf-8") -> b' dog'; list(b' dog') -> [32, 100, 111, 103]
        words: list[list[int]] = [list(c.encode("utf-8")) for c in chunk_counts]
        counts: list[int] = list(chunk_counts.values())

        # Step 2: count every adjacent pair across all words (weighted by how
        # often the word appears), and remember WHICH words contain each pair
        # so we can find them quickly later.
        #
        # `zip(w, w[1:])` is the standard Python trick for adjacent pairs:
        #   w     = [32, 100, 111, 103]
        #   w[1:] = [100, 111, 103]          (everything from index 1 on)
        #   zip   -> (32,100), (100,111), (111,103)
        #
        # defaultdict(set) is a dict that auto-creates an empty set the first
        # time you touch a missing key, so `where[pair].add(...)` just works.
        pair_counts: Counter[tuple[int, int]] = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        # enumerate() gives (index, item); zip() walks two lists side by side.
        for w_idx, (w, c) in enumerate(zip(words, counts)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += c
                where[pair].add(w_idx)

        # Reset in case train() is called twice on the same object.
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Step 3: repeatedly merge the most common pair.
        for step in range(num_merges):
            if not pair_counts:
                break  # nothing left to merge (tiny training text)

            # Find the most frequent pair. `key=` tells max() what to compare:
            # here a tuple (count, pair). Tuples compare element by element,
            # so the highest count wins, and ties are broken by the pair
            # itself. That makes training deterministic: same input -> same
            # tokenizer, every time. (`lambda p: ...` is a tiny inline function.)
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] < 2:
                break  # merging something that appears once is pointless

            new_id = 256 + step
            self.merges[best] = new_id
            # The new token's bytes are just its two parts glued together.
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            # Only the words that contain `best` change. Update pair counts
            # incrementally for just those words instead of recounting
            # everything -- this is the difference between seconds and hours.
            # (`list(...)` makes a copy, because we modify `where` inside the loop.)
            for w_idx in list(where[best]):
                w, c = words[w_idx], counts[w_idx]
                # 1) Remove this word's old pairs from the counts...
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= c
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where[pair].discard(w_idx)
                # 2) ...merge the pair inside the word...
                w = _merge(w, best, new_id)
                words[w_idx] = w
                # 3) ...and add the word's new pairs back in.
                for pair in zip(w, w[1:]):
                    pair_counts[pair] += c
                    where[pair].add(w_idx)
            where.pop(best, None)  # nothing contains `best` any more

            # Print progress every 250 merges. `!r` in an f-string shows the
            # "repr" form, e.g. b' the', which makes spaces visible.
            if verbose and (step % 250 == 0 or step == num_merges - 1):
                print(f"merge {step + 1}/{num_merges}: {self.vocab[new_id]!r}")

        # The special token gets the last id.
        self.eot_id = 256 + len(self.merges)
        self.vocab[self.eot_id] = EOT.encode("utf-8")
        self._cache = {}

    # @property lets you write `tok.vocab_size` (no parentheses) even though
    # it's computed by a function each time.
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one pre-tokenized chunk by replaying merges in learned order."""
        cached = self._cache.get(chunk)  # .get() returns None if missing
        if cached is not None:
            return cached

        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Of all adjacent pairs, find the one that was learned EARLIEST
            # (lowest merge id). Pairs never learned get infinity, so they
            # lose. We must apply merges in training order because later
            # merges were built on top of earlier ones: "the" can only form
            # after "th" exists.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no learned merge applies any more; we're done
            ids = _merge(ids, pair, self.merges[pair])

        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids. EOT markers become the EOT id."""
        out: list[int] = []
        # Split on the special token first so it never gets broken into bytes.
        parts = text.split(EOT)
        for i, part in enumerate(parts):
            for chunk in CHUNK_PATTERN.findall(part):
                out.extend(self._encode_chunk(chunk))  # extend = append many
            if i < len(parts) - 1:  # an EOT sat between this part and the next
                out.append(self.eot_id)
        return out

    def decode(self, ids: list[int]) -> str:
        """List of token ids -> text."""
        # Look up each id's bytes and glue them all together: b"" is an empty
        # bytes object, and .join() concatenates the pieces.
        raw = b"".join(self.vocab[i] for i in ids)
        # errors="replace": a model might generate a byte sequence that isn't
        # valid UTF-8 (e.g. half of an emoji). Don't crash; show a "?" box.
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Saving / loading
    # ------------------------------------------------------------------ #
    # We only need to save the merges (in order) and the special token id.
    # The full vocab can be rebuilt from those, so storing it would be redundant.
    def save(self, path: str | Path) -> None:
        data = {
            "type": "byte-bpe",
            # JSON can't store tuples as dict keys, so save merges as a list of
            # [a, b] pairs. Their position in the list gives their order/id.
            "merges": [[a, b] for (a, b) in self.merges],
            "special_tokens": {EOT: self.eot_id},
        }
        Path(path).write_text(json.dumps(data))

    # @classmethod: called on the class itself, not on an existing object:
    #   tok = BPETokenizer.load("data/tokenizer.json")
    # `cls` is the class (BPETokenizer), so `cls()` creates a fresh tokenizer.
    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text())
        tok = cls()
        # Replay the merges in order to rebuild merges and vocab exactly.
        for step, (a, b) in enumerate(data["merges"]):
            new_id = 256 + step
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.eot_id = data["special_tokens"][EOT]
        tok.vocab[tok.eot_id] = EOT.encode("utf-8")
        return tok
