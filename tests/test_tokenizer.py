"""
Tests for the BPE tokenizer.

Run with:  pytest -q

Why these tests matter: tokenizer bugs are silent. If encode/decode quietly
garbles text, the model still trains -- just on garbage -- and you'd only
notice as "the loss is a bit worse than expected". These catch that early.

How pytest works:
  - It finds files named test_*.py and runs every function named test_*.
  - Inside a test, `assert <condition>` checks something. If it's False, the
    test fails and pytest shows you the values involved.
  - A test passes if it finishes without a failed assert or an error.
"""

import random

import pytest

from tiny_gpt.tokenizer import EOT, BPETokenizer

# A small, repetitive training text: two "stories" separated by EOT.
# `* 20` repeats a string 20 times, so common pairs show up often enough to merge.
SAMPLE = (
    "Once upon a time, there was a little dog named Max. Max loved to play "
    "in the park. One day, Max saw a big red ball. He ran to the ball and "
    "played all day. The end."
) * 20 + EOT + (
    "Lily had a cat. The cat was very soft. Lily and the cat played "
    "together every day. They were best friends."
) * 20


# A "fixture" is shared setup. Any test with a parameter named `tok` gets the
# value this function returns. scope="module" means it's built once for the
# whole file instead of once per test, since training takes a moment.
@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    t = BPETokenizer()
    t.train(SAMPLE, vocab_size=400)
    return t


def test_round_trip_training_text(tok):
    # The most important property: decode(encode(x)) == x, exactly.
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


# parametrize runs this one test several times, once per string in the list.
# Each run shows up separately in the results, so you see exactly which input failed.
@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "Hello, world!",
        "  multiple   spaces\n\nand newlines\t tabs ",
        "Words the tokenizer never saw: xylophone quixotic 1234567",
        "Unicode works too: café, naïve, 日本語, emoji 🐶🐱",
    ],
)
def test_round_trip_unseen_text(tok, text):
    # Byte-level BPE can encode ANY string, even ones it never trained on,
    # because the worst case is falling back to individual bytes.
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_random_bytes(tok):
    # Fuzz test: throw 200 random strings at it. Random inputs find edge cases
    # a human wouldn't think to write. Seeding the random generator (0) makes
    # the "random" strings the same every run, so failures are reproducible.
    rng = random.Random(0)
    alphabet = "abcXYZ019 .,!?'\n\té🐶"
    for _ in range(200):  # `_` = a loop variable we don't use
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 50)))
        assert tok.decode(tok.encode(s)) == s


def test_compression(tok):
    # The whole point of BPE: far fewer tokens than raw bytes.
    n_bytes = len(SAMPLE.encode("utf-8"))
    n_tokens = len(tok.encode(SAMPLE))
    assert n_tokens < n_bytes / 2


def test_eot_is_single_token(tok):
    # The end-of-text marker must become exactly ONE id, never split into
    # the bytes "<", "|", "e", ... -- and it should be the last id in the vocab.
    ids = tok.encode("hi" + EOT + "bye")
    assert ids.count(tok.eot_id) == 1
    assert tok.eot_id == tok.vocab_size - 1


def test_vocab_size():
    t = BPETokenizer()
    t.train(SAMPLE, vocab_size=320)
    assert t.vocab_size == 320


def test_stops_early_on_tiny_text(tok):
    # SAMPLE is so small that it runs out of pairs appearing 2+ times before
    # reaching 400 tokens. Training should stop gracefully, not crash.
    assert tok.vocab_size <= 400
    assert tok.eot_id == tok.vocab_size - 1


def test_training_is_deterministic():
    # Same text in -> same merges out. Otherwise you could never reproduce a run.
    a, b = BPETokenizer(), BPETokenizer()
    a.train(SAMPLE, vocab_size=350)
    b.train(SAMPLE, vocab_size=350)
    assert a.merges == b.merges


def test_save_load(tok, tmp_path):
    # `tmp_path` is a built-in pytest fixture: a fresh temporary folder,
    # cleaned up automatically, so tests don't litter your project.
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.eot_id == tok.eot_id
    assert loaded.encode(SAMPLE) == tok.encode(SAMPLE)
