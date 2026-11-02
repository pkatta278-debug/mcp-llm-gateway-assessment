"""Rough numbers for the redactor: throughput, per-chunk cost, and how big the buffer ever gets.

    python bench.py
"""
import random
import time

from redactor import StreamRedactor

random.seed(1)
WORDS = ["the", "quick", "model", "said", "that", "you", "can", "email", "x", "42", "data", "token", "ok"]
PII = ["jane.doe@example.com", "123-45-6789", "4111 1111 1111 1111"]


def make_tokens(n):
    out = []
    for i in range(n):
        out.append(" " + random.choice(PII) if i % 500 == 0 else " " + random.choice(WORDS))
    return out


def run(chunk_chars):
    toks = make_tokens(400_000)
    text = "".join(toks)
    chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    r = StreamRedactor()
    t = time.perf_counter()
    sent = 0
    for c in chunks:
        sent += len(r.feed(c))
    sent += len(r.flush())
    dt = time.perf_counter() - t
    mb = len(text) / 1e6
    print(f"chunks of ~{chunk_chars:>3} chars: {mb:.1f} MB in {dt:.2f}s = {mb/dt:5.1f} MB/s, "
          f"{dt/len(chunks)*1e6:6.1f} us per chunk, max buffered {r.max_buffered} chars")


if __name__ == "__main__":
    for n in (4, 16, 64, 1024):
        run(n)
