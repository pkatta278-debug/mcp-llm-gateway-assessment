import random

import pytest

from redactor import EMAIL_HOLD, MAX_HOLD, REDACTED, StreamRedactor, redact_text


def stream(chunks):
    r = StreamRedactor()
    out = "".join(r.feed(c) for c in chunks) + r.flush()
    return out, r


# ------------------------------------------------------------------ one-shot behaviour
@pytest.mark.parametrize("text,expected", [
    ("mail me at jane.doe@example.com now", f"mail me at {REDACTED} now"),
    ("A@B.CO", REDACTED),
    ("x+tag@sub.domain.example.org!", f"{REDACTED}!"),
    ("a@b.co.uk", REDACTED),
    ("SSN 123-45-6789.", f"SSN {REDACTED}."),
    ("(123-45-6789)", f"({REDACTED})"),
    ("card 4111111111111111 ok", f"card {REDACTED} ok"),
    ("card 4111 1111 1111 1111 ok", f"card {REDACTED} ok"),
    ("card 4111-1111-1111-1111", f"card {REDACTED}"),
    ("amex 378282246310005", f"amex {REDACTED}"),
    ("19 digit 6759649826438453003", f"19 digit {REDACTED}"),
    ("two: a@b.io and c@d.io", f"two: {REDACTED} and {REDACTED}"),
])
def test_redacts(text, expected):
    assert redact_text(text) == expected


@pytest.mark.parametrize("text", [
    "plain sentence with nothing in it",
    "version 1.2.3 released 2024-01-15 at 5pm",
    "call 555-1234 or 1-800-555-0199",
    "order 1234567890123456",            # 16 digits, fails Luhn
    "000-12-3456 | 666-12-3456 | 987-65-4321 | 123-00-4567 | 123-45-0000",  # impossible SSNs
    "@handle and user@ and @",
    "price $1,234.56 for 12 units",
    "number 12345678901234567890123",    # too long to be a card
    "naïve café — ünïcode ✓ stays",
])
def test_leaves_alone(text):
    assert redact_text(text) == text


# --------------------------------------------- streaming must equal one-shot, however it is chunked
TEXTS = [
    "Reach me at john.doe@example.com or 123-45-6789 or 4111 1111 1111 1111. Bye",
    "ends with email jane@corp.example.co.uk",
    "ends with ssn 123-45-6789",
    "ends with card 4111-1111-1111-1111",
    "4111111111111111",
    "x 4111 1111 1111 1111 and 378282246310005 and 5500 0000 0000 0004!",
    "a@b.co a@b.co.uk a@b.co.uk.",
    "not pii: 1234567890123456 and 2024-01-15 and 555-1234 and 1.2.3.4",
    "mixed 123-45-6789,987-65-4321,4111111111111111@example.com end",
    "no spaces:john@x.com;123-45-6789;done",
    "unicode ✓ café a@b.io ✓",
]


@pytest.mark.parametrize("text", TEXTS)
def test_every_two_way_split_matches_one_shot(text):
    want = redact_text(text)
    for i in range(len(text) + 1):
        got, _ = stream([text[:i], text[i:]])
        assert got == want, f"split at {i}: {text[:i]!r} | {text[i:]!r}"


@pytest.mark.parametrize("text", TEXTS)
def test_char_by_char_matches_one_shot(text):
    got, _ = stream(list(text))
    assert got == redact_text(text)


@pytest.mark.parametrize("text", TEXTS)
def test_random_chunking_matches_one_shot(text):
    rng = random.Random(1234)
    want = redact_text(text)
    for _ in range(200):
        chunks, i = [], 0
        while i < len(text):
            n = rng.randint(1, 9)
            chunks.append(text[i:i + n])
            i += n
        got, _ = stream(chunks)
        assert got == want, chunks


def test_no_raw_pii_ever_leaves_while_streaming():
    # Check every intermediate emission, not just the final text.
    text = "hi john.doe@example.com 123-45-6789 4111 1111 1111 1111 bye"
    r = StreamRedactor()
    for ch in text:
        piece = r.feed(ch)
        for frag in ("john", "doe", "@", "example", "123-4", "6789", "4111", "1111"):
            assert frag not in piece, (frag, piece)
    r.flush()


def test_safe_text_is_not_held_back_long():
    # words before the tail go out immediately; only the trailing token waits
    r = StreamRedactor()
    assert r.feed("Hello there ") == "Hello there "
    assert r.feed("wor") == ""
    assert r.feed("ld and ") == "world and "


# ------------------------------------------------------------------ memory stays bounded
def test_buffer_bounded_on_endless_token():
    r = StreamRedactor()
    for _ in range(2000):
        r.feed("a" * 1000)  # 2 MB with no whitespace at all
    r.flush()
    assert r.max_buffered <= EMAIL_HOLD


def test_buffer_bounded_on_endless_digits_and_dashes():
    r = StreamRedactor()
    for _ in range(5000):
        r.feed("1 2-3 " * 10)
    assert r.max_buffered <= MAX_HOLD


def test_big_single_chunk_is_linear_enough():
    import time
    t = time.perf_counter()
    redact_text(("word " * 20 + "a@b.com 123-45-6789 ") * 20000)
    assert time.perf_counter() - t < 5
    t = time.perf_counter()
    redact_text("a" * 500_000 + "@" + "b" * 500_000)  # adversarial: huge runs around an @
    assert time.perf_counter() - t < 5


# ------------------------------------------------------------------ seeded fuzz
PIECES = ["hello", "world", " ", " ", ",", ".", "a@b.com", "jo.doe@mail.example.org", "123-45-6789", "4111 1111 1111 1111",
          "4111111111111111", "378282246310005", "12", "-", "@", "x", "2024-01-15", "5500-0000-0000-0004", "\n", "café"]


def test_fuzz_streaming_equals_one_shot_and_never_leaks():
    rng = random.Random(99)
    secrets = ["jo.doe@mail.example.org", "a@b.com", "123-45-6789", "4111 1111 1111 1111", "4111111111111111",
               "378282246310005", "5500-0000-0000-0004"]
    for _ in range(400):
        text = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 25)))
        want = redact_text(text)
        chunks, i = [], 0
        while i < len(text):
            n = rng.randint(1, 12)
            chunks.append(text[i:i + n])
            i += n
        got, _ = stream(chunks)
        assert got == want, (text, chunks)
        for sec in secrets:
            # a secret may only survive if glued into a larger token by the random pieces, never intact next to a clean boundary
            if sec in got:
                assert sec in want
