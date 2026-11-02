"""Streaming PII redactor.

The hard part of redacting a stream is that a secret can be split across chunks
("john.d" + "oe@exam" + "ple.com"). The trick used here:

  * keep a small buffer of text we have NOT sent yet
  * on every chunk: append, redact whatever is already a complete match, then send
    everything except the trailing piece that could still turn into a match
  * at end of stream, flush the rest

So memory is bounded by the longest possible match (a few hundred chars), and the only
delay is the last "word" being held back until the next chunk shows what it is.

Patterns: emails, US SSNs, credit card numbers (13-19 digits, Luhn checked so order
numbers and other long digit strings are left alone).
"""
from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# Local part and domain labels are length-bounded so the regexes can't go quadratic on junk.
EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){0,8}\.[A-Za-z]{2,24}"
)
# digits with optional single space/dash between them, not glued to other digits
CARD_RE = re.compile(r"(?<!\d)\d(?:[ \-]?\d){12,18}(?!\d)")
# skips impossible SSNs (area 000/666/9xx, group 00, serial 0000)
SSN_RE = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")

_EMAIL_TAIL_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-@")
_DIGIT_TAIL_CHARS = frozenset("0123456789 -")
EMAIL_HOLD = 320  # 64 + @ + 253 domain, rounded up
DIGIT_HOLD = 40   # 19 digits + 18 separators = 37 is the longest card candidate
MAX_HOLD = max(EMAIL_HOLD, DIGIT_HOLD)


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _sub(rx: re.Pattern, buf: str, cut: int, validate=None) -> tuple[str, int]:
    """Replace matches with [REDACTED], but only matches that end at or before `cut`.

    `cut` is the boundary between text that is safe to send and the held-back tail. A match
    that reaches into the tail is left alone, because more text could still change it
    (a@b.co -> a@b.co.uk, 16 digits -> 19 digits, digits -> digits@email), and `cut` is
    pulled back to the start of that match so no piece of it is sent early.
    Returns the new buffer and the cut position inside it.
    """
    out: list[str] = []
    last = 0
    shift = 0
    for m in rx.finditer(buf):
        s, e = m.span()
        if e > cut:
            cut = min(cut, s)
            break  # everything after this is in the tail too
        if validate is not None and not validate(m.group()):
            continue
        out.append(buf[last:s])
        out.append(REDACTED)
        last = e
        shift += len(REDACTED) - (e - s)
    if not out:
        return buf, cut
    out.append(buf[last:])
    return "".join(out), cut + shift


def _sub_cards(buf: str, cut: int) -> tuple[str, int]:
    """Like _sub, but for card numbers: a candidate span is only redacted if some run of 13-19
    of its digits passes the Luhn check. Longest valid run wins, and when a candidate fails
    we keep looking from the next position, so a real card sitting right behind some other
    digits (an SSN, a phone number) is still found."""
    out: list[str] = []
    last = 0
    shift = 0
    pos = 0
    while True:
        m = CARD_RE.search(buf, pos)
        if not m:
            break
        s, e = m.span()
        if e > cut:
            cut = min(cut, s)
            break
        idxs = [i for i in range(s, e) if buf[i].isdigit()]
        found = None
        for n in range(min(len(idxs), 19), 12, -1):
            end = idxs[n - 1] + 1
            if end < len(buf) and buf[end].isdigit():
                continue  # would cut a digit run in half
            if luhn_ok("".join(buf[i] for i in idxs[:n])):
                found = end
                break
        if found is None:
            pos = s + 1
            continue
        out.append(buf[last:s])
        out.append(REDACTED)
        last = found
        shift += len(REDACTED) - (found - s)
        pos = found
    if not out:
        return buf, cut
    out.append(buf[last:])
    return "".join(out), cut + shift


def _tail_start(buf: str, allowed: frozenset, cap: int) -> int:
    """Index where the trailing run of `allowed` chars starts (looking back at most `cap`)."""
    i = len(buf)
    lim = max(0, i - cap)
    while i > lim and buf[i - 1] in allowed:
        i -= 1
    return i


def _hold_from(buf: str) -> int:
    n = len(buf)
    email_start = _tail_start(buf, _EMAIL_TAIL_CHARS, EMAIL_HOLD)
    digit_start = _tail_start(buf, _DIGIT_TAIL_CHARS, DIGIT_HOLD)
    while digit_start < n and not buf[digit_start].isdigit():  # a candidate starts with a digit
        digit_start += 1
    return min(email_start, digit_start)


def _redact(buf: str, final: bool) -> tuple[str, int]:
    """Redact `buf`. Returns (text, cut): text[:cut] is safe to send, text[cut:] must wait."""
    cut = len(buf) if final else _hold_from(buf)
    # cheap prefilters first: most chunks have no '@', no dash, no digits
    if "@" in buf:
        buf, cut = _sub(EMAIL_RE, buf, cut)  # emails first so digits inside an address go with it
    if any(c.isdigit() for c in buf):
        # SSNs before cards: a card-length window can accidentally swallow the end of an SSN
        # plus the start of the next number, which would leave half the SSN visible.
        if "-" in buf:
            buf, cut = _sub(SSN_RE, buf, cut)
        buf, cut = _sub_cards(buf, cut)
    return buf, cut


class StreamRedactor:
    """One per output stream (per choice). feed() chunks in, safe text out."""

    def __init__(self) -> None:
        self._buf = ""
        self.max_buffered = 0  # high-water mark, used by tests/bench to prove memory is bounded

    def feed(self, text: str) -> str:
        buf = self._buf + text
        if not buf:
            return ""
        buf, cut = _redact(buf, final=False)
        self._buf = buf[cut:]
        self.max_buffered = max(self.max_buffered, len(self._buf))
        return buf[:cut]

    def flush(self) -> str:
        buf, self._buf = self._buf, ""
        return _redact(buf, final=True)[0] if buf else ""


def redact_text(text: str) -> str:
    """One-shot redaction of a complete string (used for non-streaming responses)."""
    return _redact(text, final=True)[0]
