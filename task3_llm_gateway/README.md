# Task 3 - LLM Gateway streaming guardrail (PII redaction)

An OpenAI-compatible endpoint (`POST /v1/chat/completions`) that forwards to a provider and streams the
answer back with emails, SSNs and credit card numbers replaced by `[REDACTED]`. Python, Starlette + httpx.

```bash
pip install -r requirements.txt
python mock_provider.py &     # fake provider on :9100
python gateway.py &           # gateway on :8081  (UPSTREAM_URL / UPSTREAM_API_KEY / PORT env vars)
curl -sN localhost:8081/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}'

python -m pytest -q           # 80 tests (the integration ones start real HTTP servers, ~10s)
python bench.py               # redactor throughput + buffer high-water mark
```

## The main idea

PII can arrive split over several chunks (`"jo"`, `"hn@exa"`, `"mple.com"`), so you can't just run a regex on each
chunk. `StreamRedactor` keeps a small buffer of text it has not sent yet:

1. append the new chunk to the buffer
2. work out the **cut**: the start of the trailing piece that could still become a match (the last run of
   email characters, or the last run of digits / spaces / dashes)
3. redact complete matches that end before the cut, send everything before the cut, keep the rest
4. at end of stream (or when a choice finishes) flush what is left

So normal words go out right away, and only the last word waits for the next chunk. Memory is bounded by the longest
possible match: the buffer never goes above 320 chars (the benchmark shows ~30 on normal text). A match that reaches
into the held tail is not redacted yet, and the cut is pulled back to its start so no piece of it leaks early. That is
what stops `a@b.co` being redacted before `.uk` shows up.

## What gets redacted

| Type | Rule |
|---|---|
| Email | `local@domain.tld`, bounded lengths so the regex can't blow up on long junk |
| SSN | `123-45-6789`, skipping impossible ones (000, 666, 9xx area, 00 group, 0000 serial) |
| Card | 13-19 digits with optional single spaces/dashes, **only if it passes Luhn**, so order numbers and IDs are left alone |

Order is email, then SSN, then card (see the comment in `redactor.py` for why).

## Gateway behaviour

- Events are read line by line from the upstream and forwarded as soon as they are redacted. Nothing is collected
  into a big string, and the generator only reads the next line when the client has taken the last one, so a slow
  client can't make memory grow.
- One redactor per choice index (`n > 1` works). The held tail is sent as its own chunk **before** the chunk that has
  `finish_reason`, and before `[DONE]` if the upstream never sent a finish.
- Role / finish / tool-call chunks pass through; a chunk whose only content is still being held back is dropped.
- `logprobs` are set to null, since they repeat the generated tokens.
- **Fails closed:** if an upstream event can't be parsed, the stream ends with a generic error event instead of
  forwarding something that was never checked.
- Non-streaming requests are redacted too (whole message, one pass).
- Errors use one body shape, `{"error": {"type", "message"}}`, with 502 / 504 / 429 and a generic message.
  Provider error bodies, stack traces and URLs are logged-only, never sent to the client.
- The gateway uses its own provider key (`UPSTREAM_API_KEY`); the client's `Authorization` is not forwarded.

## What the tests prove

- Streaming output equals one-shot redaction for **every** 2-way split, char-by-char, 200 random chunkings of each
  sample, and a seeded fuzz of 400 random strings.
- No raw fragment (`john`, `123-4`, `4111`, ...) appears in any intermediate emission, checked on the actual wire,
  including when SSE frames are cut into 5-byte pieces.
- Latency: with an upstream that sends a token every 0.3s the first token reaches the client in under 0.2s, and
  token gaps are preserved instead of being batched.
- Memory: 2 MB with no whitespace at all keeps the buffer at or below 320 chars.

## Known limits

- Only the OpenAI chat-completions chunk shape is parsed. Another provider means swapping the small function that
  finds the text in an event.
- Tool-call arguments are passed through untouched; only message text is scanned.
- Detection is pattern-based: names, addresses and phone numbers are not covered, and a Luhn-valid 13-19 digit
  string that is not really a card will be redacted (fail-safe direction).
- A single unbroken token longer than 320 chars is released after the hold limit.
