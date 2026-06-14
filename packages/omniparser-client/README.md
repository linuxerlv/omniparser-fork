# omniparser-client

A deliberately thin HTTP client for the OmniParser inference server. Depends
only on `httpx` and `pydantic`, **not** on torch, transformers, ultralytics,
or OCR engines. Use this when you want to integrate with a remote
OmniParser server (for example from an agent loop) without paying the
~3 GB cost of the full inference stack.

Migration target for
`upstream/omnitool/gradio/agent/llm_utils/omniparserclient.py`.

## Status

`0.1.0.dev0` — skeleton only.
