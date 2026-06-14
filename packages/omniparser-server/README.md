# omniparser-server

FastAPI service that wraps `omniparser-core` and exposes the parser over
HTTP. Suitable for running on a GPU host while clients (agents, IDEs,
notebooks) call it from anywhere.

Migration target for `upstream/omnitool/omniparserserver/omniparserserver.py`.
The service is rewritten to use Pydantic Settings instead of argparse,
adds proper health probes (`/healthz`, `/readyz`), structured logging,
and optional OpenTelemetry + Prometheus instrumentation.

## Status

`0.1.0.dev0` — skeleton only.
