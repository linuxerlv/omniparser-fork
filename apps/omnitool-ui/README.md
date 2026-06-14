# omnitool-ui

Gradio agent UI for OmniTool. Migrated from
`upstream/omnitool/gradio/app_new.py` and its supporting modules.

## What was migrated

```
upstream/omnitool/gradio/
├── app_new.py             →  src/omnitool_ui/app.py
├── loop.py                →  src/omnitool_ui/loop.py
├── tools/                 →  src/omnitool_ui/tools/
├── agent/                 →  src/omnitool_ui/agent/
└── executor/              →  src/omnitool_ui/executor/
```

## What was dropped

* `upstream/omnitool/gradio/app.py` — superseded by `app_new.py`.
* `upstream/omnitool/gradio/app_streamlit.py` — alternative UI, not maintained.
* `upstream/omnitool/omniparserserver/` — replaced by `omniparser-server`.

## What changed

* All internal imports rewritten from the upstream ad-hoc form
  (`from tools import ...`, `from agent.foo import ...`) to fully qualified
  `omnitool_ui.*` imports.
* `agent/llm_utils/omniparserclient.py` rewritten as a thin shim around
  `omnitool_agent.OmnitoolParserClient`. The class name `OmniParserClient`
  and its no-arg `__call__()` interface are preserved so the agent loop
  needs no further changes.
* Two `bare except:` sites fixed in
  `agent/vlm_agent.py:166` and `agent/vlm_agent_with_orchestrator.py:229`.
* Console entry point added: `omnitool-ui` (replaces
  `python app_new.py --windows_host_url ...`).

## Running

```bash
omnitool-ui --windows-host-url localhost:8006 --omniparser-server-url localhost:8000
```

The legacy underscore flag form (`--windows_host_url`, `--omniparser_server_url`,
`--run_folder`) is accepted as an alias for runbook compatibility.

## Status

This app is a deployment artefact, not a published package
(`Private :: Do Not Upload`). The 760-line `app.py` is preserved as a
script form; refactoring it into a function-based module is tracked
separately and is not required for the migration.
