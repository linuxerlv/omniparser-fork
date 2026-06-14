"""Gradio agent UI driving omnitool-agent.

This package is the relocated and de-vendored version of
``upstream/omnitool/gradio/app_new.py`` plus its supporting modules.
Legacy ``app.py`` and ``app_streamlit.py`` from upstream were dropped as
part of the migration — only the ``app_new.py`` UI is supported.

Public surface:

* ``omnitool_ui.cli.main`` — console entry point (``omnitool-ui`` script).
* ``omnitool_ui.app`` — the Gradio app module (legacy script form).

Wire traffic to the OmniParser server is routed exclusively through
:mod:`omnitool_agent`; this package never speaks the wire schema directly.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"
__all__: list[str] = []
