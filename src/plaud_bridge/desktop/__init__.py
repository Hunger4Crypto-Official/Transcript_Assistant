"""
The clickable-app layer.

Everything the desktop application needs that is not the pipeline itself: a
headless `AppController` that a window (or a test) drives, and a small local web
UI the packaged .exe opens in a browser. Nothing here changes how a recording is
processed -- it is a friendlier front door onto the same engine `run.py` uses.
"""

from .controller import AppController, Brain, LocalLLMStatus, PreflightItem, probe_local_llm

__all__ = ["AppController", "Brain", "LocalLLMStatus", "PreflightItem", "probe_local_llm"]
