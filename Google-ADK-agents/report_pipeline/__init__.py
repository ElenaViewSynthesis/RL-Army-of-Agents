"""Full-report pipeline package.

A ``SequentialAgent`` that runs the research steps in a fixed order and emits a
single institutional-style report. ADK discovers ``agent.root_agent``.
"""

from . import agent

__all__ = ["agent"]
