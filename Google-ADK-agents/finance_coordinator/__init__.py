"""Finance coordinator agent package.

ADK discovers the agent via ``agent.root_agent``. Importing the ``agent``
module here makes ``adk web`` / ``adk run`` pick it up automatically.
"""

from . import agent

__all__ = ["agent"]
