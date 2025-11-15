"""Remote input server + client."""

from .model import RemoteInputModel
from .server import ModelServer

__all__ = ["ModelServer", "RemoteInputModel"]
