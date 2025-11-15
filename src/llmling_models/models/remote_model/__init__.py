"""Remote model server + client."""

from .model import RemoteProxyModel
from .server import ModelServer

__all__ = ["ModelServer", "RemoteProxyModel"]
