__version__ = "0.0.1"


from llmling_models.base import PydanticModel
from llmling_models.multi import MultiModel, RandomMultiModel

__all__ = [
    "MultiModel",
    "PydanticModel",
    "RandomMultiModel",
]