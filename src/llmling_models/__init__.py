__version__ = "0.1.0"


from llmling_models.base import PydanticModel
from llmling_models.multi import MultiModel, RandomMultiModel

__all__ = [
    "MultiModel",
    "PydanticModel",
    "RandomMultiModel",
]
