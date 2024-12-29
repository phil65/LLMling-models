__version__ = "0.2.0"


from llmling_models.base import PydanticModel
from llmling_models.multi import MultiModel
from llmling_models.multimodels.random import RandomMultiModel
from llmling_models.multimodels.fallback import FallbackMultiModel

__all__ = [
    "FallbackMultiModel",
    "MultiModel",
    "PydanticModel",
    "RandomMultiModel",
]
