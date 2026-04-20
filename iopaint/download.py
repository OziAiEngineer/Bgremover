"""
Minimal download.py - Just enough to make model_manager work
"""
from typing import List
from iopaint.schema import ModelType, ModelInfo
from iopaint.model import models


def scan_models() -> List[ModelInfo]:
    """
    Scan and return available models
    For simplified eraser, we only return the basic erase models
    """
    available_models = []
    
    # Scan inpaint models (lama, mat, etc.)
    for name, m in models.items():
        if m.is_erase_model and m.is_downloaded():
            available_models.append(
                ModelInfo(
                    name=name,
                    path=name,
                    model_type=ModelType.INPAINT,
                )
            )
    
    return available_models
