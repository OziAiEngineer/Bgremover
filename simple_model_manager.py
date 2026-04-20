"""
Simplified Model Manager - Only for LaMa eraser
"""
import torch
from loguru import logger
import numpy as np
from iopaint.model import models
from iopaint.schema import InpaintRequest


class SimpleModelManager:
    """Simplified model manager that only loads LaMa model"""
    
    def __init__(self, name: str, device: torch.device, **kwargs):
        self.name = name
        self.device = device
        self.kwargs = kwargs
        
        logger.info(f"Loading model: {name}")
        
        if name not in models:
            raise ValueError(f"Model {name} not found. Available: {list(models.keys())}")
        
        self.model = models[name](device, **kwargs)
        logger.info(f"Model {name} loaded successfully!")
    
    @torch.inference_mode()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        Process image with mask
        
        Args:
            image: [H, W, C] RGB numpy array
            mask: [H, W, 1] 255 means area to repaint
            config: InpaintRequest configuration
        
        Returns:
            BGR image as numpy array
        """
        return self.model(image, mask, config).astype(np.uint8)
