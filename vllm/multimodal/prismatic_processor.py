# SPDX-License-Identifier: Apache-2.0  
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multimodal processor for PrismaticVLM model following Qwen2VL patterns.
"""

from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger
from vllm.multimodal.processing import BaseProcessingInfo
from vllm.multimodal.profiling import BaseDummyInputsBuilder

logger = init_logger(__name__)


class PrismaticProcessingInfo(BaseProcessingInfo):
    """Processing information for PrismaticVLM model."""
    
    def get_supported_mm_limits(self) -> Dict[str, int]:
        """Get supported multimodal limits."""
        return {"image": 1}  # Support 1 image per prompt
    
    def get_mm_max_tokens_per_item(self, seq_len: int) -> Dict[str, int]:
        """Get maximum tokens per multimodal item.""" 
        return {"image": 729}  # 729 tokens per image


class PrismaticDummyInputsBuilder(BaseDummyInputsBuilder[PrismaticProcessingInfo]):
    """Dummy inputs builder for PrismaticVLM model."""
    
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """Generate dummy processor inputs."""
        num_images = mm_counts.get("image", 1)
        
        # Create dummy image data  
        dummy_image = Image.new("RGB", (378, 378), color="black")
        
        # Create dummy text with image tokens
        image_tokens = ["<image>"] * 729 * num_images
        dummy_text = "".join(image_tokens) + " " + "dummy " * max(0, seq_len - 729 * num_images)
        
        return {
            "text": dummy_text,
            "images": [dummy_image] * num_images,
        }