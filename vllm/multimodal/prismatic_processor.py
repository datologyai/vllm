# SPDX-License-Identifier: Apache-2.0  
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multimodal processor for PrismaticVLM model following Qwen2VL patterns.
"""

from typing import Any, Dict, List, Optional, Union, Mapping, Sequence

import torch
from PIL import Image
from transformers import BatchFeature
import torchvision.transforms as transforms

from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger
from vllm.multimodal.processing import (BaseProcessingInfo, BaseMultiModalProcessor, 
                                        PromptUpdate, PromptReplacement, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalDataDict, MultiModalInputs, MultiModalKwargs
from vllm.multimodal.parse import MultiModalDataItems

logger = init_logger(__name__)


class PrismaticProcessingInfo(BaseProcessingInfo):
    """Processing information for PrismaticVLM model."""
    
    def get_supported_mm_limits(self) -> Dict[str, int]:
        """Get supported multimodal limits."""
        return {"image": 1}  # Support 1 image per prompt
    
    def get_mm_max_tokens_per_item(self, seq_len: int) -> Dict[str, int]:
        """Get maximum tokens per multimodal item.""" 
        return {"image": 729}  # 729 tokens per image
    
    def get_allowed_mm_limits(self) -> Dict[str, int]:
        """Get allowed multimodal limits."""
        return {"image": 1}  # Allow 1 image per prompt
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.ctx.tokenizer


class PrismaticDummyInputsBuilder(BaseDummyInputsBuilder[PrismaticProcessingInfo]):
    """Dummy inputs builder for PrismaticVLM model."""
    
    def get_dummy_mm_data(self, mm_counts: Dict[str, int]) -> Dict[str, Any]:
        """Generate dummy multimodal data."""
        num_images = mm_counts.get("image", 1)
        
        # Create dummy image data  
        dummy_image = Image.new("RGB", (378, 378), color="black")
        
        return {
            "image": [dummy_image] * num_images
        }
    
    def get_dummy_text(self, mm_counts: Dict[str, int]) -> str:
        """Generate dummy text."""
        num_images = mm_counts.get("image", 1)
        
        # Create dummy text with image tokens
        image_tokens = "<image>" * num_images
        dummy_text = image_tokens + " dummy text"
        
        return dummy_text
    
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Dict[str, int],
    ) -> Any:
        """Generate dummy processor inputs."""
        # Create a simple object that has the expected attributes
        class DummyInputs:
            def __init__(self, text: str, mm_data: Dict[str, Any]):
                self.prompt = text
                self.text = text
                self.multi_modal_data = mm_data
                self.mm_data = mm_data  # Alternative name
                self.hf_processor_mm_kwargs = {}  # Empty dict for HF processor args
                self.prompt_token_ids = []  # Empty for now
                self.tokenization_kwargs = {}  # Empty dict for tokenization args
        
        return DummyInputs(
            self.get_dummy_text(mm_counts),
            self.get_dummy_mm_data(mm_counts)
        )


class PrismaticMultiModalProcessor(BaseMultiModalProcessor[PrismaticProcessingInfo]):
    """Multimodal processor for PrismaticVLM."""
    
    def __init__(self, info, dummy_inputs_builder, cache=None, **kwargs):
        """Initialize the processor."""
        super().__init__(info, dummy_inputs_builder, cache=cache)
        
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Get multimodal field configuration for PrismaticVLM."""
        # Handle pixel_values tensor - check if it exists and has proper shape
        pixel_values = hf_inputs.get("pixel_values")
        if pixel_values is not None and pixel_values.numel() > 0:
            # pixel_values has shape [batch_size, num_channels, height, width]
            # For PrismaticVLM, we expect batched image data
            return dict(
                pixel_values=MultiModalFieldConfig.batched("image"),
            )
        else:
            # Return empty config if no pixel values
            return dict()
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Call HF processor for PrismaticVLM.
        
        Since PrismaticVLM doesn't have a standard HF processor, we create
        a minimal processor that handles tokenization and image processing.
        """
        # Get images from mm_data (could be "image" or "images")
        images = mm_data.get("images", mm_data.get("image", []))
        if not isinstance(images, list):
            images = [images] if images is not None else []
        
        # Validate image count
        max_images = self.info.get_supported_mm_limits().get("image", 1)
        if len(images) > max_images:
            raise ValueError(f"Too many images: {len(images)} > {max_images}")
        
        # Process images: resize and normalize for SigLIP2
        processed_images = []
        for image in images:
            if hasattr(image, 'resize'):
                # PIL Image
                resized_image = image.resize((378, 378))
                # Convert to tensor and normalize for SigLIP2
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                processed_image = transform(resized_image)
                processed_images.append(processed_image)
            else:
                # Already a tensor
                processed_images.append(image)
        
        # Stack images into a batch tensor if we have images
        if processed_images:
            pixel_values = torch.stack(processed_images)
        else:
            pixel_values = torch.empty((0, 3, 378, 378))
        
        # Tokenize the prompt normally (without image replacement)
        tokenizer = self.info.ctx.tokenizer
        
        # Filter out conflicting parameters from tok_kwargs
        filtered_tok_kwargs = {k: v for k, v in tok_kwargs.items() 
                              if k not in {'padding', 'truncation', 'return_tensors'}}
        
        tokenized = tokenizer(
            prompt,
            padding=False,
            truncation=False,
            return_tensors="pt",
            **filtered_tok_kwargs
        )
        
        input_ids = tokenized["input_ids"][0]
        
        # Create BatchFeature object with proper structure
        return BatchFeature({
            "input_ids": input_ids.unsqueeze(0),  # Add batch dimension
            "pixel_values": pixel_values,
        })
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        """
        Get prompt updates for PrismaticVLM multimodal processing.
        
        This method defines how to replace <image> tokens in the prompt with
        the appropriate number of image tokens (729 for PrismaticVLM).
        """
        # Get the correct image token ID from the tokenizer
        # Use the actual tokenizer vocabulary token ID
        tokenizer = self.info.get_tokenizer()
        
        # Try to find the correct image token ID
        image_token_id = None
        
        # First, try to get from config if available
        try:
            hf_config = self.info.get_hf_config()
            if hasattr(hf_config, 'image_token_index'):
                image_token_id = hf_config.image_token_index
        except:
            pass
        
        # If not found, look for image_pad token in tokenizer
        if image_token_id is None:
            for token_id, token_obj in tokenizer.added_tokens_decoder.items():
                token_str = str(token_obj)
                if token_str == '<|image_pad|>':
                    image_token_id = token_id
                    break
        
        # Final fallback to correct image_pad token ID
        if image_token_id is None:
            image_token_id = 151655  # <|image_pad|> token ID - verified correct
        
        # PrismaticVLM uses a fixed number of image tokens (729) per image
        NUM_IMAGE_TOKENS = 729
        
        def get_replacement(item_idx: int):
            # For PrismaticVLM, each image is represented by 729 image tokens
            # Use the actual tokenizer vocabulary token ID
            return [image_token_id] * NUM_IMAGE_TOKENS
        
        # Get the tokenized representation of "<image>" for proper replacement
        tokenizer = self.info.get_tokenizer()
        image_placeholder_tokens = tokenizer.encode("<image>", add_special_tokens=False)
        
        return [
            PromptReplacement(
                modality="image",
                target=image_placeholder_tokens,  # Use tokenized representation
                replacement=get_replacement,
            ),
        ]
    
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """
        Check if the HF processor applies updates to the prompt.
        
        For PrismaticVLM, our _call_hf_processor does NOT apply image token
        replacements, so we return False to let vLLM apply them via _get_prompt_updates.
        """
        return False

    
