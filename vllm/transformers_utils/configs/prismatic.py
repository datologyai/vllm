# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration class for PrismaticVLM model.

This module defines the configuration for the SigLIP2-Qwen3-1.7B model
with 729 visual tokens.
"""

import os
from typing import Any, Dict, List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class PrismaticConfig(PretrainedConfig):
    """
    Configuration class for PrismaticVLM model.
    
    This configuration class defines the model architecture parameters
    for the SigLIP2-Qwen3-1.7B multimodal model.
    """
    
    model_type = "prismatic"
    
    def __init__(
        self,
        # LLM backbone configuration (Qwen3-1.7B)
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-06,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 21,
        attention_dropout: float = 0.0,
        
        # Vision backbone configuration (SigLIP2-so400m)
        vision_embed_dim: int = 1152,
        vision_num_hidden_layers: int = 27,
        vision_num_attention_heads: int = 16,
        vision_intermediate_size: int = 4304,
        vision_hidden_act: str = "gelu_pytorch_tanh",
        vision_attention_dropout: float = 0.0,
        image_size: int = 378,
        patch_size: int = 14,
        
        # Multimodal configuration
        arch_specifier: str = "full-align+729-avgpool",
        image_resize_strategy: str = "resize-naive",
        
        # Training configuration
        gradient_checkpointing: bool = False,
        
        **kwargs,
    ):
        # LLM backbone parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        
        # Vision backbone parameters
        self.vision_embed_dim = vision_embed_dim
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_hidden_act = vision_hidden_act
        self.vision_attention_dropout = vision_attention_dropout
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Multimodal parameters
        self.arch_specifier = arch_specifier
        self.image_resize_strategy = image_resize_strategy
        
        # Training parameters
        self.gradient_checkpointing = gradient_checkpointing
        
        # Set architectures for model registry
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["PrismaticForConditionalGeneration"]
        
        super().__init__(**kwargs)
    
    @property
    def num_patches(self) -> int:
        """Calculate number of image patches."""
        return (self.image_size // self.patch_size) ** 2
    
    @property
    def num_image_tokens(self) -> int:
        """Get number of image tokens (same as patches for identity pooling)."""
        return self.num_patches
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        output = super().to_dict()
        
        # Add computed properties
        output["num_patches"] = self.num_patches
        output["num_image_tokens"] = self.num_image_tokens
        
        return output
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "PrismaticConfig":
        """Load configuration from pretrained model."""
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        return cls.from_dict(config_dict, **kwargs)
    
    def get_text_config(self):
        """Get text model configuration."""
        # Return self for PrismaticVLM since it's a unified architecture
        # where the main config contains the text model parameters
        return self
    
    def get_vision_config(self) -> Dict[str, Any]:
        """Get vision model configuration."""
        return {
            "embed_dim": self.vision_embed_dim,
            "num_hidden_layers": self.vision_num_hidden_layers,
            "num_attention_heads": self.vision_num_attention_heads,
            "intermediate_size": self.vision_intermediate_size,
            "hidden_act": self.vision_hidden_act,
            "attention_dropout": self.vision_attention_dropout,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
        }
    
    def get_multimodal_config(self) -> Dict[str, Any]:
        """Get multimodal configuration."""
        return {
            "arch_specifier": self.arch_specifier,
            "image_resize_strategy": self.image_resize_strategy,
            "num_patches": self.num_patches,
            "num_image_tokens": self.num_image_tokens,
        }


# Register with Transformers CONFIG_MAPPING
CONFIG_MAPPING.register("prismatic", PrismaticConfig)