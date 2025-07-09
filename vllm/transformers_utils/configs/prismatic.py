# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PretrainedConfig
from typing import Dict, Any, Optional


class PrismaticConfig(PretrainedConfig):
    """
    Configuration class for Prismatic VLM models.
    
    This configuration stores all the configuration parameters for a Prismatic VLM model.
    """
    
    model_type = "prismatic"
    
    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        projector_config: Optional[Dict[str, Any]] = None,
        image_token_index: int = 151655,  # <|image_pad|> token
        ignore_index: int = -100,
        image_resize_strategy: str = "resize-naive",
        vision_backbone_id: str = "siglip2-vit-so400m-p14-378px",
        llm_backbone_id: str = "qwen3-1.7b",
        arch_specifier: str = "full-align+729-avgpool",
        llm_max_length: int = 4096,
        model_id: str = "prismatic-vlm",
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 5504,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 4096,
        # Vision-specific parameters
        vision_embed_dim: int = 1152,
        vision_num_hidden_layers: int = 27,
        vision_num_attention_heads: int = 16,
        vision_intermediate_size: int = 4304,
        vision_hidden_act: str = "gelu_pytorch_tanh",
        vision_attention_dropout: float = 0.0,
        image_size: int = 378,
        patch_size: int = 14,
        num_image_tokens: int = 729,
        **kwargs
    ):
        """
        Initialize Prismatic configuration.
        
        Args:
            vision_config: Configuration for the vision backbone
            text_config: Configuration for the text backbone
            projector_config: Configuration for the multimodal projector
            image_token_index: Special token index for images
            ignore_index: Index to ignore in loss computation
            image_resize_strategy: Strategy for resizing images
            vision_backbone_id: Identifier for the vision backbone
            llm_backbone_id: Identifier for the LLM backbone
            arch_specifier: Architecture specifier string
            llm_max_length: Maximum sequence length for the LLM
            model_id: Model identifier
            vocab_size: Size of the vocabulary
            hidden_size: Hidden size of the model
            intermediate_size: Intermediate size in feed-forward layers
            num_hidden_layers: Number of hidden layers
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key-value heads
            max_position_embeddings: Maximum position embeddings
            vision_embed_dim: Vision embedding dimension
            vision_num_hidden_layers: Number of vision hidden layers
            vision_num_attention_heads: Number of vision attention heads
            vision_intermediate_size: Vision intermediate size
            vision_hidden_act: Vision hidden activation
            vision_attention_dropout: Vision attention dropout
            image_size: Input image size
            patch_size: Vision patch size
            num_image_tokens: Number of image tokens
            **kwargs: Additional keyword arguments
        """
        # Set default configurations if not provided
        if vision_config is None:
            vision_config = {
                "model_type": "siglip_vision_model",
                "backbone_id": vision_backbone_id,
                "image_size": image_size,
                "patch_size": patch_size,
                "num_channels": 3,
                "hidden_size": vision_embed_dim,
                "intermediate_size": vision_intermediate_size,
                "num_hidden_layers": vision_num_hidden_layers,
                "num_attention_heads": vision_num_attention_heads,
                "layer_norm_eps": 1e-06,
                "attention_dropout": vision_attention_dropout,
                "hidden_act": vision_hidden_act
            }
        
        if text_config is None:
            text_config = {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "backbone_id": llm_backbone_id,
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "max_position_embeddings": max_position_embeddings,
                "sliding_window": None,
                "use_cache": True,
                "rope_theta": 1000000.0,
                "attention_dropout": 0.0,
                "hidden_act": "silu",
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-06,
                "tie_word_embeddings": False
            }
        
        if projector_config is None:
            projector_config = {
                "projector_type": "avgpool",
                "arch_specifier": arch_specifier,
                "query_dim": num_image_tokens,
                "mm_hidden_size": vision_embed_dim,
                "llm_hidden_size": hidden_size
            }
        
        self.vision_config = vision_config
        self.text_config = text_config
        self.projector_config = projector_config
        self.image_token_index = image_token_index
        self.ignore_index = ignore_index
        self.image_resize_strategy = image_resize_strategy
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.llm_max_length = llm_max_length
        self.model_id = model_id
        
        # Set primary attributes
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        
        # Vision-specific attributes
        self.vision_embed_dim = vision_embed_dim
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_hidden_act = vision_hidden_act
        self.vision_attention_dropout = vision_attention_dropout
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_image_tokens = num_image_tokens
        self.image_token_index = image_token_index
        
        # Set default token IDs
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = 151645
        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = 151645
        if "bos_token_id" not in kwargs:
            kwargs["bos_token_id"] = 151643
        
        # Set architectures for vLLM registry
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["PrismaticVLMForCausalLM"]
        
        super().__init__(**kwargs)
    
    def get_text_config(self, decoder=False):
        """Get the text configuration for the LLM backbone."""
        # Create a simple config object from the text_config dict
        class TextConfig:
            def __init__(self, config_dict):
                self._config_dict = config_dict.copy()
                for key, value in config_dict.items():
                    setattr(self, key, value)
            
            def to_dict(self):
                return self._config_dict.copy()
        
        return TextConfig(self.text_config)
    
    def to_dict(self):
        """Convert the configuration to a dictionary."""
        output = super().to_dict()
        output.update({
            "vision_config": self.vision_config,
            "text_config": self.text_config,
            "projector_config": self.projector_config,
            "image_token_index": self.image_token_index,
            "ignore_index": self.ignore_index,
            "image_resize_strategy": self.image_resize_strategy,
            "vision_backbone_id": self.vision_backbone_id,
            "llm_backbone_id": self.llm_backbone_id,
            "arch_specifier": self.arch_specifier,
            "llm_max_length": self.llm_max_length,
            "model_id": self.model_id,
            "vision_embed_dim": self.vision_embed_dim,
            "vision_num_hidden_layers": self.vision_num_hidden_layers,
            "vision_num_attention_heads": self.vision_num_attention_heads,
            "vision_intermediate_size": self.vision_intermediate_size,
            "vision_hidden_act": self.vision_hidden_act,
            "vision_attention_dropout": self.vision_attention_dropout,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_image_tokens": self.num_image_tokens,
            "image_token_index": self.image_token_index,
        })
        return output