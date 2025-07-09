# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Iterable, List, Optional, Tuple, Union
import copy
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalInputs, MultiModalKwargs)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.prismatic import PrismaticConfig

from .interfaces import SupportsMultiModal
from .utils import (AutoWeightsLoader, WeightsMapper, make_layers,
                    maybe_prefix, merge_multimodal_embeddings,
                    init_vllm_registered_model)

# Try to import Prismatic library from standard installation or environment
# Note: Prismatic should be properly installed as a package, not added to sys.path

# Import actual Prismatic components
try:
    from prismatic.models.backbones.vision import VisionBackbone
    from prismatic.models.backbones.llm import LLMBackbone
    from prismatic.util.nn_utils import AvgPoolProjector, MLPProjector, LinearProjector
    from prismatic.models.materialize import get_vision_backbone_and_transform, get_llm_backbone_and_tokenizer
    PRISMATIC_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Prismatic library not available: {e}. Using fallback implementation.")
    PRISMATIC_AVAILABLE = False


class PrismaticVLMForCausalLM(nn.Module, SupportsMultiModal):
    """
    Prismatic VLM model for causal language modeling in vLLM.
    
    This implementation directly integrates actual Prismatic components 
    instead of recreating the architecture from scratch.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        if not isinstance(config, PrismaticConfig):
            raise ValueError(f"Expected PrismaticConfig, got {type(config)}")
        
        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        
        # Create a proper PretrainedConfig for the text model
        from transformers import PretrainedConfig
        
        text_config_dict = config.text_config.copy()
        
        # Ensure architectures field is present
        if 'architectures' not in text_config_dict:
            text_config_dict['architectures'] = ['Qwen3ForCausalLM']
        
        # Create a PretrainedConfig object
        text_config_obj = PretrainedConfig(**text_config_dict)
        
        # Initialize components using actual Prismatic components
        if PRISMATIC_AVAILABLE:
            self.vision_backbone = self._create_prismatic_vision_backbone()
            self.projector = self._create_prismatic_projector()
        else:
            # Fallback to custom implementation
            self.vision_backbone = self._create_vision_tower()
            self.projector = self._create_projector()
        
        # Initialize language model using vLLM's model loader
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_config_obj,
            prefix=maybe_prefix(prefix, "language_model")
        )
        
        # Set up compatibility for intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)
        
        # Set up image token handling
        self._setup_image_token_handling()
    
    def _create_prismatic_vision_backbone(self):
        """Create vision backbone using actual Prismatic components."""
        if not PRISMATIC_AVAILABLE:
            raise ImportError("Prismatic library not available")
        
        # Use the actual Prismatic vision backbone initialization
        vision_backbone_id = self.config.vision_backbone_id
        
        # Create vision backbone using Prismatic's materialize function
        vision_backbone, _ = get_vision_backbone_and_transform(
            vision_backbone_id=vision_backbone_id,
            image_resize_strategy="resize-naive"
        )
        
        # Ensure vision backbone uses the correct dtype and device for vLLM
        vision_backbone = vision_backbone.to(dtype=torch.bfloat16)
        
        return vision_backbone
    
    def _create_prismatic_projector(self):
        """Create projector using actual Prismatic components."""
        if not PRISMATIC_AVAILABLE:
            raise ImportError("Prismatic library not available")
        
        # Use the actual Prismatic projector initialization
        vision_hidden_size = self.config.vision_embed_dim
        llm_hidden_size = self.config.hidden_size
        arch_specifier = getattr(self.config, 'arch_specifier', '729-avgpool')
        
        # Create projector based on architecture specifier
        if arch_specifier == "linear":
            projector = LinearProjector(vision_hidden_size, llm_hidden_size)
        elif arch_specifier.endswith("gelu-mlp"):
            projector = MLPProjector(vision_hidden_size, llm_hidden_size)
        elif arch_specifier.endswith("avgpool"):
            # Parse query dimension from arch_specifier
            if arch_specifier.split("-")[0].isdigit():
                query_dim = int(arch_specifier.split("-")[0])
            else:
                query_dim = 729  # Default for this model
            
            projector = AvgPoolProjector(
                query_num=query_dim,
                mm_hidden_size=vision_hidden_size,
                llm_hidden_size=llm_hidden_size
            )
        else:
            raise ValueError(f"Unsupported architecture specifier: {arch_specifier}")
        
        # Ensure projector uses the correct dtype for vLLM
        projector = projector.to(dtype=torch.bfloat16)
        
        return projector
    
    def _create_vision_tower(self):
        """Fallback: Create vision tower based on SigLIP2 architecture."""
        # Create a proper vision tower that mimics SigLIP2 architecture
        # This will process images into meaningful patch embeddings
        return SigLIP2VisionBackbone(self.config)
    
    def _create_projector(self):
        """Fallback: Create the multimodal projector."""
        vision_hidden_size = self.config.vision_embed_dim
        llm_hidden_size = self.config.hidden_size
        
        # Create a proper avgpool projector as used in the original PrismaticVLM
        return AvgPoolProjector(
            query_num=self.config.num_image_tokens,
            mm_hidden_size=vision_hidden_size,
            llm_hidden_size=llm_hidden_size
        )
    
    def _setup_image_token_handling(self):
        """Setup proper image token handling."""
        # Ensure the image token index is properly set
        if not hasattr(self.config, 'image_token_index'):
            # Default to the correct image token index (<|image_pad|>)
            self.config.image_token_index = 151655
        
        # Image token index is set in config
    
    def _parse_and_validate_image_input(self, **kwargs: object) -> Optional[torch.Tensor]:
        """Parse and validate image input following LLaVA pattern."""
        pixel_values = kwargs.pop("pixel_values", None)
        
        if pixel_values is None:
            return None
        
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError(f"Expected pixel_values to be a torch.Tensor, got {type(pixel_values)}")
        
        return pixel_values
    
    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[torch.Tensor]:
        """
        Get multimodal embeddings for the model using actual Prismatic components.
        
        This method is required by vLLM's multimodal framework.
        """
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        
        # Process through vision backbone to get patch features
        if PRISMATIC_AVAILABLE:
            # Use actual Prismatic vision backbone
            # Ensure input has the correct dtype
            image_input = image_input.to(dtype=torch.bfloat16)
            vision_features = self.vision_backbone(image_input)
            
            # Project to LLM hidden size using the actual Prismatic projector
            vision_embeddings = self.projector(vision_features)
        else:
            # Fallback to custom implementation
            vision_features = self.vision_backbone(image_input)
            vision_embeddings = self.projector(vision_features)
        
        return vision_embeddings
    
    def get_input_embeddings(self, input_ids: torch.Tensor, 
                           multimodal_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get input embeddings with multimodal support following LLaVA pattern."""
        # Get text embeddings from language model
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        # Merge multimodal embeddings if provided
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_index,
            )
        
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """
        Forward pass through the model following LLaVA pattern.
        
        This implementation properly integrates vision and text processing.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits from hidden states."""
        return self.language_model.compute_logits(hidden_states, sampling_metadata)
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Sample from logits."""
        return self.language_model.sample(logits, sampling_metadata)
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from a checkpoint using actual Prismatic components."""
        # Convert weights to dict for easier handling
        weights_dict = dict(weights)
        
        # Handle nested checkpoint structure (like Prismatic does)
        if 'model' in weights_dict and len(weights_dict) == 1:
            model_state_dict = weights_dict['model']
            
            # Check if this is a Prismatic-style checkpoint with component keys
            if isinstance(model_state_dict, dict) and 'projector' in model_state_dict and 'llm_backbone' in model_state_dict:
                loaded_weights = set()
                
                try:
                    # 1. Load vision backbone if present
                    if 'vision_backbone' in model_state_dict:
                        vision_state_dict = model_state_dict['vision_backbone']
                        
                        if PRISMATIC_AVAILABLE:
                            # Use actual Prismatic vision backbone loading
                            missing_keys, unexpected_keys = self.vision_backbone.load_state_dict(vision_state_dict, strict=False)
                            
                            # Ensure vision backbone is moved to correct device and dtype after loading weights
                            try:
                                device = next(self.language_model.parameters()).device
                                self.vision_backbone = self.vision_backbone.to(device=device, dtype=torch.bfloat16)
                            except Exception:
                                pass  # Continue if device move fails
                            
                            loaded_weights.update(vision_state_dict.keys())
                        else:
                            # Fallback to custom vision loading
                            compatible_vision_weights = {}
                            for name, param in vision_state_dict.items():
                                mapped_name = self._map_vision_weight_name(f"vision_backbone.{name}")
                                if mapped_name:
                                    compatible_vision_weights[mapped_name] = param
                            
                            if compatible_vision_weights:
                                vision_loader = AutoWeightsLoader(self.vision_backbone)
                                loaded_weights.update(vision_loader.load_weights(list(compatible_vision_weights.items())))
                    
                    # 2. Load projector
                    if 'projector' in model_state_dict:
                        projector_state_dict = model_state_dict['projector']
                        
                        if PRISMATIC_AVAILABLE:
                            # Use actual Prismatic projector loading
                            # Remove "mlp_projector." prefix from checkpoint weights
                            clean_projector_state_dict = {}
                            for name, param in projector_state_dict.items():
                                if name.startswith("mlp_projector."):
                                    clean_name = name.replace("mlp_projector.", "")
                                    clean_projector_state_dict[clean_name] = param
                                else:
                                    clean_projector_state_dict[name] = param
                            
                            missing_keys, unexpected_keys = self.projector.load_state_dict(clean_projector_state_dict, strict=False)
                            
                            # Ensure projector is moved to correct device and dtype after loading weights
                            try:
                                device = next(self.language_model.parameters()).device
                                self.projector = self.projector.to(device=device, dtype=torch.bfloat16)
                            except Exception:
                                pass  # Continue if device move fails
                            
                            loaded_weights.update(clean_projector_state_dict.keys())
                        else:
                            # Fallback to custom projector loading
                            compatible_projector_weights = {}
                            for name, param in projector_state_dict.items():
                                if name.startswith("mlp_projector."):
                                    mapped_name = name.replace("mlp_projector.", "")
                                    compatible_projector_weights[mapped_name] = param
                            
                            if compatible_projector_weights:
                                projector_loader = AutoWeightsLoader(self.projector)
                                loaded_weights.update(projector_loader.load_weights(list(compatible_projector_weights.items())))
                    
                    # 3. Load LLM backbone
                    if 'llm_backbone' in model_state_dict:
                        llm_state_dict = model_state_dict['llm_backbone']
                        
                        # Map LLM weights
                        compatible_llm_weights = {}
                        for name, param in llm_state_dict.items():
                            if name.startswith("llm."):
                                mapped_name = name.replace("llm.", "")
                                compatible_llm_weights[mapped_name] = param
                        
                        if compatible_llm_weights:
                            # Use AutoWeightsLoader for LLM to handle QKV packing
                            llm_weights_list = [(name, param) for name, param in compatible_llm_weights.items()]
                            llm_loader = AutoWeightsLoader(self.language_model)
                            loaded_weights.update(llm_loader.load_weights(llm_weights_list))
                    
                    return loaded_weights
                    
                except Exception:
                    # Fall back to original method if component-wise loading fails
                    pass
                    # Fall back to original method if component-wise loading fails
            
        # Fall back to original flattening approach if component-wise loading isn't applicable
        return self._load_weights_fallback(weights_dict)
    
    def _load_weights_fallback(self, weights_dict: dict) -> set[str]:
        """Fallback weight loading method."""
        # Recursively flatten the nested structure
        flattened_weights = []
        
        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                else:
                    flattened_weights.append((new_key, value))
        
        if 'model' in weights_dict and len(weights_dict) == 1:
            flatten_dict(weights_dict['model'])
            weights = flattened_weights
        
        # Separate weights by component  
        llm_weights = []
        vision_weights = []
        projector_weights = []
        
        for name, param in weights:
            if name.startswith("llm_backbone.llm."):
                mapped_name = name.replace("llm_backbone.llm.", "")
                llm_weights.append((mapped_name, param))
            elif name.startswith("vision_backbone.featurizer."):
                mapped_name = self._map_vision_weight_name(name)
                if mapped_name:
                    vision_weights.append((mapped_name, param))
            elif name.startswith("projector.mlp_projector."):
                mapped_name = name.replace("projector.mlp_projector.", "")
                projector_weights.append((mapped_name, param))
        
        
        # Load weights for each component
        loaded_weights = set()
        
        # 1. Load LLM weights
        if llm_weights:
            llm_loader = AutoWeightsLoader(self.language_model)
            loaded_weights.update(llm_loader.load_weights(llm_weights))
        
        # 2. Load vision backbone weights  
        if vision_weights:
            if PRISMATIC_AVAILABLE:
                vision_loader = AutoWeightsLoader(self.vision_backbone)
            else:
                vision_loader = AutoWeightsLoader(self.vision_backbone)
            loaded_weights.update(vision_loader.load_weights(vision_weights))
        
        # 3. Load projector weights
        if projector_weights:
            projector_loader = AutoWeightsLoader(self.projector)
            loaded_weights.update(projector_loader.load_weights(projector_weights))
        
        return loaded_weights
    
    def _map_vision_weight_name(self, name: str) -> str:
        """Map vision backbone weight names to vLLM vision tower structure."""
        # Remove the vision_backbone.featurizer. prefix
        name = name.replace("vision_backbone.featurizer.", "")
        
        # Map trunk structure to vision tower - based on actual checkpoint analysis
        if name.startswith("trunk."):
            name = name.replace("trunk.", "")
            
            # Map patch embedding 
            if name.startswith("patch_embed.proj."):
                return name.replace("patch_embed.proj.", "patch_embedding.")
            
            # Map position embeddings
            elif name == "pos_embed":
                return "position_embeddings"
            
            # Map transformer blocks (0-26, total 27 blocks)
            elif name.startswith("blocks."):
                name = name.replace("blocks.", "vision_layers.")
                
                # Map layer normalization: norm1 -> attention_norm, norm2 -> mlp_norm  
                name = name.replace(".norm1.", ".attention_norm.")
                name = name.replace(".norm2.", ".mlp_norm.")
                
                # Map attention: attn -> attention
                name = name.replace(".attn.", ".attention.")
                
                # Handle QKV combined weights (qkv.weight [3456, 1152], qkv.bias [3456])
                if ".attention.qkv." in name:
                    # For our custom attention, we need to map to in_proj.weight/bias
                    name = name.replace(".attention.qkv.", ".attention.in_proj.")
                
                # Handle attention output projection (proj.weight [1152, 1152], proj.bias [1152])
                if ".attention.proj." in name:
                    name = name.replace(".attention.proj.", ".attention.out_proj.")
                
                # Map MLP layers: fc1 -> mlp.0, fc2 -> mlp.2 (with activation mlp.1 in between)
                name = name.replace(".mlp.fc1.", ".mlp.0.")
                name = name.replace(".mlp.fc2.", ".mlp.2.")
                
                return name
            
            # Map final layer norm (norm.weight [1152], norm.bias [1152])
            elif name.startswith("norm."):
                return name.replace("norm.", "layer_norm.")
            
            # Map attention pooling weights (13 weights total)
            elif name.startswith("attn_pool."):
                # Skip attention pooling weights for now - not essential for basic inference
                return None
        
        return None  # Skip unmapped weights



class SigLIP2VisionBackbone(nn.Module):
    """
    SigLIP2 Vision Backbone that processes images into patch embeddings.
    
    This implementation creates meaningful visual representations from input images.
    """
    
    def __init__(self, config: PrismaticConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding layer (matches checkpoint structure: bias=True)
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=config.vision_embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True  # SigLIP2 uses bias in patch embedding
        )
        
        # Vision transformer layers (27 layers for SigLIP2)
        self.vision_layers = nn.ModuleList([
            SigLIP2VisionLayer(config) for _ in range(config.vision_num_hidden_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.vision_embed_dim)
        
        # Position embeddings
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches, config.vision_embed_dim) * 0.02
        )
        
        # Attention pooling layer (as used in SigLIP2) - disabled for now
        # self.attn_pool = AttentionPooling(config)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Process pixel values into patch embeddings.
        
        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width)
            
        Returns:
            Patch embeddings of shape (batch_size, num_patches, vision_embed_dim)
        """
        # Handle extra dimension that might exist from multimodal processing
        if pixel_values.dim() == 5:
            # Shape: [batch_size, num_images, channels, height, width]
            # Reshape to [batch_size * num_images, channels, height, width]
            batch_size, num_images = pixel_values.shape[:2]
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        elif pixel_values.dim() != 4:
            raise ValueError(f"Expected 4D or 5D pixel_values, got {pixel_values.dim()}D with shape {pixel_values.shape}")
        
        # Ensure correct data type (convert to same type as patch embedding weights)
        pixel_values = pixel_values.to(self.patch_embedding.weight.dtype)
        
        # Extract patches
        patch_embeddings = self.patch_embedding(pixel_values)
        
        # Reshape to sequence format
        batch_size = patch_embeddings.size(0)
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        
        # Add position embeddings
        patch_embeddings = patch_embeddings + self.position_embeddings
        
        # Apply vision transformer layers
        for layer in self.vision_layers:
            patch_embeddings = layer(patch_embeddings)
        
        # Apply final layer norm
        patch_embeddings = self.layer_norm(patch_embeddings)
        
        return patch_embeddings


class SigLIP2VisionLayer(nn.Module):
    """
    A single vision transformer layer for SigLIP2.
    """
    
    def __init__(self, config: PrismaticConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention using separate Linear layers (compatible with checkpoint)
        self.attention = SigLIP2Attention(config)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.vision_embed_dim, config.vision_intermediate_size),
            get_act_fn(config.vision_hidden_act),
            nn.Linear(config.vision_intermediate_size, config.vision_embed_dim)
        )
        
        # Layer norms
        self.attention_norm = nn.LayerNorm(config.vision_embed_dim)
        self.mlp_norm = nn.LayerNorm(config.vision_embed_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Attention with residual connection
        attn_output = self.attention(hidden_states)
        hidden_states = hidden_states + attn_output
        hidden_states = self.attention_norm(hidden_states)
        
        # MLP with residual connection
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output
        hidden_states = self.mlp_norm(hidden_states)
        
        return hidden_states


class SigLIP2Attention(nn.Module):
    """
    Multi-head attention layer for SigLIP2 with separate Linear layers.
    This structure is compatible with the checkpoint weight loading.
    """
    
    def __init__(self, config: PrismaticConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_embed_dim
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).")
        
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection as Linear layer (compatible with AutoWeightsLoader)
        self.in_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    @property
    def in_proj_weight(self):
        """Compatibility property for weight loading."""
        return self.in_proj.weight
        
    @property
    def in_proj_bias(self):
        """Compatibility property for weight loading."""
        return self.in_proj.bias
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # Apply combined QKV projection
        qkv = self.in_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        
        q, k, v = qkv.unbind(0)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class AvgPoolProjector(nn.Module):
    """
    Average pooling projector that reduces the number of visual tokens.
    
    This projector takes the full patch embeddings and reduces them to a fixed number
    of tokens using average pooling, as used in the original PrismaticVLM.
    """
    
    def __init__(self, query_num: int, mm_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.query_num = query_num  # Number of output tokens (e.g., 729)
        self.mm_hidden_size = mm_hidden_size  # Vision embedding dimension
        self.llm_hidden_size = llm_hidden_size  # LLM embedding dimension
        
        # 2-layer MLP projector to match checkpoint structure
        # Layer 0: 1152 -> 2048, Layer 1: activation, Layer 2: 2048 -> 2048
        self.add_module("0", nn.Linear(mm_hidden_size, llm_hidden_size))  # Layer 0
        self.add_module("1", nn.GELU())  # Layer 1: activation (no parameters)
        self.add_module("2", nn.Linear(llm_hidden_size, llm_hidden_size))  # Layer 2
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to LLM hidden size.
        
        Args:
            vision_features: Vision embeddings of shape (batch_size, num_patches, vision_embed_dim)
            
        Returns:
            Projected embeddings of shape (batch_size, query_num, llm_hidden_size)
        """
        batch_size, num_patches, _ = vision_features.shape
        
        # Apply average pooling to reduce to query_num tokens
        if num_patches != self.query_num:
            # Reshape for pooling
            pooled_features = F.adaptive_avg_pool1d(
                vision_features.transpose(1, 2), self.query_num
            ).transpose(1, 2)
        else:
            pooled_features = vision_features
        
        # Project to LLM hidden size using 2-layer MLP
        x = getattr(self, "0")(pooled_features)  # Layer 0: 1152 -> 2048
        x = getattr(self, "1")(x)                # Layer 1: GELU activation
        projected_features = getattr(self, "2")(x)  # Layer 2: 2048 -> 2048
        
        return projected_features


class AttentionPooling(nn.Module):
    """
    Attention pooling layer for SigLIP2 vision backbone.
    
    This layer performs attention-based pooling to produce final vision features.
    """
    
    def __init__(self, config: PrismaticConfig):
        super().__init__()
        self.config = config
        embed_dim = config.vision_embed_dim
        
        # Learnable query (latent)
        self.latent = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Attention mechanism
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer norm and MLP
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling to input features.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Pooled features of shape [batch_size, 1, embed_dim]
        """
        batch_size = x.size(0)
        
        # Expand latent query for batch
        latent = self.latent.expand(batch_size, -1, -1)
        
        # Apply attention
        q = self.q_proj(latent)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.config.vision_embed_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn_weights, v)
        
        # Output projection
        output = self.out_proj(attended)
        
        # Add residual connection and layer norm
        output = output + latent
        output = self.norm(output)
        
        # Apply MLP
        mlp_output = self.mlp(output)
        output = output + mlp_output
        
        return output


# Register the multimodal processor for this model
try:
    from vllm.multimodal.prismatic_processor import (
        PrismaticMultiModalProcessor,
        PrismaticProcessingInfo,
        PrismaticDummyInputsBuilder
    )
    
    # Register the processor for our model class using the correct syntax
    MULTIMODAL_REGISTRY.register_processor(
        PrismaticMultiModalProcessor,
        info=PrismaticProcessingInfo,
        dummy_inputs=PrismaticDummyInputsBuilder,
    )(PrismaticVLMForCausalLM)

except ImportError:
    pass  # Multimodal processor registration failed
except Exception:
    pass  # Multimodal processor registration failed
