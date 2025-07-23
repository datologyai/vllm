# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SigLIP2 Vision Model implementation for vLLM.

This module implements the SigLIP2-so400m vision backbone that processes
378px images into 729 visual tokens (27x27 patches with 14px patch size).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)

logger = init_logger(__name__)


class SigLIP2VisionEmbeddings(nn.Module):
    """
    SigLIP2 vision embeddings module.
    
    Converts input images to patch embeddings with positional encoding.
    """
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        
        # Vision configuration
        self.embed_dim = getattr(config, 'vision_embed_dim', 1152)
        self.image_size = getattr(config, 'image_size', 378)
        self.patch_size = getattr(config, 'patch_size', 14)
        
        # Calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 729
        self.num_positions = self.num_patches
        
        # Patch embedding layer
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        
        # Position embeddings
        self.position_embedding = nn.Embedding(
            self.num_positions, self.embed_dim
        )
        
        # Register position IDs
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision embeddings.
        
        Args:
            pixel_values: [batch_size, 3, 378, 378] input images
            
        Returns:
            torch.Tensor: [batch_size, 729, 1152] patch embeddings
        """
        batch_size = pixel_values.shape[0]
        
        # Extract patches: [B, 1152, 27, 27]
        patch_embeds = self.patch_embedding(pixel_values)
        
        # Flatten patches: [B, 1152, 729]
        patch_embeds = patch_embeds.flatten(2)
        
        # Transpose: [B, 729, 1152]
        patch_embeds = patch_embeds.transpose(1, 2)
        
        # Add position embeddings
        position_ids = self.position_ids[:, :self.num_positions]
        position_embeddings = self.position_embedding(position_ids)
        
        embeddings = patch_embeds + position_embeddings
        
        return embeddings


class SigLIP2Attention(nn.Module):
    """Multi-head attention module for SigLIP2."""
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        
        self.embed_dim = getattr(config, 'vision_embed_dim', 1152)
        self.num_heads = getattr(config, 'vision_num_attention_heads', 16)
        self.head_dim = self.embed_dim // self.num_heads
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(getattr(config, 'vision_attention_dropout', 0.0))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention.
        
        Args:
            hidden_states: [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # Compute QKV
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class SigLIP2MLP(nn.Module):
    """Feed-forward network for SigLIP2."""
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        
        self.embed_dim = getattr(config, 'vision_embed_dim', 1152)
        self.intermediate_size = getattr(config, 'vision_intermediate_size', 4304)
        
        self.fc1 = nn.Linear(self.embed_dim, self.intermediate_size, bias=True)
        self.fc2 = nn.Linear(self.intermediate_size, self.embed_dim, bias=True)
        
        # Activation function
        self.activation = get_act_fn(
            getattr(config, 'vision_hidden_act', 'gelu_pytorch_tanh')
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            hidden_states: [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, embed_dim]
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SigLIP2EncoderLayer(nn.Module):
    """Single transformer encoder layer for SigLIP2."""
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        
        self.embed_dim = getattr(config, 'vision_embed_dim', 1152)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        
        # Attention and MLP
        self.self_attn = SigLIP2Attention(config)
        self.mlp = SigLIP2MLP(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            hidden_states: [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SigLIP2Encoder(nn.Module):
    """Multi-layer transformer encoder for SigLIP2."""
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        
        self.num_layers = getattr(config, 'vision_num_hidden_layers', 27)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            SigLIP2EncoderLayer(config) for _ in range(self.num_layers)
        ])
    
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            inputs_embeds: [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, embed_dim]
        """
        hidden_states = inputs_embeds
        
        # Forward through all encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        return hidden_states


class SigLIP2VisionModel(nn.Module):
    """
    Complete SigLIP2 vision model.
    
    This model processes 378px images into 729 visual tokens using:
    - Patch embedding (14px patches → 27x27 = 729 patches)
    - Multi-layer transformer encoder (27 layers)
    - Final layer normalization
    """
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        
        # Vision embeddings
        self.embeddings = SigLIP2VisionEmbeddings(config)
        
        # Transformer encoder
        self.encoder = SigLIP2Encoder(config)
        
        # Final layer normalization
        self.post_layernorm = nn.LayerNorm(
            getattr(config, 'vision_embed_dim', 1152), eps=1e-6
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision model.
        
        Args:
            pixel_values: [batch_size, 3, 378, 378] input images
            
        Returns:
            torch.Tensor: [batch_size, 729, 1152] vision features
        """
        # Convert to patch embeddings
        hidden_states = self.embeddings(pixel_values)
        
        # Forward through encoder
        hidden_states = self.encoder(hidden_states)
        
        # Final layer normalization
        hidden_states = self.post_layernorm(hidden_states)
        
        return hidden_states