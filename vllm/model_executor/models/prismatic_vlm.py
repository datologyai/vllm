# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PrismaticVLM model implementation for vLLM.

This module implements the SigLIP2-Qwen3-1.7B model architecture with 729 visual tokens.
Architecture: SigLIP2-so400m (1152 dim) → full-align+729-avgpool → Qwen3-1.7B (2048 dim)
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs import INPUT_REGISTRY, InputContext
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalInputs
# Processor imports moved to avoid circular dependencies
from vllm.sequence import IntermediateTensors, SequenceData

from .interfaces import SupportsMultiModal, SupportsPP
from .qwen3 import Qwen3ForCausalLM, Qwen3Model
from .siglip2_vision import SigLIP2VisionModel

logger = init_logger(__name__)

# Image token used as placeholder in text sequences
IMAGE_TOKEN_INDEX = -200


class PrismaticProjector(nn.Module):
    """
    Projector module that maps vision features to LLM input space.
    
    For the 'full-align+729-avgpool' architecture, this is an identity pooling
    that preserves all 729 visual tokens without reduction.
    """
    
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        
        # Vision embedding dimension (SigLIP2-so400m: 1152)
        vision_dim = getattr(config, 'vision_embed_dim', 1152)
        
        # LLM hidden dimension (Qwen3-1.7B: 2048)
        llm_dim = getattr(config, 'hidden_size', 2048)
        
        # Linear projection from vision to LLM space
        self.projection = nn.Linear(vision_dim, llm_dim, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.kaiming_uniform_(self.projection.weight)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to LLM input space.
        
        Args:
            vision_features: [batch_size, 729, 1152] vision features
            
        Returns:
            torch.Tensor: [batch_size, 729, 2048] projected features
        """
        return self.projection(vision_features)


class PrismaticVLMForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    """
    PrismaticVLM model for causal language modeling with vision input.
    
    This model combines:
    - SigLIP2-so400m vision backbone (378px, 14px patches → 729 tokens)
    - Identity pooling projector (preserves all 729 tokens)
    - Qwen3-1.7B language model backbone
    """
    
    supports_multimodal: bool = True
    
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        super().__init__()
        
        # Extract configs from VllmConfig
        config: PretrainedConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        quant_config = vllm_config.quant_config
        
        self.config = config
        self.multimodal_config = multimodal_config
        
        # Vision backbone - SigLIP2-so400m
        self.vision_backbone = SigLIP2VisionModel(config)
        
        # Projector - maps 1152 → 2048 dimensions
        self.projector = PrismaticProjector(config)
        
        # Language model backbone - Qwen3-1.7B
        # Use the same pattern as Qwen2VL for creating the language model
        from .utils import init_vllm_registered_model, maybe_prefix
        self.llm_backbone = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "llm_backbone"),
            architectures=["Qwen3ForCausalLM"],
        )
        
        # Language model head
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.llm_backbone.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                quant_config=quant_config,
            )
        
        # Logits processor and sampler
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
    
    def _validate_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Validate and process pixel values input."""
        if pixel_values.dim() != 4:
            raise ValueError(
                f"Expected pixel_values to have 4 dimensions [B, C, H, W], "
                f"got {pixel_values.dim()} dimensions with shape {pixel_values.shape}"
            )
        
        expected_shape = (pixel_values.shape[0], 3, 378, 378)
        if pixel_values.shape[1:] != expected_shape[1:]:
            raise ValueError(
                f"Expected pixel_values shape [..., 3, 378, 378], "
                f"got {pixel_values.shape}"
            )
        
        return pixel_values
    
    def _process_vision_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process vision input through the vision backbone and projector."""
        # Validate input
        pixel_values = self._validate_pixel_values(pixel_values)
        
        # Extract vision features: [B, 729, 1152]
        vision_features = self.vision_backbone(pixel_values)
        
        # Project to LLM space: [B, 729, 2048]
        visual_embeddings = self.projector(vision_features)
        
        return visual_embeddings
    
    def _merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        visual_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Merge text and visual embeddings based on image token positions."""
        batch_size = input_ids.shape[0]
        
        # Find positions of image tokens (IMAGE_TOKEN_INDEX = -200)
        image_token_mask = input_ids == IMAGE_TOKEN_INDEX
        
        # Process each sample in the batch
        merged_embeddings = []
        
        for i in range(batch_size):
            sample_input_ids = input_ids[i]
            sample_inputs_embeds = inputs_embeds[i]
            sample_visual_embeds = visual_embeddings[i]  # [729, 2048]
            sample_image_mask = image_token_mask[i]
            
            if sample_image_mask.sum() == 0:
                # No image tokens, use text embeddings only
                merged_embeddings.append(sample_inputs_embeds)
            else:
                # Replace image tokens with visual embeddings
                merged_embeds = sample_inputs_embeds.clone()
                
                # Get image token positions
                image_positions = torch.where(sample_image_mask)[0]
                
                # Ensure we have exactly 729 image tokens
                if len(image_positions) != 729:
                    raise ValueError(
                        f"Expected 729 image tokens, got {len(image_positions)}"
                    )
                
                # Replace image tokens with visual embeddings
                merged_embeds[image_positions] = sample_visual_embeds
                
                merged_embeddings.append(merged_embeds)
        
        return torch.stack(merged_embeddings)
    
    def get_multimodal_embeddings(self, **kwargs) -> Optional[torch.Tensor]:
        """Extract multimodal embeddings from pixel values."""
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return None
        
        return self._process_vision_input(pixel_values)
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings for token IDs."""
        return self.llm_backbone.embed_tokens(input_ids)
    
    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Prepare input embeddings by merging text and visual embeddings."""
        # Get text embeddings
        inputs_embeds = self.get_input_embeddings(input_ids)
        
        # If no pixel values, return text embeddings only
        if pixel_values is None:
            return inputs_embeds
        
        # Process visual input
        visual_embeddings = self._process_vision_input(pixel_values)
        
        # Merge text and visual embeddings
        merged_embeddings = self._merge_multimodal_embeddings(
            input_ids, inputs_embeds, visual_embeddings
        )
        
        return merged_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass through the model."""
        # Prepare input embeddings (merging text and visual)
        inputs_embeds = self.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )
        
        # Forward through LLM backbone
        hidden_states = self.llm_backbone(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits from hidden states."""
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits
    
    def _validate_checkpoint_format(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> str:
        """
        Detect the checkpoint format based on weight names.
        
        Returns:
            str: Format type - "huggingface", "prismatic", or "vllm"
        """
        weight_names = [name for name, _ in weights]
        
        # Check for HuggingFace format indicators
        if any("vision_model." in name for name in weight_names):
            return "huggingface"
        elif any("vision_backbone." in name for name in weight_names):
            if any("llm_backbone." in name for name in weight_names):
                return "prismatic"  # Original prismatic format
            else:
                return "vllm"  # vLLM format
        else:
            # Default to prismatic if unclear
            return "prismatic"
    
    def _get_expected_weight_prefixes(self) -> Dict[str, List[str]]:
        """Get expected weight prefixes for each component."""
        return {
            "vision_backbone": [
                "vision_backbone.",
                "vision_model.",  # HF format
            ],
            "projector": [
                "projector.",
            ],
            "llm_backbone": [
                "llm_backbone.", 
                "language_model.",  # HF format
            ],
            "lm_head": [
                "lm_head.",
            ]
        }
    
    def _normalize_weight_name(self, name: str, checkpoint_format: str) -> str:
        """
        Normalize weight names to vLLM format based on checkpoint format.
        
        Args:
            name: Original weight name
            checkpoint_format: Detected format type
            
        Returns:
            str: Normalized weight name
        """
        if checkpoint_format == "huggingface":
            # HuggingFace to vLLM mapping
            if "vision_model." in name:
                return name.replace("vision_model.", "vision_backbone.")
            elif "language_model." in name:
                return name.replace("language_model.", "llm_backbone.")
        elif checkpoint_format == "prismatic":
            # Prismatic format usually matches vLLM already
            pass
            
        return name
    
    def _validate_model_architecture(self) -> Dict[str, Any]:
        """
        Validate that the model has the expected architecture and return key metrics.
        
        Returns:
            Dict with architecture validation results
        """
        validation_results = {
            "vision_backbone_type": type(self.vision_backbone).__name__,
            "projector_type": type(self.projector).__name__,
            "llm_backbone_type": type(self.llm_backbone).__name__,
            "lm_head_type": type(self.lm_head).__name__,
            "expected_visual_tokens": 729,  # For SigLIP2-so400m
            "expected_vision_dim": 1152,    # SigLIP2-so400m output dim
            "expected_llm_dim": 2048,       # Qwen3-1.7B hidden dim
        }
        
        # Validate projector dimensions if it has linear layer
        if hasattr(self.projector, "projection"):
            proj_weight = self.projector.projection.weight
            validation_results["projector_input_dim"] = proj_weight.shape[1]
            validation_results["projector_output_dim"] = proj_weight.shape[0]
            validation_results["projector_dims_correct"] = (
                proj_weight.shape[1] == validation_results["expected_vision_dim"] and
                proj_weight.shape[0] == validation_results["expected_llm_dim"]
            )
        
        # Validate vision backbone output dimensions
        if hasattr(self.vision_backbone, "embed_dim"):
            validation_results["vision_embed_dim"] = self.vision_backbone.embed_dim
            validation_results["vision_dim_correct"] = (
                self.vision_backbone.embed_dim == validation_results["expected_vision_dim"]
            )
        
        # Validate LLM backbone dimensions
        if hasattr(self.llm_backbone, "hidden_size"):
            validation_results["llm_hidden_size"] = self.llm_backbone.hidden_size
            validation_results["llm_dim_correct"] = (
                self.llm_backbone.hidden_size == validation_results["expected_llm_dim"]
            )
        
        return validation_results
    
    def _save_loading_stats(self, weight_stats: Dict[str, Dict], component_stats: Dict, 
                           checkpoint_format: str, output_path: Optional[str] = None) -> None:
        """
        Save detailed loading statistics to a file for debugging.
        
        Args:
            weight_stats: Per-weight statistics
            component_stats: Per-component statistics
            checkpoint_format: Detected checkpoint format
            output_path: Optional path to save stats file
        """
        import json
        import os
        
        stats_data = {
            "checkpoint_format": checkpoint_format,
            "component_stats": component_stats,
            "weight_count": len(weight_stats),
            "total_parameters": sum(stat["numel"] for stat in weight_stats.values()),
            "architecture_validation": self._validate_model_architecture(),
            "weights_by_component": {},
            "dtype_distribution": {},
            "shape_distribution": {}
        }
        
        # Group weights by component
        for name, stat in weight_stats.items():
            component = "unknown"
            if name.startswith("vision_backbone."):
                component = "vision_backbone"
            elif name.startswith("projector."):
                component = "projector"
            elif name.startswith("llm_backbone."):
                component = "llm_backbone"
            elif name.startswith("lm_head."):
                component = "lm_head"
            
            if component not in stats_data["weights_by_component"]:
                stats_data["weights_by_component"][component] = []
            
            stats_data["weights_by_component"][component].append({
                "name": name,
                "original_name": stat["original_name"],
                "shape": stat["shape"],
                "dtype": stat["dtype"],
                "numel": stat["numel"]
            })
        
        # Analyze dtype distribution
        for stat in weight_stats.values():
            dtype = stat["dtype"]
            if dtype not in stats_data["dtype_distribution"]:
                stats_data["dtype_distribution"][dtype] = 0
            stats_data["dtype_distribution"][dtype] += 1
        
        # Analyze shape distribution
        for stat in weight_stats.values():
            shape_str = str(stat["shape"])
            if shape_str not in stats_data["shape_distribution"]:
                stats_data["shape_distribution"][shape_str] = 0
            stats_data["shape_distribution"][shape_str] += 1
        
        # Save to file
        if output_path is None:
            output_path = f"prismatic_vlm_loading_stats_{checkpoint_format}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(stats_data, f, indent=2)
            logger.info(f"Saved loading statistics to: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save loading statistics: {e}")
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Sample from logits."""
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load model weights from checkpoint with comprehensive error handling and logging.
        
        Supports loading from different checkpoint formats:
        1. HuggingFace format: vision_model, language_model, projector, lm_head
        2. Prismatic format: vision_backbone, llm_backbone, projector, lm_head
        3. Direct vLLM format: vision_backbone, llm_backbone, projector, lm_head
        
        Args:
            weights: Iterable of (name, tensor) pairs from checkpoint
        """
        logger.info("Starting PrismaticVLM weight loading...")
        
        # Convert iterator to list to allow multiple passes
        weights_list = list(weights)
        
        # Detect checkpoint format
        checkpoint_format = self._validate_checkpoint_format(iter(weights_list))
        logger.info(f"Detected checkpoint format: {checkpoint_format}")
        
        # Get model parameters for validation
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        skipped_params = set()
        weight_stats = {}
        
        # Track component loading
        component_stats = {
            "vision_backbone": {"expected": 0, "loaded": 0},
            "projector": {"expected": 0, "loaded": 0}, 
            "llm_backbone": {"expected": 0, "loaded": 0},
            "lm_head": {"expected": 0, "loaded": 0}
        }
        
        # Count expected parameters per component
        for param_name in params_dict.keys():
            if param_name.startswith("vision_backbone."):
                component_stats["vision_backbone"]["expected"] += 1
            elif param_name.startswith("projector."):
                component_stats["projector"]["expected"] += 1
            elif param_name.startswith("llm_backbone."):
                component_stats["llm_backbone"]["expected"] += 1
            elif param_name.startswith("lm_head."):
                component_stats["lm_head"]["expected"] += 1
        
        logger.info(f"Expected parameters per component: {component_stats}")
        
        # Process each weight
        for name, loaded_weight in weights_list:
            original_name = name
            
            # Normalize weight name based on detected format
            name = self._normalize_weight_name(name, checkpoint_format)
            
            if name != original_name:
                logger.debug(f"Mapped weight: {original_name} -> {name}")
            
            # Skip certain weights that are not needed or computed dynamically
            skip_patterns = [
                "rotary_emb.inv_freq",
                "rotary_emb.cos_cached", 
                "rotary_emb.sin_cached",
                "attention.bias",
                "attention.masked_bias",
                "embed_positions.weight",  # Position embeddings in some models
            ]
            
            should_skip = any(pattern in name for pattern in skip_patterns)
            if should_skip:
                skipped_params.add(name)
                logger.debug(f"Skipping parameter: {name}")
                continue
                
            # Check if parameter exists in model
            if name not in params_dict:
                logger.warning(f"Parameter {name} not found in model (original: {original_name})")
                continue
                
            param = params_dict[name]
            
            # Validate weight shapes
            if param.shape != loaded_weight.shape:
                logger.error(
                    f"Shape mismatch for {name}: "
                    f"expected {param.shape}, got {loaded_weight.shape}"
                )
                continue
                
            # Validate weight dtype compatibility
            if loaded_weight.dtype != param.dtype:
                logger.info(
                    f"Converting dtype for {name}: {loaded_weight.dtype} -> {param.dtype}"
                )
                loaded_weight = loaded_weight.to(param.dtype)
                
            # Load weight using appropriate loader
            try:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                
                # Track statistics
                weight_stats[name] = {
                    "shape": tuple(loaded_weight.shape),
                    "dtype": str(loaded_weight.dtype),
                    "numel": loaded_weight.numel(),
                    "device": str(loaded_weight.device),
                    "original_name": original_name
                }
                
                # Update component stats
                if name.startswith("vision_backbone."):
                    component_stats["vision_backbone"]["loaded"] += 1
                elif name.startswith("projector."):
                    component_stats["projector"]["loaded"] += 1
                elif name.startswith("llm_backbone."):
                    component_stats["llm_backbone"]["loaded"] += 1
                elif name.startswith("lm_head."):
                    component_stats["lm_head"]["loaded"] += 1
                    
                logger.debug(f"Loaded weight: {name} {loaded_weight.shape}")
                
            except Exception as e:
                logger.error(f"Failed to load weight {name}: {e}")
                continue
        
        # Validate that all required weights were loaded
        missing_params = []
        for param_name in params_dict.keys():
            if param_name not in loaded_params and param_name not in skipped_params:
                missing_params.append(param_name)
        
        if missing_params:
            logger.warning(f"Missing {len(missing_params)} parameters: {missing_params[:10]}...")
            
        # Log comprehensive loading statistics
        logger.info("=== PrismaticVLM Weight Loading Summary ===")
        logger.info(f"Total parameters expected: {len(params_dict)}")
        logger.info(f"Total parameters loaded: {len(loaded_params)}")
        logger.info(f"Total parameters skipped: {len(skipped_params)}")
        logger.info(f"Total parameters missing: {len(missing_params)}")
        
        # Log component-wise statistics
        for component, stats in component_stats.items():
            expected = stats["expected"]
            loaded = stats["loaded"]
            coverage = (loaded / expected * 100) if expected > 0 else 0
            logger.info(f"{component}: {loaded}/{expected} ({coverage:.1f}%)")
            
        # Log weight statistics by component
        total_params = sum(stat["numel"] for stat in weight_stats.values())
        logger.info(f"Total parameters loaded: {total_params:,}")
        
        # Group weights by component for detailed logging
        component_weights = {
            "vision_backbone": [],
            "projector": [],
            "llm_backbone": [], 
            "lm_head": []
        }
        
        for name, stat in weight_stats.items():
            if name.startswith("vision_backbone."):
                component_weights["vision_backbone"].append((name, stat))
            elif name.startswith("projector."):
                component_weights["projector"].append((name, stat))
            elif name.startswith("llm_backbone."):
                component_weights["llm_backbone"].append((name, stat))
            elif name.startswith("lm_head."):
                component_weights["lm_head"].append((name, stat))
        
        # Log detailed weight information for each component
        for component, weights_list in component_weights.items():
            if weights_list:
                logger.info(f"\n{component.upper()} weights:")
                component_params = sum(stat["numel"] for _, stat in weights_list)
                logger.info(f"  Total parameters: {component_params:,}")
                
                # Show a few example weights
                for name, stat in weights_list[:3]:
                    logger.info(f"  {name}: {stat['shape']} ({stat['dtype']})")
                    
                if len(weights_list) > 3:
                    logger.info(f"  ... and {len(weights_list) - 3} more weights")
                    
        # Verify critical components are loaded
        critical_components = ["projector", "llm_backbone"]
        for component in critical_components:
            if component_stats[component]["loaded"] == 0:
                logger.error(f"Critical component {component} has no loaded weights!")
                
        # Log any dtype inconsistencies
        dtypes_found = set(stat["dtype"] for stat in weight_stats.values())
        if len(dtypes_found) > 1:
            logger.info(f"Multiple dtypes found: {dtypes_found}")
            
        # Validate model architecture after loading
        logger.info("\n=== Model Architecture Validation ===")
        arch_validation = self._validate_model_architecture()
        
        logger.info(f"Vision Backbone: {arch_validation['vision_backbone_type']}")
        logger.info(f"Projector: {arch_validation['projector_type']}")
        logger.info(f"LLM Backbone: {arch_validation['llm_backbone_type']}")
        logger.info(f"LM Head: {arch_validation['lm_head_type']}")
        
        # Log dimension validation results
        if "projector_dims_correct" in arch_validation:
            dims_ok = arch_validation["projector_dims_correct"]
            proj_in = arch_validation.get("projector_input_dim", "unknown")
            proj_out = arch_validation.get("projector_output_dim", "unknown")
            logger.info(f"Projector dimensions: {proj_in} -> {proj_out} ({'✓' if dims_ok else '✗'})")
        
        if "vision_dim_correct" in arch_validation:
            vision_ok = arch_validation["vision_dim_correct"]
            vision_dim = arch_validation.get("vision_embed_dim", "unknown")
            logger.info(f"Vision embedding dim: {vision_dim} ({'✓' if vision_ok else '✗'})")
        
        if "llm_dim_correct" in arch_validation:
            llm_ok = arch_validation["llm_dim_correct"]
            llm_dim = arch_validation.get("llm_hidden_size", "unknown")
            logger.info(f"LLM hidden dim: {llm_dim} ({'✓' if llm_ok else '✗'})")
        
        # Save detailed loading statistics for debugging
        try:
            self._save_loading_stats(weight_stats, component_stats, checkpoint_format)
        except Exception as e:
            logger.warning(f"Failed to save loading statistics: {e}")
        
        logger.info("=== Weight Loading Complete ===")
        
        return loaded_params


# Input processing is handled by the new multimodal pipeline
# The registration pattern has changed in newer vLLM versions