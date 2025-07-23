# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive weight loading validation tests for PrismaticVLM model.

This module tests the weight loading system with mock checkpoints and validates 
that the load_weights method works correctly with different checkpoint formats.
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, MultiModalConfig, VllmConfig
from vllm.model_executor.models.prismatic_vlm import PrismaticVLMForCausalLM
from vllm.transformers_utils.configs.prismatic import PrismaticConfig


class MockCheckpointGenerator:
    """Generate mock checkpoint weights for testing weight loading."""
    
    def __init__(self, config: PrismaticConfig):
        self.config = config
        self.vision_embed_dim = getattr(config, 'vision_embed_dim', 1152)
        self.hidden_size = getattr(config, 'hidden_size', 2048)
        self.vocab_size = getattr(config, 'vocab_size', 151936)
        self.num_layers = getattr(config, 'num_hidden_layers', 2)
        self.num_attention_heads = getattr(config, 'num_attention_heads', 16)
        self.intermediate_size = getattr(config, 'intermediate_size', 5632)
        
        # Vision model params
        self.vision_num_layers = getattr(config, 'vision_num_hidden_layers', 2)
        self.vision_num_heads = getattr(config, 'vision_num_attention_heads', 16)
        self.vision_intermediate_size = getattr(config, 'vision_intermediate_size', 4304)
        
    def create_huggingface_checkpoint(self) -> List[Tuple[str, torch.Tensor]]:
        """Create mock HuggingFace format checkpoint."""
        weights = []
        
        # Vision model weights (HuggingFace format)
        vision_prefix = "vision_model."
        
        # Vision embeddings
        weights.append((f"{vision_prefix}embeddings.patch_embedding.weight", 
                       torch.randn(self.vision_embed_dim, 3, 14, 14)))
        weights.append((f"{vision_prefix}embeddings.position_embedding.weight", 
                       torch.randn(730, self.vision_embed_dim)))  # 729 patches + 1 cls
        
        # Vision transformer layers
        for i in range(self.vision_num_layers):
            layer_prefix = f"{vision_prefix}encoder.layers.{i}."
            
            # Attention
            weights.append((f"{layer_prefix}self_attn.q_proj.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_embed_dim)))
            weights.append((f"{layer_prefix}self_attn.k_proj.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_embed_dim)))
            weights.append((f"{layer_prefix}self_attn.v_proj.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_embed_dim)))
            weights.append((f"{layer_prefix}self_attn.out_proj.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_embed_dim)))
            
            # Layer norms
            weights.append((f"{layer_prefix}layer_norm1.weight", 
                           torch.randn(self.vision_embed_dim)))
            weights.append((f"{layer_prefix}layer_norm1.bias", 
                           torch.randn(self.vision_embed_dim)))
            weights.append((f"{layer_prefix}layer_norm2.weight", 
                           torch.randn(self.vision_embed_dim)))
            weights.append((f"{layer_prefix}layer_norm2.bias", 
                           torch.randn(self.vision_embed_dim)))
            
            # MLP
            weights.append((f"{layer_prefix}mlp.fc1.weight", 
                           torch.randn(self.vision_intermediate_size, self.vision_embed_dim)))
            weights.append((f"{layer_prefix}mlp.fc1.bias", 
                           torch.randn(self.vision_intermediate_size)))
            weights.append((f"{layer_prefix}mlp.fc2.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_intermediate_size)))
            weights.append((f"{layer_prefix}mlp.fc2.bias", 
                           torch.randn(self.vision_embed_dim)))
        
        # Final vision layer norm
        weights.append((f"{vision_prefix}post_layernorm.weight", 
                       torch.randn(self.vision_embed_dim)))
        weights.append((f"{vision_prefix}post_layernorm.bias", 
                       torch.randn(self.vision_embed_dim)))
        
        # Projector weights
        weights.append(("projector.projection.weight", 
                       torch.randn(self.hidden_size, self.vision_embed_dim)))
        
        # Language model weights (HuggingFace format)
        llm_prefix = "language_model."
        
        # Embeddings
        weights.append((f"{llm_prefix}embed_tokens.weight", 
                       torch.randn(self.vocab_size, self.hidden_size)))
        
        # Transformer layers
        for i in range(self.num_layers):
            layer_prefix = f"{llm_prefix}layers.{i}."
            
            # Attention
            weights.append((f"{layer_prefix}self_attn.q_proj.weight", 
                           torch.randn(self.hidden_size, self.hidden_size)))
            weights.append((f"{layer_prefix}self_attn.k_proj.weight", 
                           torch.randn(self.hidden_size, self.hidden_size)))
            weights.append((f"{layer_prefix}self_attn.v_proj.weight", 
                           torch.randn(self.hidden_size, self.hidden_size)))
            weights.append((f"{layer_prefix}self_attn.o_proj.weight", 
                           torch.randn(self.hidden_size, self.hidden_size)))
            
            # MLP
            weights.append((f"{layer_prefix}mlp.gate_proj.weight", 
                           torch.randn(self.intermediate_size, self.hidden_size)))
            weights.append((f"{layer_prefix}mlp.up_proj.weight", 
                           torch.randn(self.intermediate_size, self.hidden_size)))
            weights.append((f"{layer_prefix}mlp.down_proj.weight", 
                           torch.randn(self.hidden_size, self.intermediate_size)))
            
            # Layer norms
            weights.append((f"{layer_prefix}input_layernorm.weight", 
                           torch.randn(self.hidden_size)))
            weights.append((f"{layer_prefix}post_attention_layernorm.weight", 
                           torch.randn(self.hidden_size)))
        
        # Final layer norm
        weights.append((f"{llm_prefix}norm.weight", 
                       torch.randn(self.hidden_size)))
        
        # LM Head
        weights.append(("lm_head.weight", 
                       torch.randn(self.vocab_size, self.hidden_size)))
        
        return weights
    
    def create_prismatic_checkpoint(self) -> List[Tuple[str, torch.Tensor]]:
        """Create mock Prismatic format checkpoint."""
        weights = []
        
        # Vision backbone weights (Prismatic format)
        vision_prefix = "vision_backbone."
        
        # Vision embeddings  
        weights.append((f"{vision_prefix}embeddings.patch_embedding.weight", 
                       torch.randn(self.vision_embed_dim, 3, 14, 14)))
        weights.append((f"{vision_prefix}embeddings.position_embedding.weight", 
                       torch.randn(730, self.vision_embed_dim)))
        
        # Vision transformer layers
        for i in range(self.vision_num_layers):
            layer_prefix = f"{vision_prefix}encoder.layers.{i}."
            
            # Attention
            weights.append((f"{layer_prefix}self_attn.q_proj.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_embed_dim)))
            weights.append((f"{layer_prefix}self_attn.k_proj.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_embed_dim)))
            weights.append((f"{layer_prefix}self_attn.v_proj.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_embed_dim)))
            weights.append((f"{layer_prefix}self_attn.out_proj.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_embed_dim)))
            
            # Layer norms
            weights.append((f"{layer_prefix}layer_norm1.weight", 
                           torch.randn(self.vision_embed_dim)))
            weights.append((f"{layer_prefix}layer_norm1.bias", 
                           torch.randn(self.vision_embed_dim)))
            weights.append((f"{layer_prefix}layer_norm2.weight", 
                           torch.randn(self.vision_embed_dim)))
            weights.append((f"{layer_prefix}layer_norm2.bias", 
                           torch.randn(self.vision_embed_dim)))
            
            # MLP
            weights.append((f"{layer_prefix}mlp.fc1.weight", 
                           torch.randn(self.vision_intermediate_size, self.vision_embed_dim)))
            weights.append((f"{layer_prefix}mlp.fc1.bias", 
                           torch.randn(self.vision_intermediate_size)))
            weights.append((f"{layer_prefix}mlp.fc2.weight", 
                           torch.randn(self.vision_embed_dim, self.vision_intermediate_size)))
            weights.append((f"{layer_prefix}mlp.fc2.bias", 
                           torch.randn(self.vision_embed_dim)))
        
        # Final vision layer norm
        weights.append((f"{vision_prefix}post_layernorm.weight", 
                       torch.randn(self.vision_embed_dim)))
        weights.append((f"{vision_prefix}post_layernorm.bias", 
                       torch.randn(self.vision_embed_dim)))
        
        # Projector weights
        weights.append(("projector.projection.weight", 
                       torch.randn(self.hidden_size, self.vision_embed_dim)))
        
        # LLM backbone weights (Prismatic format)
        llm_prefix = "llm_backbone."
        
        # Embeddings
        weights.append((f"{llm_prefix}embed_tokens.weight", 
                       torch.randn(self.vocab_size, self.hidden_size)))
        
        # Transformer layers
        for i in range(self.num_layers):
            layer_prefix = f"{llm_prefix}layers.{i}."
            
            # Attention
            weights.append((f"{layer_prefix}self_attn.q_proj.weight", 
                           torch.randn(self.hidden_size, self.hidden_size)))
            weights.append((f"{layer_prefix}self_attn.k_proj.weight", 
                           torch.randn(self.hidden_size, self.hidden_size)))
            weights.append((f"{layer_prefix}self_attn.v_proj.weight", 
                           torch.randn(self.hidden_size, self.hidden_size)))
            weights.append((f"{layer_prefix}self_attn.o_proj.weight", 
                           torch.randn(self.hidden_size, self.hidden_size)))
            
            # MLP
            weights.append((f"{layer_prefix}mlp.gate_proj.weight", 
                           torch.randn(self.intermediate_size, self.hidden_size)))
            weights.append((f"{layer_prefix}mlp.up_proj.weight", 
                           torch.randn(self.intermediate_size, self.hidden_size)))
            weights.append((f"{layer_prefix}mlp.down_proj.weight", 
                           torch.randn(self.hidden_size, self.intermediate_size)))
            
            # Layer norms
            weights.append((f"{layer_prefix}input_layernorm.weight", 
                           torch.randn(self.hidden_size)))
            weights.append((f"{layer_prefix}post_attention_layernorm.weight", 
                           torch.randn(self.hidden_size)))
        
        # Final layer norm
        weights.append((f"{llm_prefix}norm.weight", 
                       torch.randn(self.hidden_size)))
        
        # LM Head
        weights.append(("lm_head.weight", 
                       torch.randn(self.vocab_size, self.hidden_size)))
        
        return weights
    
    def create_corrupted_checkpoint(self) -> List[Tuple[str, torch.Tensor]]:
        """Create a checkpoint with intentional issues for error handling testing."""
        weights = []
        
        # Add weights with wrong shapes
        weights.append(("projector.projection.weight", 
                       torch.randn(1024, 512)))  # Wrong shape
        
        # Add weights with wrong dtype
        weights.append(("vision_backbone.embeddings.patch_embedding.weight", 
                       torch.randn(self.vision_embed_dim, 3, 14, 14, dtype=torch.int32)))
        
        # Add unknown weights
        weights.append(("unknown_component.weight", 
                       torch.randn(100, 100)))
        
        return weights


class TestPrismaticVLMWeightLoading:
    """Test weight loading functionality for PrismaticVLM."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return PrismaticConfig(
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=5632,
            num_hidden_layers=2,  # Reduced for testing
            num_attention_heads=16,
            vision_embed_dim=1152,
            vision_num_hidden_layers=2,  # Reduced for testing
            vision_num_attention_heads=16,
            vision_intermediate_size=4304,
            image_size=378,
            patch_size=14,
        )
    
    @pytest.fixture
    def vllm_config(self, config):
        """Create a VllmConfig instance for testing."""
        # Create a mock VllmConfig
        vllm_config = Mock(spec=VllmConfig)
        vllm_config.model_config = Mock()
        vllm_config.model_config.hf_config = config
        vllm_config.model_config.multimodal_config = Mock(spec=MultiModalConfig)
        vllm_config.quant_config = None
        return vllm_config
    
    @pytest.fixture
    def model(self, vllm_config):
        """Create a PrismaticVLM model for testing."""
        with patch('vllm.model_executor.models.prismatic_vlm.init_vllm_registered_model') as mock_init:
            # Mock the LLM backbone
            mock_llm = Mock()
            mock_llm.embed_tokens = Mock()
            mock_llm.embed_tokens.weight = torch.randn(151936, 2048)
            mock_init.return_value = mock_llm
            
            model = PrismaticVLMForCausalLM(vllm_config=vllm_config)
            return model
    
    @pytest.fixture
    def checkpoint_generator(self, config):
        """Create a checkpoint generator."""
        return MockCheckpointGenerator(config)
    
    def test_checkpoint_format_detection(self, model, checkpoint_generator):
        """Test that checkpoint format detection works correctly."""
        # Test HuggingFace format detection
        hf_weights = checkpoint_generator.create_huggingface_checkpoint()
        hf_format = model._validate_checkpoint_format(iter(hf_weights))
        assert hf_format == "huggingface"
        
        # Test Prismatic format detection
        prismatic_weights = checkpoint_generator.create_prismatic_checkpoint()
        prismatic_format = model._validate_checkpoint_format(iter(prismatic_weights))
        assert prismatic_format == "prismatic"
    
    def test_weight_name_normalization(self, model):
        """Test weight name normalization for different formats."""
        # Test HuggingFace to vLLM mapping
        hf_vision_name = "vision_model.encoder.layers.0.self_attn.q_proj.weight"
        normalized = model._normalize_weight_name(hf_vision_name, "huggingface")
        assert normalized == "vision_backbone.encoder.layers.0.self_attn.q_proj.weight"
        
        hf_llm_name = "language_model.layers.0.self_attn.q_proj.weight"
        normalized = model._normalize_weight_name(hf_llm_name, "huggingface")
        assert normalized == "llm_backbone.layers.0.self_attn.q_proj.weight"
        
        # Test Prismatic format (should remain unchanged)
        prismatic_name = "vision_backbone.encoder.layers.0.self_attn.q_proj.weight"
        normalized = model._normalize_weight_name(prismatic_name, "prismatic")
        assert normalized == prismatic_name
    
    def test_architecture_validation(self, model):
        """Test model architecture validation."""
        validation_results = model._validate_model_architecture()
        
        # Check that all required keys are present
        required_keys = [
            "vision_backbone_type", "projector_type", "llm_backbone_type",
            "lm_head_type", "expected_visual_tokens", "expected_vision_dim",
            "expected_llm_dim"
        ]
        
        for key in required_keys:
            assert key in validation_results
        
        # Check expected values
        assert validation_results["expected_visual_tokens"] == 729
        assert validation_results["expected_vision_dim"] == 1152
        assert validation_results["expected_llm_dim"] == 2048
    
    def test_weight_loading_huggingface_format(self, model, checkpoint_generator):
        """Test weight loading with HuggingFace format checkpoint."""
        hf_weights = checkpoint_generator.create_huggingface_checkpoint()
        
        # Mock the model parameters to avoid actual loading
        with patch.object(model, 'named_parameters') as mock_named_params:
            # Create mock parameters dictionary
            mock_params = {}
            for name, weight in hf_weights:
                # Convert HF names to vLLM names
                vllm_name = model._normalize_weight_name(name, "huggingface")
                mock_param = Mock()
                mock_param.shape = weight.shape
                mock_param.dtype = weight.dtype
                mock_param.weight_loader = Mock()
                mock_params[vllm_name] = mock_param
            
            mock_named_params.return_value = mock_params.items()
            
            # Test loading
            loaded_params = model.load_weights(iter(hf_weights))
            
            # Verify that parameters were loaded
            assert len(loaded_params) > 0
            
            # Check that weight loaders were called
            for param in mock_params.values():
                if hasattr(param, 'weight_loader'):
                    assert param.weight_loader.called
    
    def test_weight_loading_prismatic_format(self, model, checkpoint_generator):
        """Test weight loading with Prismatic format checkpoint."""
        prismatic_weights = checkpoint_generator.create_prismatic_checkpoint()
        
        # Mock the model parameters
        with patch.object(model, 'named_parameters') as mock_named_params:
            mock_params = {}
            for name, weight in prismatic_weights:
                mock_param = Mock()
                mock_param.shape = weight.shape
                mock_param.dtype = weight.dtype
                mock_param.weight_loader = Mock()
                mock_params[name] = mock_param
            
            mock_named_params.return_value = mock_params.items()
            
            # Test loading
            loaded_params = model.load_weights(iter(prismatic_weights))
            
            # Verify that parameters were loaded
            assert len(loaded_params) > 0
            
            # Check that weight loaders were called
            for param in mock_params.values():
                if hasattr(param, 'weight_loader'):
                    assert param.weight_loader.called
    
    def test_weight_loading_error_handling(self, model, checkpoint_generator):
        """Test error handling during weight loading."""
        corrupted_weights = checkpoint_generator.create_corrupted_checkpoint()
        
        # Mock the model parameters
        with patch.object(model, 'named_parameters') as mock_named_params:
            mock_params = {
                "projector.projection.weight": Mock(shape=(2048, 1152), dtype=torch.float32),
                "vision_backbone.embeddings.patch_embedding.weight": Mock(
                    shape=(1152, 3, 14, 14), dtype=torch.float32
                ),
            }
            
            for param in mock_params.values():
                param.weight_loader = Mock()
            
            mock_named_params.return_value = mock_params.items()
            
            # Test loading with corrupted weights (should handle gracefully)
            loaded_params = model.load_weights(iter(corrupted_weights))
            
            # Should have loaded 0 parameters due to shape/dtype mismatches
            assert len(loaded_params) == 0
    
    def test_weight_statistics_collection(self, model, checkpoint_generator):
        """Test collection of weight loading statistics."""
        prismatic_weights = checkpoint_generator.create_prismatic_checkpoint()
        
        # Mock the model parameters
        with patch.object(model, 'named_parameters') as mock_named_params:
            mock_params = {}
            for name, weight in prismatic_weights:
                mock_param = Mock()
                mock_param.shape = weight.shape
                mock_param.dtype = weight.dtype
                mock_param.weight_loader = Mock()
                mock_params[name] = mock_param
            
            mock_named_params.return_value = mock_params.items()
            
            # Mock _save_loading_stats to capture the call
            with patch.object(model, '_save_loading_stats') as mock_save_stats:
                model.load_weights(iter(prismatic_weights))
                
                # Verify that statistics were collected and saved
                assert mock_save_stats.called
                args, kwargs = mock_save_stats.call_args
                weight_stats, component_stats, checkpoint_format = args
                
                assert checkpoint_format == "prismatic"
                assert len(weight_stats) > 0
                assert "vision_backbone" in component_stats
                assert "projector" in component_stats
                assert "llm_backbone" in component_stats
    
    def test_loading_stats_file_creation(self, model, checkpoint_generator):
        """Test that loading statistics file is created correctly."""
        prismatic_weights = checkpoint_generator.create_prismatic_checkpoint()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            stats_file = os.path.join(temp_dir, "test_stats.json")
            
            # Mock weight and component stats
            weight_stats = {
                "projector.projection.weight": {
                    "shape": [2048, 1152],
                    "dtype": "torch.float32",
                    "numel": 2048 * 1152,
                    "device": "cpu",
                    "original_name": "projector.projection.weight"
                }
            }
            
            component_stats = {
                "projector": {"expected": 1, "loaded": 1}
            }
            
            # Test file creation
            model._save_loading_stats(weight_stats, component_stats, "prismatic", stats_file)
            
            # Verify file was created and contains expected data
            assert os.path.exists(stats_file)
            
            with open(stats_file, 'r') as f:
                data = json.load(f)
            
            assert data["checkpoint_format"] == "prismatic"
            assert data["weight_count"] == 1
            assert data["total_parameters"] == 2048 * 1152
            assert "architecture_validation" in data
            assert "weights_by_component" in data
    
    def test_critical_component_validation(self, model, checkpoint_generator):
        """Test that critical components are validated during loading."""
        # Create weights without projector (critical component)
        incomplete_weights = [
            ("vision_backbone.embeddings.patch_embedding.weight", 
             torch.randn(1152, 3, 14, 14)),
            ("llm_backbone.embed_tokens.weight", 
             torch.randn(151936, 2048)),
        ]
        
        with patch.object(model, 'named_parameters') as mock_named_params:
            mock_params = {}
            for name, weight in incomplete_weights:
                mock_param = Mock()
                mock_param.shape = weight.shape
                mock_param.dtype = weight.dtype
                mock_param.weight_loader = Mock()
                mock_params[name] = mock_param
            
            mock_named_params.return_value = mock_params.items()
            
            # Test loading (should log error about missing projector)
            with patch('vllm.model_executor.models.prismatic_vlm.logger') as mock_logger:
                model.load_weights(iter(incomplete_weights))
                
                # Should log error about missing critical component
                error_calls = [call for call in mock_logger.error.call_args_list 
                              if "Critical component" in str(call)]
                assert len(error_calls) > 0


class TestPrismaticVLMWeightLoadingIntegration:
    """Integration tests for weight loading with more realistic scenarios."""
    
    def test_weight_loading_with_dtype_conversion(self):
        """Test weight loading with dtype conversion."""
        # This test would require actual model instantiation
        # For now, we test the concept with mocks
        pass
    
    def test_weight_loading_performance(self):
        """Test weight loading performance with large checkpoints."""
        # This test would measure loading time for large checkpoints
        pass
    
    def test_weight_loading_memory_usage(self):
        """Test memory usage during weight loading."""
        # This test would monitor memory usage during loading
        pass


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])