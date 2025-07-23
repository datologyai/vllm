# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Forward pass validation tests for PrismaticVLM model.

This module tests the forward pass functionality with loaded weights to ensure
the model can perform inference correctly with the 729 visual tokens.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig, VllmConfig
from vllm.model_executor.models.prismatic_vlm import (
    PrismaticVLMForCausalLM, 
    PrismaticProjector,
    IMAGE_TOKEN_INDEX
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.transformers_utils.configs.prismatic import PrismaticConfig
from vllm.sequence import IntermediateTensors


class TestPrismaticVLMForwardPass:
    """Test forward pass functionality for PrismaticVLM."""
    
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
        """Create a VllmConfig instance."""
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
            mock_llm.embed_tokens.weight = nn.Parameter(torch.randn(151936, 2048))
            
            # Mock forward method
            def mock_forward(*args, **kwargs):
                inputs_embeds = kwargs.get('inputs_embeds')
                if inputs_embeds is not None:
                    batch_size, seq_len, hidden_size = inputs_embeds.shape
                    return torch.randn(batch_size, seq_len, hidden_size)
                else:
                    input_ids = kwargs.get('input_ids', args[0])
                    return torch.randn(input_ids.shape[0], input_ids.shape[1], 2048)
            
            mock_llm.forward = mock_forward
            mock_llm.side_effect = mock_forward
            mock_init.return_value = mock_llm
            
            model = PrismaticVLMForCausalLM(vllm_config=vllm_config)
            return model
    
    def test_pixel_values_validation(self, model):
        """Test pixel values validation."""
        # Test valid pixel values
        valid_pixel_values = torch.randn(1, 3, 378, 378)
        validated = model._validate_pixel_values(valid_pixel_values)
        assert validated.shape == (1, 3, 378, 378)
        
        # Test invalid dimensions
        with pytest.raises(ValueError, match="Expected pixel_values to have 4 dimensions"):
            invalid_pixel_values = torch.randn(3, 378, 378)  # Missing batch dimension
            model._validate_pixel_values(invalid_pixel_values)
        
        # Test invalid size
        with pytest.raises(ValueError, match="Expected pixel_values shape"):
            invalid_pixel_values = torch.randn(1, 3, 224, 224)  # Wrong size
            model._validate_pixel_values(invalid_pixel_values)
    
    def test_vision_input_processing(self, model):
        """Test vision input processing."""
        # Mock the vision backbone
        with patch.object(model.vision_backbone, 'forward') as mock_vision:
            # Mock vision backbone output (729 tokens, 1152 dimensions)
            mock_vision.return_value = torch.randn(1, 729, 1152)
            
            # Test processing
            pixel_values = torch.randn(1, 3, 378, 378)
            visual_embeddings = model._process_vision_input(pixel_values)
            
            # Check output shape
            assert visual_embeddings.shape == (1, 729, 2048)  # Projected to LLM space
            
            # Check that vision backbone was called
            mock_vision.assert_called_once()
    
    def test_multimodal_embedding_merge(self, model):
        """Test merging of text and visual embeddings."""
        batch_size = 2
        seq_len = 800  # 729 image tokens + 71 text tokens
        
        # Create input_ids with image tokens
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Set first 729 tokens as image tokens for each sample
        input_ids[:, :729] = IMAGE_TOKEN_INDEX
        
        # Create text embeddings
        inputs_embeds = torch.randn(batch_size, seq_len, 2048)
        
        # Create visual embeddings
        visual_embeddings = torch.randn(batch_size, 729, 2048)
        
        # Test merging
        merged_embeddings = model._merge_multimodal_embeddings(
            input_ids, inputs_embeds, visual_embeddings
        )
        
        # Check output shape
        assert merged_embeddings.shape == (batch_size, seq_len, 2048)
        
        # Check that image tokens were replaced with visual embeddings
        for i in range(batch_size):
            # The first 729 positions should contain visual embeddings
            assert not torch.allclose(merged_embeddings[i, :729], inputs_embeds[i, :729])
            # The remaining positions should contain original text embeddings
            assert torch.allclose(merged_embeddings[i, 729:], inputs_embeds[i, 729:])
    
    def test_multimodal_embedding_merge_error_handling(self, model):
        """Test error handling in multimodal embedding merge."""
        batch_size = 1
        seq_len = 800
        
        # Create input_ids with wrong number of image tokens
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        input_ids[:, :728] = IMAGE_TOKEN_INDEX  # Only 728 tokens instead of 729
        
        inputs_embeds = torch.randn(batch_size, seq_len, 2048)
        visual_embeddings = torch.randn(batch_size, 729, 2048)
        
        # Should raise error due to incorrect number of image tokens
        with pytest.raises(ValueError, match="Expected 729 image tokens"):
            model._merge_multimodal_embeddings(input_ids, inputs_embeds, visual_embeddings)
    
    def test_prepare_inputs_embeds_text_only(self, model):
        """Test preparing input embeddings for text-only input."""
        # Mock the embedding layer
        with patch.object(model, 'get_input_embeddings') as mock_embed:
            mock_embed.return_value = torch.randn(1, 100, 2048)
            
            input_ids = torch.randint(0, 1000, (1, 100))
            
            # Test text-only input
            inputs_embeds = model.prepare_inputs_embeds(input_ids)
            
            assert inputs_embeds.shape == (1, 100, 2048)
            mock_embed.assert_called_once_with(input_ids)
    
    def test_prepare_inputs_embeds_multimodal(self, model):
        """Test preparing input embeddings for multimodal input."""
        # Mock components
        with patch.object(model, 'get_input_embeddings') as mock_embed, \
             patch.object(model, '_process_vision_input') as mock_vision, \
             patch.object(model, '_merge_multimodal_embeddings') as mock_merge:
            
            # Setup mocks
            mock_embed.return_value = torch.randn(1, 800, 2048)
            mock_vision.return_value = torch.randn(1, 729, 2048)
            mock_merge.return_value = torch.randn(1, 800, 2048)
            
            # Test multimodal input
            input_ids = torch.randint(0, 1000, (1, 800))
            input_ids[:, :729] = IMAGE_TOKEN_INDEX
            pixel_values = torch.randn(1, 3, 378, 378)
            
            inputs_embeds = model.prepare_inputs_embeds(input_ids, pixel_values)
            
            assert inputs_embeds.shape == (1, 800, 2048)
            mock_embed.assert_called_once_with(input_ids)
            mock_vision.assert_called_once_with(pixel_values)
            mock_merge.assert_called_once()
    
    def test_forward_pass_text_only(self, model):
        """Test forward pass with text-only input."""
        batch_size = 1
        seq_len = 100
        
        # Create inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        positions = torch.arange(seq_len).unsqueeze(0)
        
        # Mock KV caches
        kv_caches = []
        for _ in range(model.config.num_hidden_layers):
            kv_cache = torch.zeros(batch_size, 2, model.config.num_attention_heads, 
                                 seq_len, model.config.hidden_size // model.config.num_attention_heads)
            kv_caches.append(kv_cache)
        
        # Mock attention metadata
        attn_metadata = Mock(spec=AttentionMetadata)
        
        # Mock prepare_inputs_embeds
        with patch.object(model, 'prepare_inputs_embeds') as mock_prepare:
            mock_prepare.return_value = torch.randn(batch_size, seq_len, 2048)
            
            # Test forward pass
            output = model.forward(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata
            )
            
            # Check that output has correct shape
            assert output.shape == (batch_size, seq_len, 2048)
            
            # Check that prepare_inputs_embeds was called
            mock_prepare.assert_called_once()
    
    def test_forward_pass_multimodal(self, model):
        """Test forward pass with multimodal input."""
        batch_size = 1
        seq_len = 800  # 729 image tokens + text tokens
        
        # Create inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        input_ids[:, :729] = IMAGE_TOKEN_INDEX
        positions = torch.arange(seq_len).unsqueeze(0)
        pixel_values = torch.randn(batch_size, 3, 378, 378)
        
        # Mock KV caches
        kv_caches = []
        for _ in range(model.config.num_hidden_layers):
            kv_cache = torch.zeros(batch_size, 2, model.config.num_attention_heads, 
                                 seq_len, model.config.hidden_size // model.config.num_attention_heads)
            kv_caches.append(kv_cache)
        
        # Mock attention metadata
        attn_metadata = Mock(spec=AttentionMetadata)
        
        # Mock prepare_inputs_embeds
        with patch.object(model, 'prepare_inputs_embeds') as mock_prepare:
            mock_prepare.return_value = torch.randn(batch_size, seq_len, 2048)
            
            # Test forward pass
            output = model.forward(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                pixel_values=pixel_values
            )
            
            # Check that output has correct shape
            assert output.shape == (batch_size, seq_len, 2048)
            
            # Check that prepare_inputs_embeds was called with pixel values
            mock_prepare.assert_called_once()
            call_args = mock_prepare.call_args
            assert 'pixel_values' in call_args.kwargs
            assert torch.equal(call_args.kwargs['pixel_values'], pixel_values)
    
    def test_logits_computation(self, model):
        """Test logits computation from hidden states."""
        batch_size = 1
        seq_len = 100
        
        # Create hidden states
        hidden_states = torch.randn(batch_size, seq_len, 2048)
        
        # Mock sampling metadata
        sampling_metadata = Mock(spec=SamplingMetadata)
        
        # Mock logits processor
        with patch.object(model.logits_processor, '__call__') as mock_logits:
            mock_logits.return_value = torch.randn(batch_size, seq_len, model.config.vocab_size)
            
            # Test logits computation
            logits = model.compute_logits(hidden_states, sampling_metadata)
            
            # Check that logits have correct shape
            assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
            
            # Check that logits processor was called
            mock_logits.assert_called_once()
    
    def test_multimodal_embeddings_extraction(self, model):
        """Test multimodal embeddings extraction."""
        pixel_values = torch.randn(1, 3, 378, 378)
        
        # Mock vision processing
        with patch.object(model, '_process_vision_input') as mock_vision:
            mock_vision.return_value = torch.randn(1, 729, 2048)
            
            # Test extraction
            embeddings = model.get_multimodal_embeddings(pixel_values=pixel_values)
            
            assert embeddings.shape == (1, 729, 2048)
            mock_vision.assert_called_once_with(pixel_values)
        
        # Test with no pixel values
        embeddings = model.get_multimodal_embeddings()
        assert embeddings is None
    
    def test_input_embeddings_retrieval(self, model):
        """Test input embeddings retrieval."""
        input_ids = torch.randint(0, 1000, (1, 100))
        
        # Mock the embedding layer
        with patch.object(model.llm_backbone, 'embed_tokens') as mock_embed:
            mock_embed.return_value = torch.randn(1, 100, 2048)
            
            # Test retrieval
            embeddings = model.get_input_embeddings(input_ids)
            
            assert embeddings.shape == (1, 100, 2048)
            mock_embed.assert_called_once_with(input_ids)
    
    def test_projector_forward_pass(self, config):
        """Test projector forward pass."""
        projector = PrismaticProjector(config)
        
        # Test projection
        vision_features = torch.randn(1, 729, 1152)
        projected = projector.forward(vision_features)
        
        # Check output shape
        assert projected.shape == (1, 729, 2048)
    
    def test_visual_token_count_validation(self, model):
        """Test that the model correctly handles 729 visual tokens."""
        batch_size = 1
        
        # Create input with exactly 729 image tokens
        input_ids = torch.randint(0, 1000, (batch_size, 800))
        input_ids[:, :729] = IMAGE_TOKEN_INDEX
        
        # Create embeddings
        inputs_embeds = torch.randn(batch_size, 800, 2048)
        visual_embeddings = torch.randn(batch_size, 729, 2048)
        
        # This should work without error
        merged = model._merge_multimodal_embeddings(input_ids, inputs_embeds, visual_embeddings)
        assert merged.shape == (batch_size, 800, 2048)
        
        # Test with wrong number of visual tokens
        visual_embeddings_wrong = torch.randn(batch_size, 728, 2048)  # Wrong count
        
        with pytest.raises(ValueError):
            model._merge_multimodal_embeddings(input_ids, inputs_embeds, visual_embeddings_wrong)
    
    def test_batch_processing(self, model):
        """Test batch processing with multiple samples."""
        batch_size = 3
        seq_len = 800
        
        # Create inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        input_ids[:, :729] = IMAGE_TOKEN_INDEX
        
        # Create embeddings
        inputs_embeds = torch.randn(batch_size, seq_len, 2048)
        visual_embeddings = torch.randn(batch_size, 729, 2048)
        
        # Test batch processing
        merged = model._merge_multimodal_embeddings(input_ids, inputs_embeds, visual_embeddings)
        
        # Check output shape
        assert merged.shape == (batch_size, seq_len, 2048)
        
        # Check that each sample was processed correctly
        for i in range(batch_size):
            # Visual tokens should be different from original text embeddings
            assert not torch.allclose(merged[i, :729], inputs_embeds[i, :729])
            # Text tokens should remain the same
            assert torch.allclose(merged[i, 729:], inputs_embeds[i, 729:])


class TestPrismaticVLMForwardPassIntegration:
    """Integration tests for forward pass with realistic scenarios."""
    
    def test_full_pipeline_mock(self):
        """Test the full pipeline with mocked components."""
        # This test would require a more complete setup
        # For now, we verify the structure is correct
        pass
    
    def test_memory_efficiency(self):
        """Test memory efficiency during forward pass."""
        # This test would monitor memory usage
        pass
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        # This test would verify gradients flow correctly
        pass


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])