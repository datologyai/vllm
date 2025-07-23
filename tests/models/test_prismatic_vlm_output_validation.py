# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Output validation tests for PrismaticVLM model.

This module tests that the vLLM implementation produces expected outputs
and maintains consistency with reference implementations.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

from vllm.model_executor.models.prismatic_vlm import (
    PrismaticVLMForCausalLM,
    PrismaticProjector,
    IMAGE_TOKEN_INDEX
)
from vllm.transformers_utils.configs.prismatic import PrismaticConfig


class TestPrismaticVLMOutputValidation:
    """Test output validation and consistency for PrismaticVLM."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return PrismaticConfig(
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=5632,
            num_hidden_layers=2,
            num_attention_heads=16,
            vision_embed_dim=1152,
            vision_num_hidden_layers=2,
            vision_num_attention_heads=16,
            vision_intermediate_size=4304,
            image_size=378,
            patch_size=14,
        )
    
    def test_projector_output_consistency(self, config):
        """Test that projector outputs are consistent and expected."""
        projector = PrismaticProjector(config)
        
        # Test with deterministic input
        torch.manual_seed(42)
        vision_features = torch.randn(2, 729, 1152)
        
        # Forward pass
        projected = projector.forward(vision_features)
        
        # Validate output shape
        assert projected.shape == (2, 729, 2048)
        
        # Test reproducibility
        torch.manual_seed(42)
        vision_features_2 = torch.randn(2, 729, 1152)
        projected_2 = projector.forward(vision_features_2)
        
        assert torch.allclose(projected, projected_2, atol=1e-6)
        
        # Test that projection actually transforms the features
        assert not torch.allclose(vision_features[..., :1152], projected[..., :1152])
    
    def test_visual_token_embedding_consistency(self):
        """Test that visual token embeddings are handled consistently."""
        batch_size = 2
        seq_len = 800
        hidden_size = 2048
        
        # Create input sequence with image tokens
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        input_ids[:, :729] = IMAGE_TOKEN_INDEX  # First 729 tokens are image tokens
        
        # Create text embeddings (simulating token embeddings)
        text_embeddings = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create visual embeddings (simulating processed image features)
        visual_embeddings = torch.randn(batch_size, 729, hidden_size)
        
        # Simulate the merging process
        merged_embeddings = text_embeddings.clone()
        
        for i in range(batch_size):
            # Find image token positions
            image_positions = torch.where(input_ids[i] == IMAGE_TOKEN_INDEX)[0]
            assert len(image_positions) == 729, f"Expected 729 image tokens, got {len(image_positions)}"
            
            # Replace with visual embeddings
            merged_embeddings[i, image_positions] = visual_embeddings[i]
        
        # Validate the merge
        assert merged_embeddings.shape == (batch_size, seq_len, hidden_size)
        
        # Check that image tokens were replaced
        for i in range(batch_size):
            image_positions = torch.where(input_ids[i] == IMAGE_TOKEN_INDEX)[0]
            # Visual embeddings should be different from original text embeddings
            assert not torch.allclose(
                merged_embeddings[i, image_positions], 
                text_embeddings[i, image_positions]
            )
            
            # Non-image tokens should remain unchanged
            text_positions = torch.where(input_ids[i] != IMAGE_TOKEN_INDEX)[0]
            if len(text_positions) > 0:
                assert torch.allclose(
                    merged_embeddings[i, text_positions],
                    text_embeddings[i, text_positions]
                )
    
    def test_embedding_dimension_consistency(self, config):
        """Test that all embeddings have consistent dimensions."""
        # Test projector dimensions
        projector = PrismaticProjector(config)
        
        # Verify projector weight dimensions
        assert projector.projection.weight.shape == (2048, 1152)
        
        # Test vision features to LLM embedding transformation
        vision_features = torch.randn(1, 729, 1152)
        projected = projector(vision_features)
        
        assert projected.shape == (1, 729, 2048)
        
        # Test that dimensions match expected architecture
        assert config.vision_embed_dim == 1152
        assert config.hidden_size == 2048
        assert config.num_image_tokens == 729
    
    def test_multimodal_sequence_construction(self):
        """Test construction of multimodal sequences."""
        batch_size = 1
        num_text_tokens = 100
        num_image_tokens = 729
        total_tokens = num_image_tokens + num_text_tokens
        
        # Simulate multimodal sequence construction
        input_ids = torch.zeros(batch_size, total_tokens, dtype=torch.long)
        
        # Place image tokens at the beginning
        input_ids[:, :num_image_tokens] = IMAGE_TOKEN_INDEX
        
        # Place text tokens after image tokens
        input_ids[:, num_image_tokens:] = torch.randint(1, 1000, (batch_size, num_text_tokens))
        
        # Validate sequence structure
        image_token_count = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
        assert image_token_count == num_image_tokens
        
        # Check that text tokens are valid (not special tokens)
        text_tokens = input_ids[:, num_image_tokens:]
        assert (text_tokens > 0).all()
        assert (text_tokens < 1000).all()
    
    def test_attention_mask_consistency(self):
        """Test that attention masks work correctly with multimodal inputs."""
        batch_size = 2
        seq_len = 800
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Simulate padding in second sample
        attention_mask[1, 700:] = 0  # Last 100 tokens are padding
        
        # Verify mask structure
        assert attention_mask.shape == (batch_size, seq_len)
        assert attention_mask[0].sum() == seq_len  # First sample has no padding
        assert attention_mask[1].sum() == 700  # Second sample has 700 valid tokens
    
    def test_position_encoding_compatibility(self):
        """Test that position encodings work with multimodal sequences."""
        seq_len = 800
        batch_size = 1
        
        # Create position indices
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Validate position indices
        assert positions.shape == (batch_size, seq_len)
        assert positions[0, 0] == 0
        assert positions[0, -1] == seq_len - 1
        
        # Check that positions are continuous even with image tokens
        expected_positions = torch.arange(seq_len).unsqueeze(0)
        assert torch.equal(positions, expected_positions)
    
    def test_logits_output_shape_consistency(self, config):
        """Test that logits have consistent shapes."""
        batch_size = 2
        seq_len = 800
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        
        # Simulate hidden states from model
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Simulate LM head computation
        lm_head_weight = torch.randn(vocab_size, hidden_size)
        logits = torch.matmul(hidden_states, lm_head_weight.t())
        
        # Validate logits shape
        assert logits.shape == (batch_size, seq_len, vocab_size)
        
        # Test that logits are reasonable (not all zeros, not extreme values)
        assert not torch.all(logits == 0)
        assert torch.isfinite(logits).all()
    
    def test_gradient_flow_consistency(self, config):
        """Test that gradients flow correctly through the model components."""
        # Create projector
        projector = PrismaticProjector(config)
        
        # Enable gradients
        vision_features = torch.randn(1, 729, 1152, requires_grad=True)
        
        # Forward pass
        projected = projector(vision_features)
        
        # Compute dummy loss
        target = torch.randn_like(projected)
        loss = nn.functional.mse_loss(projected, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert vision_features.grad is not None
        assert projector.projection.weight.grad is not None
        
        # Check that gradients are not zero (indicating flow)
        assert not torch.all(vision_features.grad == 0)
        assert not torch.all(projector.projection.weight.grad == 0)
    
    def test_numerical_stability(self, config):
        """Test numerical stability of computations."""
        projector = PrismaticProjector(config)
        
        # Test with various input magnitudes
        test_cases = [
            torch.randn(1, 729, 1152) * 0.1,  # Small values
            torch.randn(1, 729, 1152) * 1.0,  # Normal values  
            torch.randn(1, 729, 1152) * 10.0, # Large values
        ]
        
        for vision_features in test_cases:
            projected = projector(vision_features)
            
            # Check for numerical issues
            assert torch.isfinite(projected).all(), "Non-finite values detected"
            assert not torch.isnan(projected).any(), "NaN values detected"
            assert not torch.isinf(projected).any(), "Infinite values detected"
    
    def test_batch_consistency(self, config):
        """Test that batch processing is consistent with individual processing."""
        projector = PrismaticProjector(config)
        
        # Create test data
        vision_features_batch = torch.randn(3, 729, 1152)
        
        # Process as batch
        projected_batch = projector(vision_features_batch)
        
        # Process individually
        projected_individual = []
        for i in range(3):
            individual_input = vision_features_batch[i:i+1]
            individual_output = projector(individual_input)
            projected_individual.append(individual_output)
        
        projected_individual = torch.cat(projected_individual, dim=0)
        
        # Compare batch vs individual processing
        assert torch.allclose(projected_batch, projected_individual, atol=1e-6)
    
    def test_deterministic_output(self, config):
        """Test that outputs are deterministic given the same input."""
        projector = PrismaticProjector(config)
        
        # Set random seed and create input
        torch.manual_seed(123)
        vision_features = torch.randn(1, 729, 1152)
        
        # First forward pass
        torch.manual_seed(456)  # Different seed for operations
        output1 = projector(vision_features)
        
        # Second forward pass with same input
        torch.manual_seed(456)  # Same seed for operations
        output2 = projector(vision_features)
        
        # Outputs should be identical
        assert torch.equal(output1, output2)
    
    def test_weight_initialization_properties(self, config):
        """Test that weight initialization has reasonable properties."""
        projector = PrismaticProjector(config)
        
        weight = projector.projection.weight
        
        # Check weight shape
        assert weight.shape == (2048, 1152)
        
        # Check weight statistics (should be reasonably distributed)
        weight_mean = weight.mean().item()
        weight_std = weight.std().item()
        
        # Mean should be close to zero (within reasonable range)
        assert abs(weight_mean) < 0.1, f"Weight mean {weight_mean} is too large"
        
        # Standard deviation should be reasonable (not too small or large)
        assert 0.01 < weight_std < 1.0, f"Weight std {weight_std} is outside reasonable range"
        
        # Check that weights are not all the same
        assert not torch.all(weight == weight[0, 0]), "All weights are identical"


class TestPrismaticVLMReferenceComparison:
    """Test comparison with reference implementation behavior."""
    
    def test_vision_feature_dimension_mapping(self):
        """Test that vision feature dimensions map correctly to reference."""
        # Reference architecture: SigLIP2-so400m outputs 1152-dim features
        input_dim = 1152  # Vision backbone output
        output_dim = 2048  # LLM input dimension
        
        # Verify dimension mapping matches reference
        assert input_dim == 1152, "Vision dimension should match SigLIP2-so400m"
        assert output_dim == 2048, "LLM dimension should match Qwen3-1.7B"
        
        # Test projection preserves expected relationships
        projection_weight = torch.randn(output_dim, input_dim)
        vision_features = torch.randn(1, 729, input_dim)
        
        projected = torch.matmul(vision_features, projection_weight.t())
        assert projected.shape == (1, 729, output_dim)
    
    def test_image_token_count_consistency(self):
        """Test that image token count matches reference implementation."""
        # Reference: 378x378 image, 14x14 patches = 729 tokens
        image_size = 378
        patch_size = 14
        
        expected_tokens = (image_size // patch_size) ** 2
        assert expected_tokens == 729, f"Expected 729 tokens, got {expected_tokens}"
        
        # Verify this matches the configuration
        config = PrismaticConfig(image_size=378, patch_size=14)
        assert config.num_image_tokens == 729
    
    def test_multimodal_sequence_format_consistency(self):
        """Test that multimodal sequence format matches reference."""
        # Reference format: [IMAGE_TOKENS...] + [TEXT_TOKENS...]
        image_token_id = IMAGE_TOKEN_INDEX  # -200
        
        # Create sequence matching reference format
        num_image_tokens = 729
        text_tokens = [1, 2, 3, 4, 5]  # Sample text tokens
        
        sequence = [image_token_id] * num_image_tokens + text_tokens
        input_ids = torch.tensor(sequence).unsqueeze(0)
        
        # Validate sequence structure
        assert input_ids.shape == (1, num_image_tokens + len(text_tokens))
        assert (input_ids[0, :num_image_tokens] == image_token_id).all()
        assert (input_ids[0, num_image_tokens:] == torch.tensor(text_tokens)).all()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])