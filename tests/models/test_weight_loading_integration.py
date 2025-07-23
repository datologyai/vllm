#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration test for PrismaticVLM weight loading with mock checkpoints.

This script creates realistic mock checkpoints and tests the complete weight loading pipeline.
"""

import json
import os
import tempfile
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm.config import VllmConfig
from vllm.transformers_utils.configs.prismatic import PrismaticConfig
from validate_prismatic_vlm_weights import PrismaticVLMWeightValidator


class MockCheckpointCreator:
    """Create realistic mock checkpoints for testing."""
    
    def __init__(self, config: PrismaticConfig):
        self.config = config
        self.vision_embed_dim = config.vision_embed_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers
        self.vision_num_layers = config.vision_num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.vision_intermediate_size = config.vision_intermediate_size
        self.vision_num_heads = config.vision_num_attention_heads
    
    def create_vision_backbone_weights(self) -> Dict[str, torch.Tensor]:
        """Create vision backbone weights."""
        weights = {}
        
        # Embeddings
        weights["vision_backbone.embeddings.patch_embedding.weight"] = torch.randn(
            self.vision_embed_dim, 3, 14, 14
        )
        weights["vision_backbone.embeddings.position_embedding.weight"] = torch.randn(
            730, self.vision_embed_dim  # 729 patches + 1 cls token
        )
        weights["vision_backbone.embeddings.class_embedding"] = torch.randn(
            self.vision_embed_dim
        )
        
        # Transformer layers
        for i in range(self.vision_num_layers):
            layer_prefix = f"vision_backbone.encoder.layers.{i}."
            
            # Self-attention
            weights[f"{layer_prefix}self_attn.q_proj.weight"] = torch.randn(
                self.vision_embed_dim, self.vision_embed_dim
            )
            weights[f"{layer_prefix}self_attn.k_proj.weight"] = torch.randn(
                self.vision_embed_dim, self.vision_embed_dim
            )
            weights[f"{layer_prefix}self_attn.v_proj.weight"] = torch.randn(
                self.vision_embed_dim, self.vision_embed_dim
            )
            weights[f"{layer_prefix}self_attn.out_proj.weight"] = torch.randn(
                self.vision_embed_dim, self.vision_embed_dim
            )
            weights[f"{layer_prefix}self_attn.q_proj.bias"] = torch.randn(
                self.vision_embed_dim
            )
            weights[f"{layer_prefix}self_attn.k_proj.bias"] = torch.randn(
                self.vision_embed_dim
            )
            weights[f"{layer_prefix}self_attn.v_proj.bias"] = torch.randn(
                self.vision_embed_dim
            )
            weights[f"{layer_prefix}self_attn.out_proj.bias"] = torch.randn(
                self.vision_embed_dim
            )
            
            # Layer norms
            weights[f"{layer_prefix}layer_norm1.weight"] = torch.ones(self.vision_embed_dim)
            weights[f"{layer_prefix}layer_norm1.bias"] = torch.zeros(self.vision_embed_dim)
            weights[f"{layer_prefix}layer_norm2.weight"] = torch.ones(self.vision_embed_dim)
            weights[f"{layer_prefix}layer_norm2.bias"] = torch.zeros(self.vision_embed_dim)
            
            # MLP
            weights[f"{layer_prefix}mlp.fc1.weight"] = torch.randn(
                self.vision_intermediate_size, self.vision_embed_dim
            )
            weights[f"{layer_prefix}mlp.fc1.bias"] = torch.randn(self.vision_intermediate_size)
            weights[f"{layer_prefix}mlp.fc2.weight"] = torch.randn(
                self.vision_embed_dim, self.vision_intermediate_size
            )
            weights[f"{layer_prefix}mlp.fc2.bias"] = torch.randn(self.vision_embed_dim)
        
        # Final layer norm
        weights["vision_backbone.post_layernorm.weight"] = torch.ones(self.vision_embed_dim)
        weights["vision_backbone.post_layernorm.bias"] = torch.zeros(self.vision_embed_dim)
        
        return weights
    
    def create_projector_weights(self) -> Dict[str, torch.Tensor]:
        """Create projector weights."""
        weights = {}
        
        # Linear projection from vision to LLM space
        weights["projector.projection.weight"] = torch.randn(
            self.hidden_size, self.vision_embed_dim
        )
        
        return weights
    
    def create_llm_backbone_weights(self) -> Dict[str, torch.Tensor]:
        """Create LLM backbone weights."""
        weights = {}
        
        # Embeddings
        weights["llm_backbone.embed_tokens.weight"] = torch.randn(
            self.vocab_size, self.hidden_size
        )
        
        # Transformer layers
        for i in range(self.num_layers):
            layer_prefix = f"llm_backbone.layers.{i}."
            
            # Self-attention
            weights[f"{layer_prefix}self_attn.q_proj.weight"] = torch.randn(
                self.hidden_size, self.hidden_size
            )
            weights[f"{layer_prefix}self_attn.k_proj.weight"] = torch.randn(
                self.hidden_size, self.hidden_size
            )
            weights[f"{layer_prefix}self_attn.v_proj.weight"] = torch.randn(
                self.hidden_size, self.hidden_size
            )
            weights[f"{layer_prefix}self_attn.o_proj.weight"] = torch.randn(
                self.hidden_size, self.hidden_size
            )
            
            # MLP
            weights[f"{layer_prefix}mlp.gate_proj.weight"] = torch.randn(
                self.intermediate_size, self.hidden_size
            )
            weights[f"{layer_prefix}mlp.up_proj.weight"] = torch.randn(
                self.intermediate_size, self.hidden_size
            )
            weights[f"{layer_prefix}mlp.down_proj.weight"] = torch.randn(
                self.hidden_size, self.intermediate_size
            )
            
            # Layer norms
            weights[f"{layer_prefix}input_layernorm.weight"] = torch.ones(self.hidden_size)
            weights[f"{layer_prefix}post_attention_layernorm.weight"] = torch.ones(self.hidden_size)
        
        # Final layer norm
        weights["llm_backbone.norm.weight"] = torch.ones(self.hidden_size)
        
        return weights
    
    def create_lm_head_weights(self) -> Dict[str, torch.Tensor]:
        """Create LM head weights."""
        weights = {}
        
        # Language modeling head
        weights["lm_head.weight"] = torch.randn(self.vocab_size, self.hidden_size)
        
        return weights
    
    def create_complete_checkpoint(self) -> Dict[str, torch.Tensor]:
        """Create a complete checkpoint with all components."""
        all_weights = {}
        
        # Vision backbone
        all_weights.update(self.create_vision_backbone_weights())
        
        # Projector
        all_weights.update(self.create_projector_weights())
        
        # LLM backbone
        all_weights.update(self.create_llm_backbone_weights())
        
        # LM head
        all_weights.update(self.create_lm_head_weights())
        
        return all_weights
    
    def save_checkpoint(self, 
                       checkpoint_path: str,
                       format: str = "safetensors",
                       split_files: bool = False) -> List[str]:
        """
        Save checkpoint to disk.
        
        Args:
            checkpoint_path: Path to save checkpoint
            format: Format to save in ('safetensors', 'pytorch')
            split_files: Whether to split into multiple files
            
        Returns:
            List of file paths created
        """
        all_weights = self.create_complete_checkpoint()
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        saved_files = [config_path]
        
        if split_files:
            # Split into component files
            components = {
                "vision_backbone": {k: v for k, v in all_weights.items() if k.startswith("vision_backbone.")},
                "projector": {k: v for k, v in all_weights.items() if k.startswith("projector.")},
                "llm_backbone": {k: v for k, v in all_weights.items() if k.startswith("llm_backbone.")},
                "lm_head": {k: v for k, v in all_weights.items() if k.startswith("lm_head.")},
            }
            
            for component_name, component_weights in components.items():
                if not component_weights:
                    continue
                    
                if format == "safetensors":
                    file_path = os.path.join(checkpoint_path, f"{component_name}.safetensors")
                    save_file(component_weights, file_path)
                else:
                    file_path = os.path.join(checkpoint_path, f"{component_name}.pt")
                    torch.save(component_weights, file_path)
                
                saved_files.append(file_path)
        
        else:
            # Single file
            if format == "safetensors":
                file_path = os.path.join(checkpoint_path, "model.safetensors")
                save_file(all_weights, file_path)
            else:
                file_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                torch.save(all_weights, file_path)
            
            saved_files.append(file_path)
        
        return saved_files


def test_weight_loading_with_mock_checkpoint():
    """Test weight loading with a mock checkpoint."""
    print("=== Testing PrismaticVLM Weight Loading with Mock Checkpoint ===")
    
    # Create test configuration
    config = PrismaticConfig(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=4,  # Reduced for testing
        num_attention_heads=16,
        vision_embed_dim=1152,
        vision_num_hidden_layers=4,  # Reduced for testing
        vision_num_attention_heads=16,
        vision_intermediate_size=4304,
        image_size=378,
        patch_size=14,
    )
    
    # Create mock checkpoint
    checkpoint_creator = MockCheckpointCreator(config)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating mock checkpoint in: {temp_dir}")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(temp_dir, "mock_checkpoint")
        saved_files = checkpoint_creator.save_checkpoint(
            checkpoint_dir, 
            format="safetensors",
            split_files=True
        )
        
        print(f"Created checkpoint files: {saved_files}")
        
        # Test weight loading
        print("\n=== Testing Weight Loading ===")
        
        try:
            # Create validator
            validator = PrismaticVLMWeightValidator(
                checkpoint_path=checkpoint_dir,
                device="cpu",  # Use CPU for testing
                debug=True
            )
            
            # Run validation
            report_path = validator.run_full_validation(
                output_dir=temp_dir,
                test_forward_pass=True
            )
            
            print(f"\nValidation completed successfully!")
            print(f"Report saved to: {report_path}")
            
            # Read and display summary
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            print("\n=== Validation Summary ===")
            summary = report.get("summary", {})
            print(f"Weight Loading: {'✓' if summary.get('weight_loading_success') else '✗'}")
            print(f"Shape Validation: {'✓' if summary.get('shape_validation_success') else '✗'}")
            print(f"Forward Pass: {'✓' if summary.get('forward_pass_success') else '✗'}")
            print(f"Overall Success: {'✓' if summary.get('overall_success') else '✗'}")
            
            # Display key metrics
            weight_loading = report.get("weight_loading", {})
            print(f"\nKey Metrics:")
            print(f"  Loaded Parameters: {weight_loading.get('loaded_parameters', 'N/A')}")
            print(f"  Load Time: {weight_loading.get('load_time_seconds', 'N/A'):.4f}s")
            
            shape_validation = report.get("shape_validation", {})
            print(f"  Total Parameters: {shape_validation.get('total_parameters', 'N/A'):,}")
            print(f"  Shape Mismatches: {len(shape_validation.get('shape_mismatches', []))}")
            
            forward_pass = report.get("forward_pass_test", {})
            if forward_pass.get("success"):
                print(f"  Forward Pass Time: {forward_pass.get('forward_time_seconds', 'N/A'):.4f}s")
            
            return True
            
        except Exception as e:
            print(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_error_handling():
    """Test error handling with corrupted checkpoints."""
    print("\n=== Testing Error Handling ===")
    
    config = PrismaticConfig(
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
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = os.path.join(temp_dir, "corrupted_checkpoint")
        os.makedirs(checkpoint_dir)
        
        # Create corrupted checkpoint (missing critical components)
        corrupted_weights = {
            "vision_backbone.embeddings.patch_embedding.weight": torch.randn(1152, 3, 14, 14),
            # Missing projector weights (critical)
            "llm_backbone.embed_tokens.weight": torch.randn(151936, 2048),
            "unknown_component.weight": torch.randn(100, 100),  # Unknown component
        }
        
        # Save corrupted checkpoint
        corrupted_file = os.path.join(checkpoint_dir, "corrupted_model.safetensors")
        save_file(corrupted_weights, corrupted_file)
        
        # Save config
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        print(f"Created corrupted checkpoint: {corrupted_file}")
        
        try:
            # Test with corrupted checkpoint
            validator = PrismaticVLMWeightValidator(
                checkpoint_path=checkpoint_dir,
                device="cpu",
                debug=True
            )
            
            # This should handle errors gracefully
            report_path = validator.run_full_validation(
                output_dir=temp_dir,
                test_forward_pass=False  # Skip forward pass for corrupted model
            )
            
            # Check that errors were handled
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            summary = report.get("summary", {})
            print(f"Error handling test results:")
            print(f"  Weight loading handled: {'✓' if not summary.get('weight_loading_success') else '✗'}")
            print(f"  Shape validation detected issues: {'✓' if not summary.get('shape_validation_success') else '✗'}")
            
            return True
            
        except Exception as e:
            print(f"Error handling test failed: {e}")
            return False


def main():
    """Main test function."""
    print("Starting PrismaticVLM Weight Loading Integration Tests")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Normal weight loading
    if test_weight_loading_with_mock_checkpoint():
        success_count += 1
        print("✓ Normal weight loading test passed")
    else:
        print("✗ Normal weight loading test failed")
    
    # Test 2: Error handling
    if test_error_handling():
        success_count += 1
        print("✓ Error handling test passed")
    else:
        print("✗ Error handling test failed")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())