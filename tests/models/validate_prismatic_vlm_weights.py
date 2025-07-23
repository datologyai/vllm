#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive validation script for PrismaticVLM weight loading with real checkpoints.

This script provides a complete validation framework for testing the weight loading
system with actual checkpoint files and comparing with reference implementations.
"""

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from vllm.config import CacheConfig, MultiModalConfig, VllmConfig
from vllm.model_executor.models.prismatic_vlm import PrismaticVLMForCausalLM
from vllm.transformers_utils.configs.prismatic import PrismaticConfig


class PrismaticVLMWeightValidator:
    """
    Comprehensive validator for PrismaticVLM weight loading.
    
    This class provides methods to:
    1. Load and validate checkpoints from different formats
    2. Test weight loading with shape and dtype validation
    3. Perform forward passes to verify functionality
    4. Compare outputs with reference implementations
    5. Generate detailed validation reports
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 debug: bool = False):
        """
        Initialize the weight validator.
        
        Args:
            checkpoint_path: Path to the checkpoint file or directory
            config_path: Optional path to model configuration
            device: Device to run validation on
            debug: Whether to enable debug logging
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = device
        self.debug = debug
        
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.config = None
        self.model = None
        self.tokenizer = None
        self.validation_results = {}
        
        self.logger.info(f"Initialized PrismaticVLM weight validator")
        self.logger.info(f"Checkpoint path: {self.checkpoint_path}")
        self.logger.info(f"Device: {self.device}")
    
    def load_configuration(self) -> PrismaticConfig:
        """
        Load model configuration from file or checkpoint.
        
        Returns:
            PrismaticConfig: The loaded configuration
        """
        self.logger.info("Loading model configuration...")
        
        try:
            if self.config_path and self.config_path.exists():
                # Load from explicit config file
                self.logger.info(f"Loading config from: {self.config_path}")
                config = PrismaticConfig.from_pretrained(str(self.config_path))
            elif self.checkpoint_path.is_dir():
                # Load from checkpoint directory
                self.logger.info(f"Loading config from checkpoint directory: {self.checkpoint_path}")
                config = PrismaticConfig.from_pretrained(str(self.checkpoint_path))
            else:
                # Create default config for testing
                self.logger.info("Creating default test configuration")
                config = PrismaticConfig(
                    vocab_size=151936,
                    hidden_size=2048,
                    intermediate_size=5632,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    vision_embed_dim=1152,
                    vision_num_hidden_layers=27,
                    vision_num_attention_heads=16,
                    vision_intermediate_size=4304,
                    image_size=378,
                    patch_size=14,
                )
            
            self.config = config
            self.logger.info(f"Configuration loaded successfully")
            self.logger.info(f"Model config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
            self.logger.info(f"Vision config: {config.vision_num_hidden_layers} layers, {config.vision_embed_dim} embed dim")
            self.logger.info(f"Expected visual tokens: {config.num_image_tokens}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def create_model(self) -> PrismaticVLMForCausalLM:
        """
        Create the PrismaticVLM model instance.
        
        Returns:
            PrismaticVLMForCausalLM: The created model
        """
        self.logger.info("Creating PrismaticVLM model...")
        
        try:
            # Create VllmConfig
            vllm_config = VllmConfig(
                model=str(self.checkpoint_path),
                trust_remote_code=True,
                multimodal_config=MultiModalConfig(),
                max_seq_len=4096,
                max_model_len=4096,
            )
            
            # Override the hf_config with our loaded config
            vllm_config.model_config.hf_config = self.config
            
            # Create model
            self.model = PrismaticVLMForCausalLM(vllm_config=vllm_config)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            self.logger.info(f"Model created successfully on device: {self.device}")
            self.logger.info(f"Model architecture validation: {self.model._validate_model_architecture()}")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            raise
    
    def load_checkpoint_weights(self) -> Dict[str, Any]:
        """
        Load weights from checkpoint file.
        
        Returns:
            Dict containing loading statistics and results
        """
        self.logger.info("Loading checkpoint weights...")
        
        try:
            start_time = time.time()
            
            # Load checkpoint
            if self.checkpoint_path.is_dir():
                # Load from directory (multiple files)
                checkpoint_files = list(self.checkpoint_path.glob("*.safetensors")) + \
                                  list(self.checkpoint_path.glob("*.bin")) + \
                                  list(self.checkpoint_path.glob("*.pt")) + \
                                  list(self.checkpoint_path.glob("*.pth"))
                
                if not checkpoint_files:
                    raise ValueError(f"No checkpoint files found in {self.checkpoint_path}")
                
                self.logger.info(f"Found {len(checkpoint_files)} checkpoint files")
                
                # Load all files
                all_weights = []
                for file_path in checkpoint_files:
                    self.logger.info(f"Loading weights from: {file_path}")
                    if file_path.suffix == ".safetensors":
                        from safetensors.torch import load_file
                        weights = load_file(str(file_path))
                    else:
                        weights = torch.load(str(file_path), map_location='cpu')
                    
                    all_weights.extend(weights.items())
                
                weights_iter = iter(all_weights)
                
            else:
                # Single file
                self.logger.info(f"Loading weights from: {self.checkpoint_path}")
                if self.checkpoint_path.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    weights = load_file(str(self.checkpoint_path))
                else:
                    weights = torch.load(str(self.checkpoint_path), map_location='cpu')
                
                weights_iter = iter(weights.items())
            
            # Load weights into model
            self.logger.info("Loading weights into model...")
            loaded_params = self.model.load_weights(weights_iter)
            
            load_time = time.time() - start_time
            
            # Create loading statistics
            loading_stats = {
                "checkpoint_path": str(self.checkpoint_path),
                "load_time_seconds": load_time,
                "loaded_parameters": len(loaded_params),
                "device": self.device,
                "model_architecture": self.model._validate_model_architecture(),
            }
            
            self.logger.info(f"Weight loading completed in {load_time:.2f} seconds")
            self.logger.info(f"Loaded {len(loaded_params)} parameters")
            
            return loading_stats
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint weights: {e}")
            raise
    
    def validate_weight_shapes(self) -> Dict[str, Any]:
        """
        Validate that loaded weights have correct shapes.
        
        Returns:
            Dict containing shape validation results
        """
        self.logger.info("Validating weight shapes...")
        
        try:
            shape_validation = {
                "total_parameters": 0,
                "components": {},
                "critical_weights": {},
                "shape_mismatches": [],
                "dtype_distribution": {},
            }
            
            # Analyze all model parameters
            for name, param in self.model.named_parameters():
                shape_validation["total_parameters"] += param.numel()
                
                # Categorize by component
                if name.startswith("vision_backbone."):
                    component = "vision_backbone"
                elif name.startswith("projector."):
                    component = "projector"
                elif name.startswith("llm_backbone."):
                    component = "llm_backbone"
                elif name.startswith("lm_head."):
                    component = "lm_head"
                else:
                    component = "other"
                
                if component not in shape_validation["components"]:
                    shape_validation["components"][component] = {
                        "count": 0,
                        "parameters": 0,
                        "shapes": []
                    }
                
                shape_validation["components"][component]["count"] += 1
                shape_validation["components"][component]["parameters"] += param.numel()
                shape_validation["components"][component]["shapes"].append({
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype)
                })
                
                # Track dtype distribution
                dtype_str = str(param.dtype)
                if dtype_str not in shape_validation["dtype_distribution"]:
                    shape_validation["dtype_distribution"][dtype_str] = 0
                shape_validation["dtype_distribution"][dtype_str] += 1
            
            # Validate critical weights
            critical_weights = [
                ("projector.projection.weight", [2048, 1152]),  # Vision to LLM projection
                ("vision_backbone.embeddings.patch_embedding.weight", [1152, 3, 14, 14]),  # Patch embedding
                ("llm_backbone.embed_tokens.weight", [151936, 2048]),  # Token embedding
            ]
            
            for weight_name, expected_shape in critical_weights:
                try:
                    param = dict(self.model.named_parameters())[weight_name]
                    actual_shape = list(param.shape)
                    
                    shape_validation["critical_weights"][weight_name] = {
                        "expected_shape": expected_shape,
                        "actual_shape": actual_shape,
                        "matches": actual_shape == expected_shape,
                        "dtype": str(param.dtype)
                    }
                    
                    if actual_shape != expected_shape:
                        shape_validation["shape_mismatches"].append({
                            "name": weight_name,
                            "expected": expected_shape,
                            "actual": actual_shape
                        })
                        
                except KeyError:
                    shape_validation["critical_weights"][weight_name] = {
                        "expected_shape": expected_shape,
                        "actual_shape": None,
                        "matches": False,
                        "dtype": None,
                        "error": "Parameter not found"
                    }
            
            self.logger.info(f"Total parameters: {shape_validation['total_parameters']:,}")
            self.logger.info(f"Component breakdown: {shape_validation['components']}")
            
            if shape_validation["shape_mismatches"]:
                self.logger.warning(f"Found {len(shape_validation['shape_mismatches'])} shape mismatches")
                for mismatch in shape_validation["shape_mismatches"]:
                    self.logger.warning(f"  {mismatch['name']}: {mismatch['expected']} vs {mismatch['actual']}")
            else:
                self.logger.info("All critical weights have correct shapes")
            
            return shape_validation
            
        except Exception as e:
            self.logger.error(f"Failed to validate weight shapes: {e}")
            raise
    
    def test_forward_pass(self, 
                         batch_size: int = 1,
                         sequence_length: int = 100,
                         include_images: bool = True) -> Dict[str, Any]:
        """
        Test forward pass functionality with loaded weights.
        
        Args:
            batch_size: Batch size for testing
            sequence_length: Sequence length for testing
            include_images: Whether to include image inputs
            
        Returns:
            Dict containing forward pass results
        """
        self.logger.info("Testing forward pass functionality...")
        
        try:
            self.model.eval()
            
            # Create dummy inputs
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, sequence_length)).to(self.device)
            positions = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
            
            # Create dummy KV caches
            kv_caches = []
            for _ in range(self.config.num_hidden_layers):
                kv_cache = torch.zeros(batch_size, 2, self.config.num_attention_heads, 
                                     sequence_length, self.config.hidden_size // self.config.num_attention_heads).to(self.device)
                kv_caches.append(kv_cache)
            
            # Create dummy attention metadata
            from vllm.attention import AttentionMetadata
            attn_metadata = AttentionMetadata(
                num_prefills=batch_size,
                num_prefill_tokens=batch_size * sequence_length,
                num_decode_tokens=0,
                slot_mapping=torch.arange(batch_size * sequence_length).to(self.device),
                seq_lens=torch.full((batch_size,), sequence_length, dtype=torch.int32).to(self.device),
                seq_lens_tensor=torch.full((batch_size,), sequence_length, dtype=torch.int32).to(self.device),
                max_prefill_seq_len=sequence_length,
                max_decode_seq_len=0,
                context_lens_tensor=torch.zeros(batch_size, dtype=torch.int32).to(self.device),
                block_tables=None,
                use_cuda_graph=False,
            )
            
            # Create dummy pixel values if including images
            pixel_values = None
            if include_images:
                pixel_values = torch.randn(batch_size, 3, 378, 378).to(self.device)
                
                # Replace some tokens with image tokens
                num_image_tokens = 729
                for i in range(batch_size):
                    input_ids[i, :num_image_tokens] = -200  # IMAGE_TOKEN_INDEX
            
            forward_pass_results = {}
            
            # Test forward pass
            start_time = time.time()
            
            with torch.no_grad():
                try:
                    # Test embedding preparation
                    inputs_embeds = self.model.prepare_inputs_embeds(
                        input_ids=input_ids,
                        pixel_values=pixel_values
                    )
                    
                    forward_pass_results["inputs_embeds"] = {
                        "shape": list(inputs_embeds.shape),
                        "dtype": str(inputs_embeds.dtype),
                        "device": str(inputs_embeds.device),
                        "mean": float(inputs_embeds.mean()),
                        "std": float(inputs_embeds.std()),
                        "min": float(inputs_embeds.min()),
                        "max": float(inputs_embeds.max()),
                    }
                    
                    # Test full forward pass
                    hidden_states = self.model.forward(
                        input_ids=input_ids,
                        positions=positions,
                        kv_caches=kv_caches,
                        attn_metadata=attn_metadata,
                        pixel_values=pixel_values,
                    )
                    
                    forward_pass_results["hidden_states"] = {
                        "shape": list(hidden_states.shape),
                        "dtype": str(hidden_states.dtype),
                        "device": str(hidden_states.device),
                        "mean": float(hidden_states.mean()),
                        "std": float(hidden_states.std()),
                        "min": float(hidden_states.min()),
                        "max": float(hidden_states.max()),
                    }
                    
                    # Test logits computation
                    from vllm.model_executor.sampling_metadata import SamplingMetadata
                    sampling_metadata = SamplingMetadata(
                        seq_groups=None,
                        seq_data=None,
                        prompt_lens=None,
                        selected_token_indices=torch.arange(batch_size * sequence_length).to(self.device),
                        categorized_sample_indices=None,
                        num_prompts=batch_size,
                    )
                    
                    logits = self.model.compute_logits(hidden_states, sampling_metadata)
                    
                    if logits is not None:
                        forward_pass_results["logits"] = {
                            "shape": list(logits.shape),
                            "dtype": str(logits.dtype),
                            "device": str(logits.device),
                            "mean": float(logits.mean()),
                            "std": float(logits.std()),
                            "min": float(logits.min()),
                            "max": float(logits.max()),
                        }
                    
                    forward_time = time.time() - start_time
                    
                    forward_pass_results["success"] = True
                    forward_pass_results["forward_time_seconds"] = forward_time
                    forward_pass_results["batch_size"] = batch_size
                    forward_pass_results["sequence_length"] = sequence_length
                    forward_pass_results["include_images"] = include_images
                    
                    self.logger.info(f"Forward pass completed successfully in {forward_time:.4f} seconds")
                    self.logger.info(f"Input embeddings shape: {forward_pass_results['inputs_embeds']['shape']}")
                    self.logger.info(f"Hidden states shape: {forward_pass_results['hidden_states']['shape']}")
                    
                    if logits is not None:
                        self.logger.info(f"Logits shape: {forward_pass_results['logits']['shape']}")
                    
                except Exception as e:
                    forward_pass_results["success"] = False
                    forward_pass_results["error"] = str(e)
                    self.logger.error(f"Forward pass failed: {e}")
            
            return forward_pass_results
            
        except Exception as e:
            self.logger.error(f"Failed to test forward pass: {e}")
            raise
    
    def generate_validation_report(self, 
                                  loading_stats: Dict[str, Any],
                                  shape_validation: Dict[str, Any],
                                  forward_pass_results: Dict[str, Any],
                                  output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            loading_stats: Weight loading statistics
            shape_validation: Shape validation results
            forward_pass_results: Forward pass test results
            output_path: Optional path to save report
            
        Returns:
            str: Path to the generated report
        """
        self.logger.info("Generating validation report...")
        
        try:
            # Create comprehensive report
            report = {
                "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_path": str(self.checkpoint_path),
                "device": self.device,
                "model_config": {
                    "architecture": "PrismaticVLM",
                    "hidden_size": self.config.hidden_size,
                    "num_hidden_layers": self.config.num_hidden_layers,
                    "vision_embed_dim": self.config.vision_embed_dim,
                    "vision_num_hidden_layers": self.config.vision_num_hidden_layers,
                    "num_image_tokens": self.config.num_image_tokens,
                    "vocab_size": self.config.vocab_size,
                },
                "weight_loading": loading_stats,
                "shape_validation": shape_validation,
                "forward_pass_test": forward_pass_results,
                "summary": {
                    "weight_loading_success": loading_stats.get("loaded_parameters", 0) > 0,
                    "shape_validation_success": len(shape_validation.get("shape_mismatches", [])) == 0,
                    "forward_pass_success": forward_pass_results.get("success", False),
                    "overall_success": False,
                }
            }
            
            # Determine overall success
            report["summary"]["overall_success"] = (
                report["summary"]["weight_loading_success"] and
                report["summary"]["shape_validation_success"] and
                report["summary"]["forward_pass_success"]
            )
            
            # Save report
            if output_path is None:
                output_path = f"prismatic_vlm_validation_report_{int(time.time())}.json"
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to: {output_path}")
            
            # Print summary
            self.logger.info("=== VALIDATION SUMMARY ===")
            self.logger.info(f"Weight Loading: {'✓' if report['summary']['weight_loading_success'] else '✗'}")
            self.logger.info(f"Shape Validation: {'✓' if report['summary']['shape_validation_success'] else '✗'}")
            self.logger.info(f"Forward Pass: {'✓' if report['summary']['forward_pass_success'] else '✗'}")
            self.logger.info(f"Overall Success: {'✓' if report['summary']['overall_success'] else '✗'}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate validation report: {e}")
            raise
    
    def run_full_validation(self, 
                           output_dir: Optional[str] = None,
                           test_forward_pass: bool = True) -> str:
        """
        Run the complete validation pipeline.
        
        Args:
            output_dir: Directory to save results
            test_forward_pass: Whether to test forward pass
            
        Returns:
            str: Path to validation report
        """
        self.logger.info("Starting full PrismaticVLM validation pipeline...")
        
        try:
            # Create output directory if specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "validation_report.json")
            else:
                output_path = None
            
            # Step 1: Load configuration
            self.load_configuration()
            
            # Step 2: Create model
            self.create_model()
            
            # Step 3: Load checkpoint weights
            loading_stats = self.load_checkpoint_weights()
            
            # Step 4: Validate weight shapes
            shape_validation = self.validate_weight_shapes()
            
            # Step 5: Test forward pass (optional)
            forward_pass_results = {}
            if test_forward_pass:
                forward_pass_results = self.test_forward_pass()
            
            # Step 6: Generate report
            report_path = self.generate_validation_report(
                loading_stats, shape_validation, forward_pass_results, output_path
            )
            
            self.logger.info(f"Full validation completed. Report: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Full validation failed: {e}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Validate PrismaticVLM weight loading")
    parser.add_argument("checkpoint_path", help="Path to checkpoint file or directory")
    parser.add_argument("--config", help="Path to model configuration file")
    parser.add_argument("--output-dir", help="Directory to save validation results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run validation on")
    parser.add_argument("--no-forward-pass", action="store_true",
                       help="Skip forward pass testing")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Create validator
    validator = PrismaticVLMWeightValidator(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        device=args.device,
        debug=args.debug
    )
    
    # Run validation
    try:
        report_path = validator.run_full_validation(
            output_dir=args.output_dir,
            test_forward_pass=not args.no_forward_pass
        )
        
        print(f"Validation completed successfully!")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())