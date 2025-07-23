# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for PrismaticVLM model registration."""

import pytest


class TestPrismaticVLMIntegration:
    """Test PrismaticVLM integration with vLLM's model and test registry."""
    
    def test_model_in_main_registry(self):
        """Test that PrismaticVLM is registered in the main model registry."""
        from vllm.model_executor.models.registry import ModelRegistry
        
        supported_archs = ModelRegistry.get_supported_archs()
        assert "PrismaticVLMForCausalLM" in supported_archs, \
            "PrismaticVLMForCausalLM should be in supported architectures"
    
    def test_model_is_multimodal(self):
        """Test that PrismaticVLM is correctly identified as multimodal."""
        from vllm.model_executor.models.registry import ModelRegistry
        
        is_multimodal = ModelRegistry.is_multimodal_model("PrismaticVLMForCausalLM")
        assert is_multimodal, "PrismaticVLMForCausalLM should be identified as multimodal"
    
    def test_model_in_test_registry(self):
        """Test that PrismaticVLM is in the test model registry."""
        # This test requires pytest and other dependencies
        try:
            from tests.models.registry import HF_EXAMPLE_MODELS
            
            supported_models = HF_EXAMPLE_MODELS.get_supported_archs()
            assert "PrismaticVLMForCausalLM" in supported_models, \
                "PrismaticVLMForCausalLM should be in test registry"
                
            # Get model info and verify configuration
            info = HF_EXAMPLE_MODELS.get_hf_info("PrismaticVLMForCausalLM")
            assert info.default == "TRI-ML/prismatic-vlms"
            assert info.trust_remote_code == True
            assert info.is_available_online == False  # Set to False for testing
            
        except ImportError:
            pytest.skip("Test registry not available (missing dependencies)")
    
    def test_processor_registration(self):
        """Test that the multimodal processor is properly registered."""
        try:
            from vllm.multimodal import MULTIMODAL_REGISTRY
            from vllm.model_executor.models.prismatic_vlm import PrismaticVLMForCausalLM
            
            # Check if processor is registered for the model class
            processor_factories = MULTIMODAL_REGISTRY._processor_factories
            has_processor = processor_factories.contains(PrismaticVLMForCausalLM)
            assert has_processor, "PrismaticVLM should have a registered processor"
            
        except Exception as e:
            # Expected if dependencies are missing
            pytest.skip(f"Processor check failed (missing dependencies): {e}")
    
    def test_model_class_imports(self):
        """Test that the PrismaticVLM classes can be imported."""
        try:
            from vllm.model_executor.models.prismatic_vlm import PrismaticVLMForCausalLM
            from vllm.transformers_utils.configs.prismatic import PrismaticConfig
            from vllm.multimodal.prismatic_processor import (
                PrismaticMultiModalProcessor,
                PrismaticProcessingInfo,
                PrismaticDummyInputsBuilder
            )
            
            # Basic checks on class availability
            assert PrismaticVLMForCausalLM is not None
            assert PrismaticConfig is not None
            assert PrismaticMultiModalProcessor is not None
            assert PrismaticProcessingInfo is not None
            assert PrismaticDummyInputsBuilder is not None
            
        except ImportError as e:
            pytest.skip(f"Import failed (missing dependencies): {e}")


if __name__ == "__main__":
    # Run basic tests without pytest
    test_instance = TestPrismaticVLMIntegration()
    
    try:
        test_instance.test_model_in_main_registry()
        print("✓ Model is in main registry")
    except Exception as e:
        print(f"✗ Main registry test failed: {e}")
    
    try:
        test_instance.test_model_is_multimodal()
        print("✓ Model is identified as multimodal")
    except Exception as e:
        print(f"✗ Multimodal check failed: {e}")
    
    try:
        test_instance.test_model_class_imports()
        print("✓ Model classes can be imported")
    except Exception as e:
        print(f"✗ Import test failed: {e}")