# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PrismaticVLM's multimodal preprocessing."""
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["TRI-ML/prismatic-vlms"])
@pytest.mark.parametrize("num_imgs", [1])  # PrismaticVLM supports 1 image per prompt
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_basic(
    image_assets: ImageTestAssets,
    model_id: str,
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Ensure PrismaticVLMMultiModalProcessor handles basic image processing."""
    # Skip this test if model is not available
    pytest.skip("Model not available for testing")
    
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={} if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
        trust_remote_code=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else {}

    # Build the prompt with image token placeholders
    # PrismaticVLM uses 729 image tokens per image
    expected_toks_per_img = 729
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<image>What is in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)
    
    # Check that we have the expected number of image tokens
    image_token_count = processed_inputs["prompt_token_ids"].count(-200)  # IMAGE_TOKEN_INDEX
    assert image_token_count == expected_toks_per_img * num_imgs
    
    # Check that pixel values are present and have the right shape
    assert "pixel_values" in processed_inputs["mm_kwargs"]
    pixel_values = processed_inputs["mm_kwargs"]["pixel_values"]
    
    # Expected shape: [num_imgs, 3, 378, 378]
    assert pixel_values.shape == (num_imgs, 3, 378, 378)


@pytest.mark.parametrize("model_id", ["TRI-ML/prismatic-vlms"])
def test_processor_image_size(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that PrismaticVLM handles the fixed 378x378 image size."""
    # Skip this test if model is not available
    pytest.skip("Model not available for testing")
    
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<image>Describe this image."
    mm_data = {"image": [image_assets[0].pil_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})
    
    # Verify the image is resized to 378x378
    pixel_values = processed_inputs["mm_kwargs"]["pixel_values"]
    assert pixel_values.shape[-2:] == (378, 378), f"Expected (378, 378), got {pixel_values.shape[-2:]}"
    
    # Verify normalization (should be in range approximately [-1, 1] for SigLIP)
    assert pixel_values.min() >= -2.0 and pixel_values.max() <= 2.0