# SPDX-License-Identifier: Apache-2.0  
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Dict, Mapping, Sequence

import torch
from PIL import Image
from transformers import BatchFeature
import torchvision.transforms as transforms

from vllm.logger import init_logger
from vllm.multimodal.processing import (BaseProcessingInfo, BaseMultiModalProcessor, 
                                        PromptUpdate, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import MultiModalDataItems

logger = init_logger(__name__)


class PrismaticProcessingInfo(BaseProcessingInfo):
    
    def get_supported_mm_limits(self) -> Dict[str, int]:
        return {"image": 1}
    
    def get_mm_max_tokens_per_item(self, seq_len: int) -> Dict[str, int]:
        return {"image": 729}
    
    def get_allowed_mm_limits(self) -> Dict[str, int]:
        return {"image": 1}
    
    def get_tokenizer(self):
        return self.ctx.tokenizer


class PrismaticDummyInputsBuilder(BaseDummyInputsBuilder[PrismaticProcessingInfo]):
    
    def get_dummy_mm_data(self, mm_counts: Dict[str, int]) -> Dict[str, Any]:
        num_images = mm_counts.get("image", 1)
        dummy_image = Image.new("RGB", (378, 378), color="black")
        return {
            "image": [dummy_image] * num_images
        }
    
    def get_dummy_text(self, mm_counts: Dict[str, int]) -> str:
        num_images = mm_counts.get("image", 1)
        image_tokens = "<image>" * num_images
        dummy_text = image_tokens + " dummy text"
        return dummy_text
    
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Dict[str, int],
    ) -> Any:
        class DummyInputs:
            def __init__(self, text: str, mm_data: Dict[str, Any]):
                self.prompt = text
                self.text = text
                self.multi_modal_data = mm_data
                self.mm_data = mm_data
                self.hf_processor_mm_kwargs = {}
                self.prompt_token_ids = []
                self.tokenization_kwargs = {}
        
        return DummyInputs(
            self.get_dummy_text(mm_counts),
            self.get_dummy_mm_data(mm_counts)
        )


class PrismaticMultiModalProcessor(BaseMultiModalProcessor[PrismaticProcessingInfo]):
    
    def __init__(self, info, dummy_inputs_builder, cache=None, **kwargs):
        super().__init__(info, dummy_inputs_builder, cache=cache)
        
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        pixel_values = hf_inputs.get("pixel_values")
        if pixel_values is not None and pixel_values.numel() > 0:
            return dict(
                pixel_values=MultiModalFieldConfig.batched("image"),
            )
        else:
            return dict()
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        images = mm_data.get("images", mm_data.get("image", []))
        if not isinstance(images, list):
            images = [images] if images is not None else []
        
        max_images = self.info.get_supported_mm_limits().get("image", 1)
        if len(images) > max_images:
            raise ValueError(f"Too many images: {len(images)} > {max_images}")
        
        processed_images = []
        for image in images:
            if hasattr(image, 'resize'):
                resized_image = image.resize((378, 378))
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                processed_image = transform(resized_image)
                processed_images.append(processed_image)
            else:
                processed_images.append(image)
        
        if processed_images:
            pixel_values = torch.stack(processed_images)
        else:
            pixel_values = torch.empty((0, 3, 378, 378))
        
        tokenizer = self.info.ctx.tokenizer
        
        filtered_tok_kwargs = {k: v for k, v in tok_kwargs.items() 
                              if k not in {'padding', 'truncation', 'return_tensors'}}
        
        tokenized = tokenizer(
            prompt,
            padding=False,
            truncation=False,
            return_tensors="pt",
            **filtered_tok_kwargs
        )
        
        input_ids = tokenized["input_ids"][0]
        
        return BatchFeature({
            "input_ids": input_ids.unsqueeze(0),
            "pixel_values": pixel_values,
        })
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        
        image_token_id = None
        
        try:
            hf_config = self.info.get_hf_config()
            if hasattr(hf_config, 'image_token_index'):
                image_token_id = hf_config.image_token_index
        except:
            pass
        
        if image_token_id is None:
            for token_id, token_obj in tokenizer.added_tokens_decoder.items():
                token_str = str(token_obj)
                if token_str == '<|image_pad|>':
                    image_token_id = token_id
                    break
        
        if image_token_id is None:
            image_token_id = 151655
        
        NUM_IMAGE_TOKENS = 729
        
        def get_replacement(item_idx: int):
            return [image_token_id] * NUM_IMAGE_TOKENS
        
        tokenizer = self.info.get_tokenizer()
        image_placeholder_tokens = tokenizer.encode("<image>", add_special_tokens=False)
        
        return [
            PromptReplacement(
                modality="image",
                target=image_placeholder_tokens,
                replacement=get_replacement,
            ),
        ]
    
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False
    
