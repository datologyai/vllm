# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.prismatic import PrismaticConfig

from .interfaces import SupportsMultiModal
from .utils import (AutoWeightsLoader, maybe_prefix, merge_multimodal_embeddings,
                    init_vllm_registered_model)

import sys
sys.path.append('/home/ubuntu/datopenqwen/prismatic-vlms')

from prismatic.util.nn_utils import MLPProjector, LinearProjector
from prismatic.models.materialize import get_vision_backbone_and_transform


class PrismaticVLMForCausalLM(nn.Module, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, PrismaticConfig):
            raise ValueError(f"Expected PrismaticConfig, got {type(config)}")
        
        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        
        from transformers import PretrainedConfig
        text_config_dict = config.text_config.copy()
        if 'architectures' not in text_config_dict:
            text_config_dict['architectures'] = ['Qwen3ForCausalLM']
        
        text_config_obj = PretrainedConfig(**text_config_dict)
        self.vision_backbone = self._create_prismatic_vision_backbone()
        self.projector = self._create_prismatic_projector()
        
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_config_obj,
            prefix=maybe_prefix(prefix, "language_model")
        )
        
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)
        self._setup_image_token_handling()
    
    def _create_prismatic_vision_backbone(self):
        vision_backbone_id = self.config.vision_backbone_id
        vision_backbone, _ = get_vision_backbone_and_transform(
            vision_backbone_id=vision_backbone_id,
            image_resize_strategy="resize-naive"
        )
        device = self.config.device_config.device if hasattr(self.config, "device_config") else "cuda"
        return vision_backbone.to(device=device, dtype=torch.bfloat16)
    
    def _create_prismatic_projector(self):
        vision_hidden_size = self.config.vision_embed_dim
        llm_hidden_size = self.config.hidden_size
        arch_specifier = getattr(self.config, 'arch_specifier', '729-avgpool')
        
        if arch_specifier == "linear":
            projector = LinearProjector(vision_hidden_size, llm_hidden_size)
        elif arch_specifier.endswith("gelu-mlp"):
            projector = MLPProjector(vision_hidden_size, llm_hidden_size)
        elif arch_specifier.endswith("avgpool"):
            query_dim = int(arch_specifier.split("-")[0]) if arch_specifier.split("-")[0].isdigit() else 729
            projector = AvgPoolProjector(
                query_num=query_dim,
                mm_hidden_size=vision_hidden_size,
                llm_hidden_size=llm_hidden_size
            )
        else:
            raise ValueError(f"Unsupported architecture specifier: {arch_specifier}")
        
        return projector.to(dtype=torch.bfloat16)
    
    def _setup_image_token_handling(self):
        if not hasattr(self.config, 'image_token_index'):
            self.config.image_token_index = 151655
    
    def _parse_and_validate_image_input(self, **kwargs: object) -> Optional[torch.Tensor]:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError(f"Expected pixel_values to be a torch.Tensor, got {type(pixel_values)}")
        return pixel_values
    
    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[torch.Tensor]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        
        image_input = image_input.to(dtype=torch.bfloat16)
        vision_features = self.vision_backbone(image_input)
        return self.projector(vision_features)
    
    def get_input_embeddings(self, input_ids: torch.Tensor, 
                           multimodal_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_index,
            )
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.sample(logits, sampling_metadata)
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        weights_dict = dict(weights)
        
        if 'model' in weights_dict and len(weights_dict) == 1:
            model_state_dict = weights_dict['model']
        else: 
            model_state_dict = weights_dict
            
        if isinstance(model_state_dict, dict) and 'projector' in model_state_dict and 'llm_backbone' in model_state_dict:
            loaded_weights = set()
            
            if 'vision_backbone' in model_state_dict:
                vision_state_dict = model_state_dict['vision_backbone']
                
                self.vision_backbone.load_state_dict(vision_state_dict, strict=False)
                
                device = next(self.language_model.parameters()).device
                self.vision_backbone = self.vision_backbone.to(device=device, dtype=torch.bfloat16)
                loaded_weights.update(vision_state_dict.keys())
            
            if 'projector' in model_state_dict:
                projector_state_dict = model_state_dict['projector']
                
                clean_projector_state_dict = {}
                for name, param in projector_state_dict.items():
                    if name.startswith("mlp_projector."):
                        clean_name = name.replace("mlp_projector.", "")
                        clean_projector_state_dict[clean_name] = param
                    else:
                        clean_projector_state_dict[name] = param
                
                self.projector.load_state_dict(clean_projector_state_dict, strict=False)
                
                device = next(self.language_model.parameters()).device
                self.projector = self.projector.to(device=device, dtype=torch.bfloat16)
                loaded_weights.update(clean_projector_state_dict.keys())
            
            if 'llm_backbone' in model_state_dict:
                llm_state_dict = model_state_dict['llm_backbone']
                
                compatible_llm_weights = {}
                for name, param in llm_state_dict.items():
                    if name.startswith("llm."):
                        mapped_name = name.replace("llm.", "")
                        compatible_llm_weights[mapped_name] = param
                
                if compatible_llm_weights:
                    llm_weights_list = [(name, param) for name, param in compatible_llm_weights.items()]
                    llm_loader = AutoWeightsLoader(self.language_model)
                    loaded_weights.update(llm_loader.load_weights(llm_weights_list))
            
            return loaded_weights
    
      
    def _map_vision_weight_name(self, name: str) -> str:
        name = name.replace("vision_backbone.featurizer.", "")
        
        if name.startswith("trunk."):
            name = name.replace("trunk.", "")
            
            if name.startswith("patch_embed.proj."):
                return name.replace("patch_embed.proj.", "patch_embedding.")
            
            elif name == "pos_embed":
                return "position_embeddings"
            
            elif name.startswith("blocks."):
                name = name.replace("blocks.", "vision_layers.")
                
                name = name.replace(".norm1.", ".attention_norm.")
                name = name.replace(".norm2.", ".mlp_norm.")
                
                name = name.replace(".attn.", ".attention.")
                
                if ".attention.qkv." in name:
                    name = name.replace(".attention.qkv.", ".attention.in_proj.")
                
                if ".attention.proj." in name:
                    name = name.replace(".attention.proj.", ".attention.out_proj.")
                
                name = name.replace(".mlp.fc1.", ".mlp.0.")
                name = name.replace(".mlp.fc2.", ".mlp.2.")
                
                return name
            
            elif name.startswith("norm."):
                return name.replace("norm.", "layer_norm.")
            
            elif name.startswith("attn_pool."):
                return None
        
        return None


class AvgPoolProjector(nn.Module):
    def __init__(self, query_num: int, mm_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.query_num = query_num
        self.mm_hidden_size = mm_hidden_size
        self.llm_hidden_size = llm_hidden_size
        
        self.add_module("0", nn.Linear(mm_hidden_size, llm_hidden_size))
        self.add_module("1", nn.GELU())
        self.add_module("2", nn.Linear(llm_hidden_size, llm_hidden_size))
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        _, num_patches, _ = vision_features.shape
        
        if num_patches != self.query_num:
            pooled_features = F.adaptive_avg_pool1d(
                vision_features.transpose(1, 2), self.query_num
            ).transpose(1, 2)
        else:
            pooled_features = vision_features
        
        x = getattr(self, "0")(pooled_features)
        x = getattr(self, "1")(x)
        projected_features = getattr(self, "2")(x)
        
        return projected_features


from vllm.multimodal.prismatic_processor import (
    PrismaticMultiModalProcessor,
    PrismaticProcessingInfo,
    PrismaticDummyInputsBuilder
)

MULTIMODAL_REGISTRY.register_processor(
    PrismaticMultiModalProcessor,
    info=PrismaticProcessingInfo,
    dummy_inputs=PrismaticDummyInputsBuilder,
)(PrismaticVLMForCausalLM)