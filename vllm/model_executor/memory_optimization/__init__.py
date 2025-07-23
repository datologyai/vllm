# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Memory optimization utilities for vLLM.

This module provides memory optimization features specifically designed for
multimodal models with large numbers of visual tokens, such as PrismaticVLM
with 729 visual tokens per image.
"""

from .cache_manager import MultiModalKVCacheManager
from .checkpoint_utils import GradientCheckpointing
from .memory_monitor import MemoryProfiler, MemoryOptimizationConfig
from .tensor_ops import MemoryEfficientTensorOps
from .batch_optimization import MultiModalBatchOptimizer
from .precision_utils import PrecisionManager

__all__ = [
    "MultiModalKVCacheManager",
    "GradientCheckpointing", 
    "MemoryProfiler",
    "MemoryOptimizationConfig",
    "MemoryEfficientTensorOps",
    "MultiModalBatchOptimizer",
    "PrecisionManager",
]