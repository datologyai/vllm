# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Memory monitoring and profiling utilities for multimodal models.

This module provides comprehensive memory tracking and optimization
configuration for models with large visual token counts like PrismaticVLM.
"""

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
from pydantic import BaseModel

from vllm.logger import init_logger
from vllm.utils import GiB_bytes

logger = init_logger(__name__)


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization strategies."""
    
    # Vision backbone optimizations
    enable_vision_checkpointing: bool = True
    """Enable gradient checkpointing for vision backbone layers."""
    
    vision_checkpoint_interval: int = 3
    """Checkpoint every N vision transformer layers."""
    
    # Projector optimizations
    enable_projector_chunking: bool = True
    """Enable chunked processing for large projector operations."""
    
    projector_chunk_size: int = 128
    """Chunk size for projector operations on 729 tokens."""
    
    # KV cache optimizations
    enable_multimodal_kv_compression: bool = True
    """Enable specialized KV cache compression for multimodal sequences."""
    
    kv_cache_dtype: str = "fp16"
    """Data type for KV cache storage (fp16, bf16, fp8)."""
    
    visual_token_cache_ratio: float = 0.8
    """Ratio of cache blocks allocated for visual tokens."""
    
    # Memory management
    enable_dynamic_memory_scaling: bool = True
    """Enable dynamic memory allocation based on sequence length."""
    
    memory_pool_size_gb: float = 2.0
    """Size of memory pool for temporary allocations (GB)."""
    
    enable_aggressive_gc: bool = True
    """Enable aggressive garbage collection for memory cleanup."""
    
    gc_interval: int = 50
    """Garbage collection interval (number of forward passes)."""
    
    # Precision optimizations  
    mixed_precision_vision: bool = True
    """Use mixed precision for vision backbone."""
    
    fp16_vision_layers: List[str] = field(default_factory=lambda: [
        "attention", "mlp", "norm"
    ])
    """Vision layers to use FP16 precision."""
    
    # Batching optimizations
    adaptive_batch_sizing: bool = True
    """Enable adaptive batch sizing based on memory usage."""
    
    max_visual_tokens_per_batch: int = 2916  # 4 images * 729 tokens
    """Maximum visual tokens per batch."""
    
    # Monitoring settings
    enable_memory_profiling: bool = False
    """Enable detailed memory profiling (performance impact)."""
    
    memory_warning_threshold: float = 0.85
    """Memory usage threshold for warnings (0.0-1.0)."""
    
    memory_critical_threshold: float = 0.95
    """Memory usage threshold for critical alerts (0.0-1.0)."""


class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    
    def __init__(
        self,
        timestamp: float,
        torch_allocated: int,
        torch_reserved: int,
        torch_max_allocated: int,
        torch_max_reserved: int,
        context: str = "",
    ):
        self.timestamp = timestamp
        self.torch_allocated = torch_allocated
        self.torch_reserved = torch_reserved
        self.torch_max_allocated = torch_max_allocated
        self.torch_max_reserved = torch_max_reserved
        self.context = context
    
    @classmethod
    def capture(cls, context: str = "") -> "MemorySnapshot":
        """Capture current memory state."""
        return cls(
            timestamp=time.time(),
            torch_allocated=torch.cuda.memory_allocated(),
            torch_reserved=torch.cuda.memory_reserved(),
            torch_max_allocated=torch.cuda.max_memory_allocated(),
            torch_max_reserved=torch.cuda.max_memory_reserved(),
            context=context,
        )
    
    def memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        return self.torch_allocated / GiB_bytes
    
    def memory_reserved_gb(self) -> float:
        """Get reserved memory in GB."""
        return self.torch_reserved / GiB_bytes
    
    def memory_efficiency(self) -> float:
        """Calculate memory efficiency (allocated/reserved)."""
        if self.torch_reserved == 0:
            return 0.0
        return self.torch_allocated / self.torch_reserved
    
    def __str__(self) -> str:
        return (
            f"MemorySnapshot({self.context}): "
            f"Allocated={self.memory_usage_gb():.2f}GB, "
            f"Reserved={self.memory_reserved_gb():.2f}GB, "
            f"Efficiency={self.memory_efficiency():.2%}"
        )


class MemoryProfiler:
    """
    Comprehensive memory profiler for multimodal models.
    
    Tracks memory usage patterns, identifies bottlenecks, and provides
    optimization recommendations for models with large visual token counts.
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.snapshots: List[MemorySnapshot] = []
        self.peak_usage = 0
        self.baseline_usage = 0
        self.forward_count = 0
        self.gc_count = 0
        
        # Memory tracking
        self._tracking_enabled = False
        self._component_usage: Dict[str, List[Tuple[float, int]]] = {}
        
        # Set memory warning thresholds
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            self.warning_threshold = int(total_memory * config.memory_warning_threshold)
            self.critical_threshold = int(total_memory * config.memory_critical_threshold)
        else:
            self.warning_threshold = 0
            self.critical_threshold = 0
    
    def start_tracking(self) -> None:
        """Start memory tracking."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, memory tracking disabled")
            return
            
        self._tracking_enabled = True
        self.baseline_usage = torch.cuda.memory_allocated()
        self.snapshots.clear()
        
        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats()
        
        logger.info(f"Memory tracking started. Baseline: {self.baseline_usage / GiB_bytes:.2f}GB")
    
    def stop_tracking(self) -> Dict[str, Any]:
        """Stop tracking and return summary statistics."""
        if not self._tracking_enabled:
            return {}
        
        self._tracking_enabled = False
        
        # Capture final snapshot
        final_snapshot = MemorySnapshot.capture("final")
        self.snapshots.append(final_snapshot)
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        logger.info("Memory tracking stopped")
        logger.info(f"Peak usage: {stats['peak_usage_gb']:.2f}GB")
        logger.info(f"Memory efficiency: {stats['average_efficiency']:.2%}")
        
        return stats
    
    @contextmanager
    def profile_component(self, component_name: str):
        """Context manager for profiling individual components."""
        if not self._tracking_enabled:
            yield
            return
        
        # Capture before
        start_memory = torch.cuda.memory_allocated()
        start_time = time.time()
        
        try:
            yield
        finally:
            # Capture after
            end_memory = torch.cuda.memory_allocated()
            end_time = time.time()
            
            # Track usage
            memory_delta = end_memory - start_memory
            time_delta = end_time - start_time
            
            if component_name not in self._component_usage:
                self._component_usage[component_name] = []
            
            self._component_usage[component_name].append((time_delta, memory_delta))
            
            # Log if significant memory usage
            if abs(memory_delta) > 100 * 1024 * 1024:  # 100MB threshold
                logger.debug(
                    f"{component_name}: {memory_delta / (1024**2):.1f}MB, "
                    f"{time_delta*1000:.1f}ms"
                )
    
    def capture_snapshot(self, context: str = "") -> MemorySnapshot:
        """Capture and store a memory snapshot."""
        snapshot = MemorySnapshot.capture(context)
        
        if self._tracking_enabled:
            self.snapshots.append(snapshot)
            
            # Update peak usage
            if snapshot.torch_allocated > self.peak_usage:
                self.peak_usage = snapshot.torch_allocated
            
            # Check thresholds
            self._check_memory_thresholds(snapshot)
        
        return snapshot
    
    def track_forward_pass(self, batch_size: int, visual_tokens: int) -> None:
        """Track memory usage for a forward pass."""
        if not self._tracking_enabled:
            return
        
        self.forward_count += 1
        
        # Capture snapshot
        context = f"forward_{self.forward_count}_bs{batch_size}_vt{visual_tokens}"
        self.capture_snapshot(context)
        
        # Trigger GC if needed
        if (self.config.enable_aggressive_gc and 
            self.forward_count % self.config.gc_interval == 0):
            self._trigger_garbage_collection()
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations based on profiling data."""
        recommendations = []
        
        if not self.snapshots:
            return ["No profiling data available"]
        
        # Analyze memory patterns
        if len(self.snapshots) > 1:
            # Check for memory leaks
            first_usage = self.snapshots[0].torch_allocated
            last_usage = self.snapshots[-1].torch_allocated
            growth_rate = (last_usage - first_usage) / len(self.snapshots)
            
            if growth_rate > 10 * 1024 * 1024:  # 10MB per step
                recommendations.append(
                    f"Potential memory leak detected: {growth_rate / (1024**2):.1f}MB/step growth"
                )
        
        # Check memory efficiency
        avg_efficiency = sum(s.memory_efficiency() for s in self.snapshots) / len(self.snapshots)
        if avg_efficiency < 0.7:
            recommendations.append(
                f"Low memory efficiency ({avg_efficiency:.2%}). Consider enabling memory pooling."
            )
        
        # Check peak vs average usage
        avg_usage = sum(s.torch_allocated for s in self.snapshots) / len(self.snapshots)
        peak_ratio = self.peak_usage / avg_usage if avg_usage > 0 else 1.0
        
        if peak_ratio > 1.5:
            recommendations.append(
                f"High memory spikes detected ({peak_ratio:.1f}x average). "
                "Consider gradient checkpointing or batch size reduction."
            )
        
        # Component-specific recommendations
        for component, usages in self._component_usage.items():
            if usages:
                avg_memory = sum(memory for _, memory in usages) / len(usages)
                if avg_memory > 500 * 1024 * 1024:  # 500MB
                    recommendations.append(
                        f"{component} uses {avg_memory / (1024**2):.0f}MB on average. "
                        "Consider optimization."
                    )
        
        if not recommendations:
            recommendations.append("Memory usage appears optimal.")
        
        return recommendations
    
    def _check_memory_thresholds(self, snapshot: MemorySnapshot) -> None:
        """Check if memory usage exceeds thresholds."""
        if snapshot.torch_allocated > self.critical_threshold:
            logger.error(
                f"CRITICAL: Memory usage {snapshot.memory_usage_gb():.2f}GB "
                f"exceeds critical threshold ({self.critical_threshold / GiB_bytes:.2f}GB)"
            )
        elif snapshot.torch_allocated > self.warning_threshold:
            logger.warning(
                f"Memory usage {snapshot.memory_usage_gb():.2f}GB "
                f"exceeds warning threshold ({self.warning_threshold / GiB_bytes:.2f}GB)"
            )
    
    def _trigger_garbage_collection(self) -> None:
        """Trigger garbage collection and log results."""
        before_memory = torch.cuda.memory_allocated()
        
        # Python GC
        collected = gc.collect()
        
        # PyTorch cache cleanup
        torch.cuda.empty_cache()
        
        after_memory = torch.cuda.memory_allocated()
        freed_memory = before_memory - after_memory
        
        self.gc_count += 1
        
        if freed_memory > 10 * 1024 * 1024:  # Log if freed > 10MB
            logger.debug(
                f"GC #{self.gc_count}: Freed {freed_memory / (1024**2):.1f}MB, "
                f"collected {collected} objects"
            )
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive memory usage statistics."""
        if not self.snapshots:
            return {}
        
        allocations = [s.torch_allocated for s in self.snapshots]
        efficiencies = [s.memory_efficiency() for s in self.snapshots]
        
        stats = {
            "total_snapshots": len(self.snapshots),
            "peak_usage_gb": self.peak_usage / GiB_bytes,
            "average_usage_gb": sum(allocations) / len(allocations) / GiB_bytes,
            "min_usage_gb": min(allocations) / GiB_bytes,
            "max_usage_gb": max(allocations) / GiB_bytes,
            "average_efficiency": sum(efficiencies) / len(efficiencies),
            "forward_passes": self.forward_count,
            "gc_triggered": self.gc_count,
            "component_usage": dict(self._component_usage),
        }
        
        # Calculate memory growth rate
        if len(self.snapshots) > 1:
            first_usage = self.snapshots[0].torch_allocated
            last_usage = self.snapshots[-1].torch_allocated
            time_span = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
            
            if time_span > 0:
                stats["memory_growth_rate_mb_per_sec"] = (
                    (last_usage - first_usage) / time_span / (1024**2)
                )
        
        return stats
    
    def export_profile_data(self, filepath: str) -> None:
        """Export profiling data to file."""
        import json
        
        data = {
            "config": {
                "enable_memory_profiling": self.config.enable_memory_profiling,
                "gc_interval": self.config.gc_interval,
                "memory_warning_threshold": self.config.memory_warning_threshold,
            },
            "statistics": self._calculate_statistics(),
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "allocated_gb": s.memory_usage_gb(),
                    "reserved_gb": s.memory_reserved_gb(),
                    "efficiency": s.memory_efficiency(),
                    "context": s.context,
                }
                for s in self.snapshots
            ],
            "recommendations": self.get_memory_recommendations(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Memory profile data exported to {filepath}")


def create_memory_profiler(
    enable_profiling: bool = True,
    **config_kwargs
) -> MemoryProfiler:
    """
    Create a memory profiler with default configuration.
    
    Args:
        enable_profiling: Whether to enable detailed profiling
        **config_kwargs: Additional configuration options
        
    Returns:
        MemoryProfiler instance
    """
    config = MemoryOptimizationConfig(
        enable_memory_profiling=enable_profiling,
        **config_kwargs
    )
    
    return MemoryProfiler(config)


@contextmanager
def profile_memory(context: str = "", export_path: Optional[str] = None):
    """
    Context manager for simple memory profiling.
    
    Args:
        context: Description of the profiled operation
        export_path: Optional path to export profile data
        
    Usage:
        with profile_memory("vision_backbone"):
            output = model.vision_backbone(images)
    """
    profiler = create_memory_profiler()
    profiler.start_tracking()
    
    try:
        with profiler.profile_component(context):
            yield profiler
    finally:
        stats = profiler.stop_tracking()
        
        if export_path:
            profiler.export_profile_data(export_path)
        
        # Log summary
        if stats:
            logger.info(f"Memory profiling complete for '{context}':")
            logger.info(f"  Peak usage: {stats['peak_usage_gb']:.2f}GB")
            logger.info(f"  Average efficiency: {stats['average_efficiency']:.2%}")
            
            recommendations = profiler.get_memory_recommendations()
            for rec in recommendations[:3]:  # Show top 3 recommendations
                logger.info(f"  Recommendation: {rec}")