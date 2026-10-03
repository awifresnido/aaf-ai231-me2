"""
Latency & Performance Measurement Utilities
Measures inference time, model size, and tracks across trials
"""

import torch
import torch.nn as nn
import time
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class LatencyMeasurer:
    """Measure model inference latency on different devices"""
    
    def __init__(self, device: torch.device, n_warmup: int = 10, n_samples: int = 100):
        self.device = device
        self.n_warmup = n_warmup
        self.n_samples = n_samples
    
    def measure_latency(
        self,
        model: nn.Module,
        input_shape: Tuple = (1, 1, 101, 80),  # (batch, channels, time, n_mels)
    ) -> float:
        """
        Measure average inference latency in milliseconds
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (default: mel-spectrogram)
        
        Returns:
            Average latency in milliseconds
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(self.n_warmup):
                _ = model(dummy_input)
            
            # Synchronize GPU if using CUDA
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Measure
            start_time = time.perf_counter()
            
            with torch.no_grad():
                for _ in range(self.n_samples):
                    _ = model(dummy_input)
            
            # Synchronize GPU if using CUDA
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed_time = time.perf_counter() - start_time
        
        # Average latency in milliseconds
        avg_latency_ms = (elapsed_time / self.n_samples) * 1000
        
        return avg_latency_ms
    
    def measure_latency_batch(
        self,
        model: nn.Module,
        batch_sizes: list = [1, 4, 8],
        input_shape_base: Tuple = (1, 101, 80),  # (channels, time, n_mels)
    ) -> dict:
        """
        Measure latency across different batch sizes
        
        Args:
            model: PyTorch model
            batch_sizes: List of batch sizes to test
            input_shape_base: Base input shape (batch size added)
        
        Returns:
            Dictionary mapping batch_size -> latency_ms
        """
        results = {}
        model.eval()
        device = next(model.parameters()).device
        
        for batch_size in batch_sizes:
            input_shape = (batch_size,) + input_shape_base
            dummy_input = torch.randn(*input_shape, device=device)
            
            with torch.no_grad():
                # Warmup
                for _ in range(self.n_warmup):
                    _ = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Measure
                start_time = time.perf_counter()
                for _ in range(self.n_samples):
                    _ = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed_time = time.perf_counter() - start_time
            
            avg_latency_ms = (elapsed_time / self.n_samples) * 1000
            results[batch_size] = avg_latency_ms
        
        return results


class ModelProfiler:
    """Profile model size and parameters"""
    
    @staticmethod
    def get_model_size(model: nn.Module, format: str = 'kb') -> float:
        """
        Get model size in KB or MB
        
        Args:
            model: PyTorch model
            format: 'kb' or 'mb'
        
        Returns:
            Model size in specified format
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate size (assuming float32 = 4 bytes per parameter)
        size_bytes = total_params * 4
        
        if format == 'mb':
            return size_bytes / (1024 * 1024)
        else:  # kb
            return size_bytes / 1024
    
    @staticmethod
    def get_model_size_on_disk(model: nn.Module) -> float:
        """
        Get actual model size when saved to disk (state_dict)
        
        Args:
            model: PyTorch model
        
        Returns:
            Actual file size in KB
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(model.state_dict(), f.name)
            actual_size_kb = os.path.getsize(f.name) / 1024
            os.unlink(f.name)
        
        return actual_size_kb
    
    @staticmethod
    def count_parameters(model: nn.Module) -> dict:
        """
        Count trainable and non-trainable parameters
        
        Args:
            model: PyTorch model
        
        Returns:
            Dictionary with parameter counts
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = trainable + non_trainable
        
        return {
            'trainable': trainable,
            'non_trainable': non_trainable,
            'total': total,
        }
    
    @staticmethod
    def estimate_memory_usage(model: nn.Module, batch_size: int = 1, dtype: str = 'float32') -> float:
        """
        Estimate GPU memory usage for model + batch
        
        Args:
            model: PyTorch model
            batch_size: Batch size
            dtype: 'float32' (4 bytes), 'float16' (2 bytes)
        
        Returns:
            Estimated memory in MB
        """
        bytes_per_param = 4 if dtype == 'float32' else 2
        total_params = sum(p.numel() for p in model.parameters())
        
        # Model weights
        model_memory = (total_params * bytes_per_param) / (1024 * 1024)
        
        # Activation memory (rough estimate: 2x model size for batch processing)
        activation_memory = model_memory * 2 * batch_size
        
        total_memory_mb = model_memory + activation_memory
        
        return total_memory_mb


def log_performance_metrics(
    model: nn.Module,
    device: torch.device,
    latency_measurer: LatencyMeasurer,
) -> dict:
    """
    Log all performance metrics for a model
    
    Args:
        model: PyTorch model
        device: Torch device
        latency_measurer: LatencyMeasurer instance
    
    Returns:
        Dictionary of all metrics
    """
    logger.info("\n" + "="*70)
    logger.info("MODEL PERFORMANCE METRICS")
    logger.info("="*70)
    
    # Parameters
    param_counts = ModelProfiler.count_parameters(model)
    logger.info(f"\nParameters:")
    logger.info(f"  Trainable: {param_counts['trainable']:,}")
    logger.info(f"  Non-trainable: {param_counts['non_trainable']:,}")
    logger.info(f"  Total: {param_counts['total']:,}")
    
    # Model size
    size_kb_estimate = ModelProfiler.get_model_size(model, format='kb')
    size_kb_actual = ModelProfiler.get_model_size_on_disk(model)
    logger.info(f"\nModel Size:")
    logger.info(f"  Estimated (FP32): {size_kb_estimate:.2f} KB")
    logger.info(f"  Actual (on disk): {size_kb_actual:.2f} KB")
    
    # Latency
    latency_ms = latency_measurer.measure_latency(model)
    logger.info(f"\nInference Latency:")
    logger.info(f"  Batch size 1: {latency_ms:.2f} ms")
    logger.info(f"  Real-time factor: {latency_ms / 4000:.4f} (4s audio @ 16kHz)")
    
    # Memory usage estimate
    mem_mb = ModelProfiler.estimate_memory_usage(model, batch_size=1)
    logger.info(f"\nMemory Usage (estimated):")
    logger.info(f"  Batch size 1: {mem_mb:.2f} MB")
    
    logger.info(f"{'='*70}\n")
    
    return {
        'total_params': param_counts['total'],
        'trainable_params': param_counts['trainable'],
        'model_size_kb_estimate': size_kb_estimate,
        'model_size_kb_actual': size_kb_actual,
        'latency_ms': latency_ms,
        'rtf': latency_ms / 4000,  # Real-time factor
        'memory_mb_estimated': mem_mb,
    }
