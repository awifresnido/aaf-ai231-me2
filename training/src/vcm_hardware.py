"""
Hardware Detection & Configuration Manager
Automatically detects and optimizes for Jetson Orin Nano or DGX A100
"""

import torch
import psutil
import platform
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class HardwareProfiler:
    """Detect and profile available hardware"""
    
    def __init__(self):
        self.device = self._detect_device()
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'
        self.total_vram = self._get_vram()
        self.cpu_cores = psutil.cpu_count()
        self.system_ram = psutil.virtual_memory().total / 1e9  # GB
        
        logger.info(f"Device: {self.device_name}")
        logger.info(f"GPU VRAM: {self.total_vram:.2f} GB")
        logger.info(f"System RAM: {self.system_ram:.2f} GB")
        logger.info(f"CPU Cores: {self.cpu_cores}")
    
    def _detect_device(self) -> torch.device:
        """Detect CUDA device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _get_vram(self) -> float:
        """Get GPU VRAM in GB"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        return 0.0
    
    def is_jetson(self) -> bool:
        """Check if running on Jetson (reads /etc/os-release exactly once)."""
        try:
            with open('/etc/os-release') as f:
                content = f.read().lower()
        except OSError:
            return False
        return 'tegra' in content or 'jetson' in content
    
    def is_a100(self) -> bool:
        """Check if GPU is A100"""
        return 'A100' in self.device_name
    
    def get_device_tier(self) -> str:
        """Classify device: 'low' (Jetson), 'high' (A100), 'cpu'"""
        if self.device.type == 'cpu':
            return 'cpu'
        elif self.is_a100() or self.total_vram >= 40:
            return 'high'
        elif self.total_vram >= 6:
            return 'low'
        else:
            return 'low'


class HardwareAwareConfig:
    """Generate hardware-aware training configuration"""
    
    # Hardware profiles
    PROFILES = {
        'low': {  # Jetson Orin Nano 8GB
            'batch_size': 16,
            'num_workers': 2,
            'mixed_precision': True,
            'gradient_accumulation_steps': 2,
            'max_trials': 30,
            'trials_per_gpu_hour': 2.0,
            'pin_memory': True,
            'cudnn_benchmark': False,
        },
        'high': {  # DGX A100
            'batch_size': 64,
            'num_workers': 8,
            'mixed_precision': True,
            'gradient_accumulation_steps': 1,
            'max_trials': 50,
            'trials_per_gpu_hour': 15.0,
            'pin_memory': True,
            'cudnn_benchmark': True,
        },
        'cpu': {  # CPU only
            'batch_size': 8,
            'num_workers': 1,
            'mixed_precision': False,
            'gradient_accumulation_steps': 4,
            'max_trials': 10,
            'trials_per_gpu_hour': 0.5,
            'pin_memory': False,
            'cudnn_benchmark': False,
        }
    }
    
    # Hyperparameter search spaces (different per hardware)
    SEARCH_SPACES = {
        'low': {
            'batch_size': [16, 32],
            'learning_rate': [1e-4, 1e-3],
            'weight_decay': [1e-6, 1e-5, 1e-4],
            'dropout_rate': [0.2, 0.3, 0.4],
            'n_filters_block1': [16, 32],
            'n_filters_block2': [32, 64],
            'n_filters_block3': [64, 128],
        },
        'high': {
            'batch_size': [32, 64, 128],
            'learning_rate': [5e-5, 1e-4, 5e-4, 1e-3],
            'weight_decay': [1e-6, 1e-5, 1e-4, 5e-4],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'n_filters_block1': [16, 32, 48],
            'n_filters_block2': [32, 64, 96],
            'n_filters_block3': [64, 128, 192],
        },
    }
    
    def __init__(self, profiler: HardwareProfiler):
        self.profiler = profiler
        self.tier = profiler.get_device_tier()
        self.config = self.PROFILES.get(self.tier, self.PROFILES['low'])
        self.search_space = self.SEARCH_SPACES.get(self.tier, self.SEARCH_SPACES['low'])
    
    def get_config(self) -> Dict:
        """Get hardware-aware configuration"""
        return self.config.copy()
    
    def get_search_space(self) -> Dict:
        """Get hardware-appropriate hyperparameter search space"""
        return self.search_space.copy()
    
    def get_memory_safe_batch_size(self, base_batch_size: int) -> int:
        """Reduce batch size if approaching memory limit"""
        if self.tier == 'low':
            # Jetson: monitor memory more closely
            reserved = torch.cuda.memory_reserved(0) / 1e9
            available = self.profiler.total_vram - reserved
            if available < 2.0:  # Less than 2GB available
                return max(8, base_batch_size // 2)
        return base_batch_size
    
    def get_gradient_accumulation_steps(self, batch_size: int) -> int:
        """Calculate gradient accumulation steps based on batch size"""
        base_steps = self.config['gradient_accumulation_steps']
        if batch_size < self.config['batch_size']:
            return max(1, base_steps * (self.config['batch_size'] // batch_size))
        return base_steps
    
    def get_num_workers(self, batch_size: int) -> int:
        """Reduce num_workers for small batch sizes"""
        base_workers = self.config['num_workers']
        if batch_size < 16:
            return 1
        return base_workers


def setup_hardware():
    """Initialize hardware profiling and return config"""
    profiler = HardwareProfiler()
    config = HardwareAwareConfig(profiler)
    
    # Set cudnn settings
    if config.tier != 'cpu':
        torch.backends.cudnn.benchmark = config.config['cudnn_benchmark']
        if not config.config['mixed_precision']:
            torch.set_float32_matmul_precision('medium')
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Hardware Tier: {config.tier.upper()}")
    logger.info(f"Training Config:")
    for key, val in config.config.items():
        logger.info(f"  {key}: {val}")
    logger.info(f"{'='*70}\n")
    
    return profiler, config


def get_device_string() -> str:
    """Get human-readable device string"""
    if torch.cuda.is_available():
        return f"cuda:0 ({torch.cuda.get_device_name(0)})"
    return "cpu"
