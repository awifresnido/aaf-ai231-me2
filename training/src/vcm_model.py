"""
VCM Model: Lightweight Separable CNN for Intent Classification
Architecture: 14K parameters, 14KB quantized, <100ms latency on RPi4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SeparableConvBlock(nn.Module):
    """Depthwise-separable convolution block (5-10× fewer params than standard conv)"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Depthwise: separate kernel per input channel
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.relu_dw = nn.ReLU(inplace=True)
        
        # Pointwise: 1×1 to mix channels
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu_pw = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu_dw(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu_pw(x)
        x = self.dropout(x)
        return x


class VCMModel(nn.Module):
    """Lightweight separable CNN for Voice Command Intent Classification (13 classes)"""
    
    def __init__(
        self,
        n_intents: int,
        n_mel_bins: int = 80,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        
        self.n_intents = n_intents
        
        # 1→32, no stride
        self.sep_conv_1 = SeparableConvBlock(1, 32, dropout=dropout_rate)
        # 32→64, stride 2
        self.sep_conv_2 = SeparableConvBlock(32, 64, stride=(2, 2), dropout=dropout_rate)
        # 64→128, stride 2
        self.sep_conv_3 = SeparableConvBlock(64, 128, stride=(2, 2), dropout=dropout_rate)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_final = nn.Dropout(dropout_rate)
        self.intent_head = nn.Linear(128, n_intents)
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: (batch, 1, time, 80). Output: (batch, n_intents)"""
        x = self.sep_conv_1(mel_spectrogram)  # (batch, 32, time, 80)
        x = self.sep_conv_2(x)                 # (batch, 64, time/2, 40)
        x = self.sep_conv_3(x)                 # (batch, 128, time/4, 20)
        x = self.global_avg_pool(x)            # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)              # (batch, 128)
        x = self.dropout_final(x)
        intent_logits = self.intent_head(x)    # (batch, n_intents)
        return intent_logits
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Total trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_vcm_model(
    n_intents: int,
    n_mel_bins: int = 80,
    dropout_rate: float = 0.3,
    device: str = "cpu",
) -> VCMModel:
    """Create and initialize VCM model"""
    model = VCMModel(
        n_mel_bins=n_mel_bins,
        n_intents=n_intents,
        dropout_rate=dropout_rate,
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    print("Creating VCM model...")
    import json
    from pathlib import Path
    _ontology = json.loads(Path("configs/ontology.json").read_text())
    model = create_vcm_model(n_mel_bins=80, n_intents=len(_ontology["training_labels"]), device="cpu")
    print(f"Total parameters: {VCMModel.count_parameters(model):,}")
    
    # Test with dummy input
    batch_size, time_steps, mel_bins = 4, 100, 80
    dummy_input = torch.randn(batch_size, 1, time_steps, mel_bins)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape} → Output: {output.shape}")
