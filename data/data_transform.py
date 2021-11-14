import torch.nn as nn
import torch
import torchaudio
import torch.nn.functional as F

class ToMono(nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.mean(waveform, dim=0, keepdim=True)

class Normalize(nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return (waveform-waveform.mean())/waveform.std()

class Pad(nn.Module):
    def __init__(self, value:float, size:int):
        super().__init__()
        self.value = value
        self.size = size

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return F.pad(waveform, (0, self.size-max(waveform.shape)), "constant", self.value)

audio_transform = nn.Sequential(*[
    ToMono(),
    torchaudio.transforms.Resample(orig_freq=441000, new_freq=8000),
    Normalize(),
    Pad(value=0, size=32000)
])

