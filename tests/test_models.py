"""
CPU-only forward-pass tests for the three model architectures.
No checkpoint files, no real audio files, no GPU required.
Architectures are defined inline, matching src/models/ exactly.
Uses torchaudio for mel spectrogram (no librosa dependency).
"""
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import timm

SR, DUR = 22050, 10.0
N_MELS, N_FFT, HOP = 128, 2048, 512
FMIN, FMAX = 20, 8000

_mel_transform = T.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX,
)
_amp_to_db = T.AmplitudeToDB(top_db=80)


def make_mel_input():
    """Synthetic 440 Hz sine wave → (1, 1, n_mels, T) mel spectrogram tensor."""
    t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
    y = torch.from_numpy(np.sin(2 * np.pi * 440 * t).astype(np.float32))
    mel = _amp_to_db(_mel_transform(y))          # (n_mels, T)
    return mel.unsqueeze(0).unsqueeze(0)          # (1, 1, n_mels, T)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


# ── test 1: ScratchCNN ────────────────────────────────────────────────────────

class ScratchCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def test_scratch_cnn():
    x = make_mel_input()
    model = ScratchCNN(num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10), f"expected (1,10), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in ScratchCNN output"


# ── test 2: GenreClassifier (EfficientNet-B0) ─────────────────────────────────

class GenreClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inst_norm = nn.InstanceNorm2d(1)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False,
                                          in_chans=1, num_classes=0, global_pool='')
        nf = self.backbone.num_features
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(nn.LayerNorm(nf), nn.Dropout(0.5), nn.Linear(nf, num_classes))

    def forward(self, x):
        x = self.inst_norm(x)
        return self.head(self.gem(self.backbone(x)))


def test_efficientnet_cnn():
    x = make_mel_input()
    model = GenreClassifier(num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10), f"expected (1,10), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in GenreClassifier output"


# ── test 3: GenreResNet (ResNet-50) ───────────────────────────────────────────

class GenreResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False,
                                          in_chans=1, num_classes=0, global_pool='')
        nf = self.backbone.num_features
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(nn.LayerNorm(nf), nn.Dropout(0.4), nn.Linear(nf, num_classes))

    def forward(self, x):
        return self.head(self.gem(self.backbone(x)))


def test_resnet50():
    x = make_mel_input()
    mel_t = nn.InstanceNorm2d(1, affine=False)(x)
    model = GenreResNet(num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(mel_t)
    assert out.shape == (1, 10), f"expected (1,10), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in GenreResNet output"
