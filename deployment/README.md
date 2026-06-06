# Messy Mashup — Music Genre Classifier (Deployment)

3-model ensemble for music genre classification from noisy audio mashups.
Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/aloktripathi/music-genre-classifier).

## Live Demo

👉 **[Try it here](https://huggingface.co/spaces/aloktripathi/music-genre-classifier)**

## Architecture

| Model | Params | Weight | Role |
|-------|--------|--------|------|
| EfficientNet-B0 | 4M | 10% | Local spectral patterns |
| AST (AudioSet pretrained) | 86M | 60% | Global temporal structure |
| ResNet-50 | 23.5M | 30% | Architectural diversity |

**Ensemble:** Weighted softmax averaging → **Live demo: 3-model ensemble · 0.9504 F1 | Full 6-model submission: 0.9614 F1**

## What the App Does

1. Accepts a WAV / MP3 / OGG / FLAC clip (up to 10 seconds)
2. Displays waveform and mel spectrogram visualizations
3. Runs all 3 models independently and shows per-model predictions
4. Shows ensemble confidence; flags low-confidence results (< 35%) with a warning
5. Displays inference latency

## Files

```
├── app.py              # Streamlit app
├── Dockerfile          # HF Spaces Docker config
├── best_cnn.pth        # EfficientNet-B0 weights
├── best_ast.pth        # AST weights
└── best_resnet50.pth   # ResNet-50 weights
```

## Run Locally

```bash
pip install streamlit==1.35.0 torch==2.3.1 torchaudio==2.3.1 timm==1.0.3 transformers==4.41.2 librosa==0.10.2 numpy==1.26.4
streamlit run app.py

# or with Docker
docker build -t genre-classifier .
docker run -p 7860:7860 genre-classifier
```

## Competition

Built for the **Messy Mashup** Kaggle competition — Jan 2026 DL & GenAI Project, IIT Madras.

Training code: [messy-mashup-classifier](https://github.com/aloktripathi1/messy-mashup-classifier)
