import logging, time
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, librosa, librosa.display, timm, os, tempfile
import matplotlib.pyplot as plt
from io import BytesIO
import torchaudio.transforms as T
from transformers import ASTFeatureExtractor, ASTForAudioClassification

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")

st.markdown("""<style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    .stDeployButton {display:none;}
    .block-container {padding-top:2rem; max-width:720px;}
</style>""", unsafe_allow_html=True)

DEVICE = torch.device('cpu')
GENRES = sorted(['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
GENRE_EMOJI = {'blues':'🎸','classical':'🎻','country':'🤠','disco':'🪩','hiphop':'🎤',
               'jazz':'🎷','metal':'🤘','pop':'🎵','reggae':'🌴','rock':'🎸'}
W_CNN, W_AST, W_RES = 0.10, 0.60, 0.30
SR_CNN, SR_AST, SR_RES = 22050, 16000, 22050
N_MELS, N_FFT, HOP = 128, 2048, 512
FMIN, FMAX, DUR = 20, 8000, 10.0

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2,-1)).pow(1.0/self.p)

class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(sample_rate=SR_CNN, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=FMIN, f_max=FMAX)
        self.amp_to_db = T.AmplitudeToDB(top_db=80)
        self.inst_norm = nn.InstanceNorm2d(1)
        self.spec_aug_freq = T.FrequencyMasking(27)
        self.spec_aug_time = T.TimeMasking(80)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, in_chans=1, num_classes=0, global_pool='')
        nf = self.backbone.num_features
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(nn.LayerNorm(nf), nn.Dropout(0.5), nn.Linear(nf, 10))
    def forward(self, x):
        with torch.no_grad():
            s = self.mel_spec(x)
            s = self.amp_to_db(s).unsqueeze(1)
            s = self.inst_norm(s)
        return self.head(self.gem(self.backbone(s)))

class ResnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False, in_chans=1, num_classes=0, global_pool='')
        nf = self.backbone.num_features
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(nn.LayerNorm(nf), nn.Dropout(0.4), nn.Linear(nf, 10))
    def forward(self, x):
        return self.head(self.gem(self.backbone(x)))

class AstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=10, ignore_mismatched_sizes=True)
    def forward(self, x):
        return self.ast(input_values=x).logits

@st.cache_resource
def load_models():
    cnn = CnnModel()
    cnn.load_state_dict(torch.load("best_cnn.pth", map_location=DEVICE, weights_only=True), strict=True)
    cnn.eval()
    res = ResnetModel()
    res.load_state_dict(torch.load("best_resnet50.pth", map_location=DEVICE, weights_only=True), strict=True)
    res.eval()
    ast_m = AstModel()
    ast_m.load_state_dict(torch.load("best_ast.pth", map_location=DEVICE, weights_only=True), strict=True)
    ast_m.eval()
    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    return cnn, res, ast_m, fe

cnn_model, resnet_model, ast_model, ast_fe = load_models()

def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=DUR)
    t = int(sr * DUR)
    if len(y) < t: y = np.pad(y, (0, t - len(y)))
    elif len(y) > t: y = y[:t]
    return y

def get_probs(path):
    # Load once at 22050; resample to 16000 for AST
    y_base = load_audio(path, SR_CNN)
    y_ast = librosa.resample(y_base, orig_sr=SR_CNN, target_sr=SR_AST)

    # CNN / EfficientNet
    with torch.no_grad():
        logits_cnn = cnn_model(torch.from_numpy(y_base).float().unsqueeze(0))
    cp = F.softmax(logits_cnn, dim=1).numpy()[0]

    # AST
    inp = ast_fe([y_ast], sampling_rate=SR_AST, return_tensors="pt",
                 padding="max_length", max_length=1024, truncation=True)
    with torch.no_grad():
        logits_ast = ast_model(inp["input_values"])
    ap = F.softmax(logits_ast, dim=1).numpy()[0]

    # ResNet with InstanceNorm2d preprocessing
    S = librosa.feature.melspectrogram(y=y_base, sr=SR_RES, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
    mel_t = torch.from_numpy(S_db).float().unsqueeze(0).unsqueeze(0)
    mel_t = nn.InstanceNorm2d(1, affine=False)(mel_t)
    with torch.no_grad():
        logits_res = resnet_model(mel_t)
    rp = F.softmax(logits_res, dim=1).numpy()[0]

    return cp, ap, rp

# ═══════════ UI ═══════════

st.markdown("""
<div style="text-align:center; margin-bottom:8px;">
    <h1 style="font-size:2.2em; color:#1a1a2e; margin-bottom:2px;">🎵 Music Genre Classifier</h1>
    <p style="color:#666; font-size:0.95em; margin-top:0;">
        Classify music using an ensemble of <strong>EfficientNet-B0</strong>,
        <strong>Audio Spectrogram Transformer</strong> &amp; <strong>ResNet-50</strong>
    </p>
    <span style="background:#1a1a2e; color:white; padding:4px 16px; border-radius:16px;
                 font-size:0.8em; font-weight:600;">Live demo: 3-model ensemble · 0.9504 F1 | Full 6-model submission: 0.9614 F1</span>
</div>
""", unsafe_allow_html=True)

st.markdown("")

uploaded = st.file_uploader(
    "Upload a music clip (WAV, MP3, OGG, FLAC · up to 10 seconds)",
    type=["wav", "mp3", "ogg", "flac"],
    label_visibility="visible"
)

if uploaded:
    st.audio(uploaded, format="audio/wav")

    # Waveform visualization
    audio_bytes = uploaded.read()
    uploaded.seek(0)
    y_vis, sr_vis = librosa.load(BytesIO(audio_bytes), sr=SR_CNN, mono=True, duration=DUR)
    fig_wave, ax_wave = plt.subplots(figsize=(7, 1.8))
    librosa.display.waveshow(y_vis, sr=sr_vis, ax=ax_wave)
    ax_wave.set_title("Waveform", fontsize=10)
    ax_wave.set_xlabel(""); ax_wave.set_ylabel("")
    plt.tight_layout()
    buf_wave = BytesIO()
    fig_wave.savefig(buf_wave, format="png", dpi=100)
    plt.close(fig_wave)
    buf_wave.seek(0)
    st.image(buf_wave, caption="Waveform", use_container_width=True)

    # Mel spectrogram visualization
    S_vis = librosa.feature.melspectrogram(y=y_vis, sr=sr_vis, n_fft=N_FFT, hop_length=HOP,
                                           n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    S_db_vis = librosa.power_to_db(S_vis, ref=np.max)
    fig_mel, ax_mel = plt.subplots(figsize=(7, 2.5))
    img = librosa.display.specshow(S_db_vis, sr=sr_vis, hop_length=HOP,
                                   x_axis='time', y_axis='mel', ax=ax_mel, fmin=FMIN, fmax=FMAX)
    ax_mel.set_title("Mel Spectrogram", fontsize=10)
    fig_mel.colorbar(img, ax=ax_mel, format="%+2.0f dB")
    plt.tight_layout()
    buf_mel = BytesIO()
    fig_mel.savefig(buf_mel, format="png", dpi=100)
    plt.close(fig_mel)
    buf_mel.seek(0)
    st.image(buf_mel, caption="Mel Spectrogram", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing audio with 3 models..."):
        t_start = time.perf_counter()
        try:
            cp, ap, rp = get_probs(tmp_path)
        except Exception as e:
            log.error("Inference failed: %s", e)
            st.error("Could not process this file. Try a different clip.")
            st.stop()
        finally:
            os.unlink(tmp_path)
        elapsed = time.perf_counter() - t_start
        ep = W_CNN * cp + W_AST * ap + W_RES * rp

    genre = GENRES[ep.argmax()]
    conf = float(ep.max()) * 100
    log.info("pred=%s conf=%.1f%% latency=%.2fs", genre, conf, elapsed)

    st.caption(f"⏱ Analyzed in {elapsed:.1f}s")

    if ep.max() < 0.35:
        st.warning("⚠️ Low confidence — the audio doesn't strongly match any genre. Try a cleaner or longer clip.")
    else:
        emoji = GENRE_EMOJI.get(genre, '🎵')
        cg, ag, rg = GENRES[cp.argmax()], GENRES[ap.argmax()], GENRES[rp.argmax()]
        agree = cg == ag == rg
        conf_color = "#16a34a" if conf >= 80 else "#ca8a04" if conf >= 50 else "#dc2626"

        st.markdown("---")

        st.markdown(f"""
        <div style="text-align:center; padding:24px; background:#1a1a2e; border-radius:14px;
                    color:white; margin-bottom:16px;">
            <div style="font-size:2.8em; margin-bottom:2px;">{emoji}</div>
            <div style="font-size:1.6em; font-weight:700; text-transform:uppercase;
                        letter-spacing:2px;">{genre}</div>
            <div style="margin-top:8px;">
                <span style="background:rgba(255,255,255,0.15); padding:4px 14px; border-radius:16px;
                             font-size:0.88em;">{conf:.1f}% confidence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.85em; color:#555; font-weight:500;">Confidence</span>
                <span style="font-size:0.85em; color:{conf_color}; font-weight:700;">{conf:.1f}%</span>
            </div>
            <div style="background:#eee; border-radius:6px; height:6px; overflow:hidden;">
                <div style="background:{conf_color}; height:100%; width:{min(conf,100)}%;
                            border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Model Predictions")
        c1, c2, c3 = st.columns(3)
        for col, (name, w, g, probs) in zip(
            [c1, c2, c3],
            [("🧠 AST", W_AST, ag, ap),
             ("🏗️ ResNet-50", W_RES, rg, rp),
             ("⚡ EfficientNet", W_CNN, cg, cp)]
        ):
            mc = float(probs.max()) * 100
            match_icon = "✅" if g == genre else "❌"
            bg = "#f0fdf4" if g == genre else "#fef2f2"
            border = "#bbf7d0" if g == genre else "#fecaca"
            with col:
                st.markdown(f"""
                <div style="background:{bg}; border:1px solid {border}; border-radius:10px;
                            padding:12px; text-align:center;">
                    <div style="font-size:0.75em; color:#888; font-weight:500;">{name} ({int(w*100)}%)</div>
                    <div style="font-size:1.1em; font-weight:700; margin:4px 0; color:#333;">
                        {GENRE_EMOJI.get(g,'')} {g.title()} {match_icon}</div>
                    <div style="font-size:0.78em; color:#888;">{mc:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        if agree:
            st.success("✅ All 3 models agree on the prediction")
        else:
            st.warning("⚠️ Models disagree — ensemble resolves the final prediction")

        st.markdown("")
        st.markdown("##### Genre Probabilities")
        import pandas as pd
        prob_df = pd.DataFrame({
            'Genre': [f"{GENRE_EMOJI.get(g,'')} {g.title()}" for g in GENRES],
            'Probability (%)': (ep * 100).round(2)
        }).sort_values('Probability (%)', ascending=True)
        st.bar_chart(prob_df.set_index('Genre'), horizontal=True, color="#1a1a2e")

        sp = sorted(enumerate(ep), key=lambda x: x[1], reverse=True)
        top3 = " → ".join([f"{GENRE_EMOJI.get(GENRES[i],'')} {GENRES[i].title()} ({p*100:.1f}%)" for i, p in sp[:3]])
        st.caption(f"**Top 3:** {top3}")

st.markdown("---")
st.caption("""
**How it works:** Your audio is processed by three models —
EfficientNet-B0 (local patterns), AST (global structure, pretrained on AudioSet),
and ResNet-50 (architectural diversity). Predictions are combined with optimized weights (10/60/30%).
· Built for the Messy Mashup competition — IIT Madras DL & GenAI Project.
""")
