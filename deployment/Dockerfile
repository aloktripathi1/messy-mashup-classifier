FROM python:3.12-slim

RUN pip install --no-cache-dir streamlit==1.35.0 torch==2.3.1 torchaudio==2.3.1 timm==1.0.3 transformers==4.41.2 librosa==0.10.2 numpy==1.26.4

WORKDIR /app
COPY . .

# Pre-download AST model during build (avoids 403 at runtime)
RUN python -c "from transformers import ASTFeatureExtractor, ASTForAudioClassification; \
    ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593'); \
    ASTForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593', num_labels=10, ignore_mismatched_sizes=True)"

EXPOSE 7860
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.maxUploadSize=200"]