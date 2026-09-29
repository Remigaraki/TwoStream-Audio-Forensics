# CPU-only image for the TwoStream-Audio-Forensics demo (Hugging Face Docker Space).
# Build context = the directory produced by demo/stage_space.py.
FROM python:3.11.9-slim-bookworm

# FFmpeg as requested for media tooling; the served decode path is libsndfile
# (bundled in the soundfile wheel), which is what the evaluation pipeline used.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    DEMO_TORCH_THREADS=2 \
    DEMO_MAX_CONCURRENT=1
WORKDIR /home/user/app

# Fully pinned set (incl. transitive deps) captured from a verified build.
COPY --chown=user demo/requirements.lock demo/requirements.lock
RUN pip install --no-cache-dir --user --no-deps -r demo/requirements.lock

COPY --chown=user . .

# Fail the build if any artifact is missing, an LFS pointer, or tampered with.
RUN python -c "from demo import inference as i; [i.load_model(m) for m in i.MODELS]; i.load_thresholds(); print('assets ok')"

WORKDIR /home/user/app/demo
EXPOSE 7860
# XSRF protection is disabled because HF serves Spaces in a cross-site iframe,
# where Streamlit's XSRF cookie is blocked and uploads fail with 403. The app
# has no accounts or state-changing actions for XSRF to protect.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
