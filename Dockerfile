FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TERM=xterm-256color \
    OLLAMA_HOST=http://host.docker.internal:11434

# Core ML packages used by the project
RUN pip install --no-cache-dir \
    pandas \
    statsmodels \
    groq \
    google-genai \
    datasets \
    loguru \
    pyarrow \
    matplotlib
