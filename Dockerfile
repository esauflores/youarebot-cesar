FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl supervisor build-essential cmake && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# classifier
COPY classifier/requirements.txt /tmp/clf.txt
RUN pip install --no-cache-dir -r /tmp/clf.txt
COPY classifier/classifier.onnx /app/classifier.onnx
COPY classifier/main.py /app/classifier_main.py

# orchestrator
COPY orchestrator/requirements.txt /tmp/orch.txt
RUN pip install --no-cache-dir -r /tmp/orch.txt
COPY orchestrator/main.py /app/orchestrator_main.py

# streamlit  
COPY streamlit/requirements.txt /tmp/st.txt
RUN pip install --no-cache-dir -r /tmp/st.txt
COPY streamlit/chat_ui.py /app/streamlit_app.py

# LLM
RUN pip install --no-cache-dir 'llama-cpp-python[server]'
RUN mkdir -p /models && \
    curl -L -o /models/Qwen3.5-0.8B-Q4_K_M.gguf \
    https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8672 8000 8080 8501
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
