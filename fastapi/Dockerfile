FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

RUN mkdir -p /app/logs
COPY . .

EXPOSE 8672
CMD ["python3", "main.py"]