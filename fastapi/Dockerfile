FROM python:3.12-slim

# ponytail: psycopg2-binary instead of compiling from source, no gcc needed

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/logs
COPY . .

EXPOSE 8672
CMD ["python3", "main.py"]
