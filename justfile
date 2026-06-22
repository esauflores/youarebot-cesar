# youarebot-cesar — docker compose shortcuts

default:
    @just --list

# start everything
up:
    docker compose up --build

# start in background
upd:
    docker compose up --build -d

# stop everything
down:
    docker compose down

# stop and remove volumes
nuke:
    docker compose down -v

# rebuild images only
build:
    docker compose build

# view logs
logs:
    docker compose logs -f

# check service status
ps:
    docker compose ps

# download the GGUF model (one-time)
model:
    ./setup.sh

# open chat UI
chat:
    @echo "→ http://localhost:8501"
    xdg-open http://localhost:8501 2>/dev/null || open http://localhost:8501 2>/dev/null || true

# test the chat bot API
test-bot:
    curl -s -X POST http://localhost:8672/get_message \
      -H 'Content-Type: application/json' \
      -d '{"dialog_id":"00000000-0000-0000-0000-000000000001","last_msg_text":"Hello, how are you?","last_message_id":"00000000-0000-0000-0000-000000000002"}' | python3 -m json.tool

# test the classifier API
test-clf:
    curl -s -X POST http://localhost:8672/predict \
      -H 'Content-Type: application/json' \
      -d '{"id":"00000000-0000-0000-0000-000000000001","dialog_id":"00000000-0000-0000-0000-000000000010","text":"Hello how are you doing today?","participant_index":0}' | python3 -m json.tool

# start SSH tunnel to register bot on youare.bot
tunnel:
    ssh -o StrictHostKeyChecking=no -i portforward_key -N -R 0.0.0.0:42067:localhost:8672 forwarduser@158.160.135.246 &
    @echo "Registered at: http://158.160.135.246:42067"

# kill the SSH tunnel
tunnel-kill:
    pkill -f "ssh.*portforward_key.*forwarduser" || true

# check llama.cpp server
llm-test:
    curl -s http://localhost:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer not-needed' \
      -d '{"model":"qwen2.5-0.5b-instruct","messages":[{"role":"user","content":"Say hi in one sentence"}],"max_tokens":32}' | python3 -m json.tool
