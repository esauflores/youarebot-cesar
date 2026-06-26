# test — API endpoints

# run all tests
test:
    @echo "=== health ==="
    @curl -sf http://localhost:8672/health > /dev/null && echo "orchestrator  ✓" || echo "orchestrator  ✗"
    @curl -sf http://localhost:8000/health > /dev/null && echo "classifier   ✓" || echo "classifier   ✗"
    @echo "=== predict ==="
    @curl -sf -X POST http://localhost:8672/predict -H 'Content-Type: application/json' -d '{"text":"Hello"}' | python3 -m json.tool
    @echo "=== message ==="
    @curl -sf -X POST http://localhost:8672/get_message -H 'Content-Type: application/json' -d '{"last_msg_text":"Hi"}' | python3 -m json.tool
    @echo "=== llm ==="
    @curl -sf http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}' | python3 -m json.tool
    @echo "=== all done ==="

# test classifier (/predict)
test-predict:
    curl -s -X POST http://localhost:8672/predict \
      -H 'Content-Type: application/json' \
      -d '{"text":"Hello how are you doing today?"}' | python3 -m json.tool

# test LLM (/get_message)
test-message:
    curl -s -X POST http://localhost:8672/get_message \
      -H 'Content-Type: application/json' \
      -d '{"dialog_id":"00000000-0000-0000-0000-000000000001","last_msg_text":"Hello, how are you?","last_message_id":"00000000-0000-0000-0000-000000000002"}' | python3 -m json.tool

# check llama.cpp server
test-llm:
    curl -s http://localhost:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"Say hi in one sentence"}],"max_tokens":32}' | python3 -m json.tool

# check health of all services
test-health:
    @echo "orchestrator:" && curl -s http://localhost:8672/health | python3 -m json.tool
    @echo "classifier:" && curl -s http://localhost:8000/health | python3 -m json.tool
    @echo "streamlit:" && curl -s http://localhost:8501/_stcore/health 2>/dev/null | python3 -m json.tool || echo "streamlit OK"
