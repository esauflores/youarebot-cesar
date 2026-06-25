# test — API + LLM

# test the chat bot API
test-bot:
    curl -s -X POST http://localhost:8672/get_message \
      -H 'Content-Type: application/json' \
      -d '{"dialog_id":"00000000-0000-0000-0000-000000000001","last_msg_text":"Hello, how are you?","last_message_id":"00000000-0000-0000-0000-000000000002"}' | python3 -m json.tool

# full classifier test suite
test-clf:
    #!/usr/bin/env bash
    set -euo pipefail
    BASE=http://localhost:8672
    DA=aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa
    DB=bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb

    echo "=== 1. Health ==="
    s=$(curl -s -o /dev/null -w '%{http_code}' $BASE/health)
    echo "$s /health"; [ "$s" = 200 ]

    echo "=== 2. Single predict ==="
    r=$(curl -s -X POST $BASE/predict -H 'Content-Type: application/json' \
      -d '{"text":"Hi, how are you?","dialog_id":"'$DA'","id":"c1","participant_index":0}')
    p0=$(echo "$r" | jq '.is_bot_probability')
    echo "prob=$p0"; [ "$p0" != null ]

    echo "=== 3. Dialog context ==="
    p1=$(curl -s -X POST $BASE/predict -H 'Content-Type: application/json' \
      -d '{"text":"Hi, how are you?","dialog_id":"'$DA'","id":"c2","participant_index":0}' | jq '.is_bot_probability')
    p2=$(curl -s -X POST $BASE/predict -H 'Content-Type: application/json' \
      -d '{"text":"I can help you write a structured answer.","dialog_id":"'$DA'","id":"c3","participant_index":1}' | jq '.is_bot_probability')
    p3=$(curl -s -X POST $BASE/predict -H 'Content-Type: application/json' \
      -d '{"text":"As an AI language model, I can solve this task.","dialog_id":"'$DA'","id":"c4","participant_index":1}' | jq '.is_bot_probability')
    echo "[0]=$p1 [1]=$p2 [1]=$p3"
    [ "$(echo "$p3 > $p1" | bc)" = 1 ]

    echo "=== 4. Fresh dialog ==="
    pb=$(curl -s -X POST $BASE/predict -H 'Content-Type: application/json' \
      -d '{"text":"Hi, how are you?","dialog_id":"'$DB'","id":"c5","participant_index":0}' | jq '.is_bot_probability')
    echo "prob=$pb"

    echo "=== 5. Bad request ==="
    s=$(curl -s -o /dev/null -w '%{http_code}' -X POST $BASE/predict \
      -H 'Content-Type: application/json' \
      -d '{"text":"Hello","dialog_id":"bad-id","id":"bad-id","participant_index":0}')
    [ "$s" = 422 ]

    echo "=== All passed ==="

# single classifier call (quick smoke)
test-clf-quick:
    curl -s -X POST http://localhost:8672/predict \
      -H 'Content-Type: application/json' \
      -d '{"id":"00000000-0000-0000-0000-000000000001","dialog_id":"00000000-0000-0000-0000-000000000010","text":"Hello how are you doing today?","participant_index":0}' | python3 -m json.tool

# check llama.cpp server
test-llm:
    curl -s http://localhost:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer not-needed' \
      -d '{"model":"qwen2.5-0.5b-instruct","messages":[{"role":"user","content":"Say hi in one sentence"}],"max_tokens":32}' | python3 -m json.tool
