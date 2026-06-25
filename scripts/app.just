# app — tunnel + proxy

# open chat UI
chat:
    @echo "→ http://localhost:8501"
    xdg-open http://localhost:8501 2>/dev/null || open http://localhost:8501 2>/dev/null || true

# start SSH tunnel to register bot on youare.bot
tunnel:
    ssh -o StrictHostKeyChecking=no -i portforward_key -N -R 0.0.0.0:42067:localhost:8672 forwarduser@158.160.135.246 &
    @echo "Registered at: http://158.160.135.246:42067"

# kill the SSH tunnel
tunnel-kill:
    pkill -f "ssh.*portforward_key.*forwarduser" || true
