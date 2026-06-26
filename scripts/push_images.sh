#!/bin/bash
set -e

REGISTRY="registry.fly.io/esauflores2025-gmail-com-harbour-images"

echo "Logging in..."
fly auth docker

for svc in classifier orchestrator streamlit llm; do
    echo "=== $svc ==="
    docker build -t "$REGISTRY:$svc" "./$svc"
    docker push "$REGISTRY:$svc"
    echo
done

echo "Done. Run: fly deploy"
