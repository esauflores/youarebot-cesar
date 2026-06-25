# sync — push to GPU machine

SSH_HOST := "s_01kvpzzh31ch3abnxp9tfmv6aa@ssh.lightning.ai"
REMOTE_PATH := "/teamspace/studios/this_studio/youarebot-cesar"

sync:
    rsync -avz --delete \
        --exclude '.venv/' \
        --exclude '__pycache__/' \
        --exclude '.git/' \
        --exclude 'checkpoints/' \
        --exclude 'models/' \
        --exclude 'data/' \
        ./* \
        "{{SSH_HOST}}:{{REMOTE_PATH}}/"
    @echo "Synced → {{SSH_HOST}}:{{REMOTE_PATH}}"
