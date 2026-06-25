# docker — compose shortcuts

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
