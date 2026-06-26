# docker — compose shortcuts

# start everything
up:
    docker compose -f compose.yml up --build

# start in background
upd:
    docker compose -f compose.yml up --build -d

# stop everything
down:
    docker compose -f compose.yml down

# stop and remove volumes
nuke:
    docker compose -f compose.yml down -v

# rebuild images only
build:
    docker compose -f compose.yml build

# view logs
logs:
    docker compose -f compose.yml logs -f

# check service status
ps:
    docker compose -f compose.yml ps
