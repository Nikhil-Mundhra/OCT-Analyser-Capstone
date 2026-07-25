#!/bin/bash

# ==============================================================================
# MUTUAL EXCLUSION & MUTEX CLEANUP PRE-FLIGHT
# Automatically stops any previously running start.sh, uvicorn, celery, or next-dev
# ==============================================================================
PID_FILE="logs/start_services.pid"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "🛑 Previous start.sh instance (PID $OLD_PID) detected. Terminating..."
        kill -9 "$OLD_PID" 2>/dev/null
    fi
    rm -f "$PID_FILE"
fi

# Save current PID for mutual exclusion
mkdir -p logs
echo $$ > "$PID_FILE"

echo "🧹 Pre-flight cleanup: clearing old backend & frontend processes..."
pkill -9 -f "uvicorn backend.oct_analyzer.api:app" 2>/dev/null || true
pkill -9 -f "celery -A backend.oct_analyzer.celery_app" 2>/dev/null || true
pkill -9 -f "next-dev" 2>/dev/null || true

# Free ports 8000 and 3000 if occupied
for PORT in 8000 3000; do
    PIDS=$(lsof -t -i:$PORT 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "  - Clearing port $PORT (PIDs: $PIDS)..."
        kill -9 $PIDS 2>/dev/null || true
    fi
done

# Device & Offload selection: priority CLI arg -> environment variable -> default "cpu"
ARG_DEVICE=$(echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/^--//')
if [ "$ARG_DEVICE" = "gpu" ]; then
    export OCT_LOCAL_DEVICE="auto"
    export OCT_REMOTE_OFFLOAD="false"
elif [ "$ARG_DEVICE" = "remote" ]; then
    export OCT_LOCAL_DEVICE="cpu"
    export OCT_REMOTE_OFFLOAD="true"
elif [ -n "$ARG_DEVICE" ]; then
    export OCT_LOCAL_DEVICE="$ARG_DEVICE"
    export OCT_REMOTE_OFFLOAD="false"
else
    export OCT_LOCAL_DEVICE=${OCT_LOCAL_DEVICE:-cpu}
    export OCT_REMOTE_OFFLOAD=${OCT_REMOTE_OFFLOAD:-false}
fi

echo "Starting OCT Analyzer Services (Device Mode: $OCT_LOCAL_DEVICE | Remote Offload: $OCT_REMOTE_OFFLOAD)..."

echo "0. Ensuring Redis is running..."
brew services start redis > /dev/null 2>&1 || /opt/homebrew/bin/redis-server --daemonize yes > /dev/null 2>&1
sleep 1

echo "1. Starting Celery Worker..."
PYTHONPATH=. uv run celery -A backend.oct_analyzer.celery_app worker --loglevel=info --pool=solo > logs/celery.log 2>&1 &
CELERY_PID=$!

echo "2. Starting FastAPI Backend..."
PYTHONPATH=. uv run python -m uvicorn backend.oct_analyzer.api:app --reload --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!

echo "3. Starting React Frontend..."
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "----------------------------------------"
echo "✅ All services are booting up in the background!"
echo "🌐 Frontend: http://localhost:3000"
echo "⚙️  Backend:  http://127.0.0.1:8000"
echo "📄 Logs are written to 'logs/'"
echo "----------------------------------------"

cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $CELERY_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null
    rm -f "$PID_FILE"
}

trap cleanup SIGINT SIGTERM EXIT

wait $FRONTEND_PID
