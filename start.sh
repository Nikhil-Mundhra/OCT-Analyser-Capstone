#!/bin/bash

echo "Starting OCT Analyzer Services..."

# Create a logs directory if it doesn't exist
mkdir -p logs

echo "1. Starting Celery Worker..."
PYTHONPATH=. uv run celery -A backend.oct_analyzer.celery_app worker --loglevel=info --pool=solo > logs/celery.log 2>&1 &
CELERY_PID=$!

echo "2. Starting FastAPI Backend..."
PYTHONPATH=. uv run python -m uvicorn backend.oct_analyzer.api:app --reload --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!

echo "3. Starting React Frontend..."
cd frontend
# Ensure dependencies are installed just in case
npm install > /dev/null 2>&1
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "----------------------------------------"
echo "✅ All services are booting up in the background!"
echo "🌐 Frontend: http://localhost:3000 (or 3001)"
echo "⚙️  Backend:  http://127.0.0.1:8000"
echo "📄 Logs are being written to the 'logs/' directory."
echo "----------------------------------------"
echo "Services are running. Press [Ctrl+C] to stop everything and exit."
echo ""

# Function to cleanly kill all background processes when the user presses Ctrl+C
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $CELERY_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null
}

# Trap Ctrl+C (SIGINT) and run the cleanup function
trap cleanup SIGINT SIGTERM

# Wait for processes and capture their exit codes
wait $CELERY_PID
echo "Celery Worker exited with code $?"

wait $BACKEND_PID
echo "FastAPI Backend exited with code $?"

wait $FRONTEND_PID
echo "React Frontend exited with code $?"

echo "Goodbye!"
