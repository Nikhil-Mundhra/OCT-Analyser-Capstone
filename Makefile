SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/python -m pip
PYTEST := $(VENV)/bin/python -m pytest
UVICORN := $(VENV)/bin/python -m uvicorn
HOST ?= 127.0.0.1
API_PORT ?= 8000
WEB_PORT ?= 5173

.PHONY: help check-ports venv install build test run clean

help:
	@echo "Local OCT Analyzer MVP"
	@echo ""
	@echo "Targets:"
	@echo "  make run      Create/update env, build frontend, and run API + web app"
	@echo "  make install  Install Python and Node dependencies"
	@echo "  make build    Build the React frontend bundle"
	@echo "  make test     Run the Python test suite"
	@echo "  make clean    Remove local runtime/cache artifacts"
	@echo ""
	@echo "URLs after make run:"
	@echo "  Frontend: http://$(HOST):$(WEB_PORT)"
	@echo "  API docs: http://$(HOST):$(API_PORT)/docs"

check-ports:
	@if lsof -nP -iTCP:$(API_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $(API_PORT) is already in use. Try: API_PORT=8001 make run"; \
		exit 1; \
	fi
	@if lsof -nP -iTCP:$(WEB_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $(WEB_PORT) is already in use. Try: WEB_PORT=5174 make run"; \
		exit 1; \
	fi

venv:
	@test -x "$(VENV)/bin/python" || $(PYTHON) -m venv "$(VENV)"
	@$(PIP) install --upgrade pip

install: venv
	@$(PIP) install -r requirements.txt
	@npm install

build: install
	@npm run build

test: install
	@$(PYTEST)

run: check-ports build
	@echo "Starting API at http://$(HOST):$(API_PORT)"
	@echo "Starting frontend at http://$(HOST):$(WEB_PORT)"
	@echo "Press Ctrl-C to stop both servers."
	@trap 'kill $$API_PID $$WEB_PID 2>/dev/null || true' INT TERM EXIT; \
	$(UVICORN) src.api:app --host $(HOST) --port $(API_PORT) & \
	API_PID=$$!; \
	sleep 1; \
	if ! kill -0 $$API_PID 2>/dev/null; then \
		wait $$API_PID; \
		exit $$?; \
	fi; \
	$(PYTHON) -m http.server $(WEB_PORT) --bind $(HOST) & \
	WEB_PID=$$!; \
	wait $$API_PID $$WEB_PID

clean:
	@rm -rf .coverage __pycache__ src/__pycache__ tests/__pycache__ runtime_uploads
