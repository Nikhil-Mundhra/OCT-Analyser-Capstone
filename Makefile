SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(abspath $(VENV)/bin/python)
PIP := $(VENV_PYTHON) -m pip
PYTEST := $(VENV_PYTHON) -m pytest
UVICORN := $(VENV_PYTHON) -m uvicorn
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
	@echo "  Frontend: http://$(HOST):$(WEB_PORT) or next open port"
	@echo "  API docs: http://$(HOST):$(API_PORT)/docs or next open port"

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
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		$(PYTHON) -m venv "$(VENV)"; \
		$(PIP) install --upgrade pip; \
	fi

install: venv
	@$(PIP) install -r backend/requirements.txt
	@npm --prefix frontend install

build: install
	@npm --prefix frontend run build

test: install
	@$(PYTEST) -c backend/pytest.ini backend/tests

run: build
	@set -e; \
	find_open_port() { \
		port="$$1"; \
		while lsof -nP -iTCP:$$port -sTCP:LISTEN >/dev/null 2>&1; do \
			port=$$((port + 1)); \
		done; \
		echo "$$port"; \
	}; \
	RESOLVED_API_PORT=$$(find_open_port "$(API_PORT)"); \
	RESOLVED_WEB_PORT=$$(find_open_port "$(WEB_PORT)"); \
	API_URL="http://$(HOST):$$RESOLVED_API_PORT"; \
	WEB_URL="http://$(HOST):$$RESOLVED_WEB_PORT"; \
	if [ "$$RESOLVED_API_PORT" != "$(API_PORT)" ]; then \
		echo "Port $(API_PORT) is in use. Using API port $$RESOLVED_API_PORT instead."; \
	fi; \
	if [ "$$RESOLVED_WEB_PORT" != "$(WEB_PORT)" ]; then \
		echo "Port $(WEB_PORT) is in use. Using frontend port $$RESOLVED_WEB_PORT instead."; \
	fi; \
	if [ "$$RESOLVED_API_PORT" != "8000" ]; then \
		WEB_URL="$$WEB_URL/?apiBase=$$API_URL"; \
	fi; \
	echo "Starting API at $$API_URL"; \
	echo "Starting frontend at $$WEB_URL"; \
	echo "Press Ctrl-C to stop both servers."; \
	cleanup() { \
		kill $$API_PID $$WEB_PID 2>/dev/null || true; \
		wait $$API_PID $$WEB_PID 2>/dev/null || true; \
	}; \
	trap cleanup INT TERM EXIT; \
	$(UVICORN) backend.oct_analyzer.api:app --host $(HOST) --port $$RESOLVED_API_PORT & \
	API_PID=$$!; \
	sleep 1; \
	if ! kill -0 $$API_PID 2>/dev/null; then \
		wait $$API_PID; \
		exit $$?; \
	fi; \
	cd frontend && npx serve . -l $$RESOLVED_WEB_PORT & \
	WEB_PID=$$!; \
	sleep 1; \
	if ! kill -0 $$WEB_PID 2>/dev/null; then \
		kill $$API_PID 2>/dev/null || true; \
		wait $$WEB_PID; \
		exit $$?; \
	fi; \
	wait $$API_PID $$WEB_PID

clean:
	@rm -rf .coverage __pycache__ backend/oct_analyzer/__pycache__ backend/tests/__pycache__ runtime_uploads frontend/dist
