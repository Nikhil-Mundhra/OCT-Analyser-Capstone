SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(abspath $(VENV)/bin/python)
PIP := $(VENV_PYTHON) -m pip
PYTEST := $(VENV_PYTHON) -m pytest
UVICORN := $(VENV_PYTHON) -m uvicorn
HOST ?= 127.0.0.1
API_PORT ?= 8000
WEB_PORT ?= 3000

.PHONY: help check-ports venv install build test run run-hf-local docker-up docker-down clean

help:
	@echo "Local OCT Analyzer MVP"
	@echo ""
	@echo "Targets:"
	@echo "  make run           Create/update env, build frontend, and run API + web app"
	@echo "  make run-hf-local  Run the HF Space image locally on :7860 (env parity test)"
	@echo "  make docker-up     Start backend + HF Space mirror via docker-compose"
	@echo "  make docker-down   Stop all docker-compose services"
	@echo "  make install       Install Python and Node dependencies"
	@echo "  make build         Build the Next.js frontend bundle"
	@echo "  make test          Run the Python test suite"
	@echo "  make clean         Remove local runtime/cache artifacts"
	@echo ""
	@echo "URLs after make run:"
	@echo "  Frontend: http://$(HOST):$(WEB_PORT) or next open port"
	@echo "  API docs: http://$(HOST):$(API_PORT)/docs or next open port"
	@echo ""
	@echo "Local inference env vars:"
	@echo "  OCT_LOCAL_DEVICE=cpu|mps|cuda   Override auto device selection (default: auto)"
	@echo "  NEXT_PUBLIC_SEGMENTATION_API_URL Set in frontend/.env.local (default: http://127.0.0.1:8000)"

check-ports:
	@if lsof -nP -iTCP:$(API_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $(API_PORT) is already in use. Try: API_PORT=8001 make run"; \
		exit 1; \
	fi
	@if lsof -nP -iTCP:$(WEB_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $(WEB_PORT) is already in use. Try: WEB_PORT=3001 make run"; \
		exit 1; \
	fi

venv:
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		$(PYTHON) -m venv "$(VENV)"; \
		$(PIP) install --upgrade pip; \
	fi

install: venv
	@$(PIP) install -r web-app/backend/requirements.txt
	@npm --prefix web-app/frontend install

build: install
	@npm --prefix web-app/frontend run build

test: install
	@PYTHONPATH=web-app $(PYTEST) -c web-app/backend/pytest.ini web-app/backend/tests

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
	PYTHONPATH=web-app $(UVICORN) backend.oct_analyzer.api:app --host $(HOST) --port $$RESOLVED_API_PORT & \
	API_PID=$$!; \
	sleep 1; \
	if ! kill -0 $$API_PID 2>/dev/null; then \
		wait $$API_PID; \
		exit $$?; \
	fi; \
	cd web-app/frontend && npm start -- -p $$RESOLVED_WEB_PORT & \
	WEB_PID=$$!; \
	sleep 1; \
	if ! kill -0 $$WEB_PID 2>/dev/null; then \
		kill $$API_PID 2>/dev/null || true; \
		wait $$WEB_PID; \
		exit $$?; \
	fi; \
	wait $$API_PID $$WEB_PID

clean:
	@rm -rf .coverage __pycache__ web-app/backend/oct_analyzer/__pycache__ web-app/backend/tests/__pycache__ runtime_uploads web-app/frontend/dist web-app/frontend/.next

# ---------------------------------------------------------------------------
# Local HF Space mirror targets
# ---------------------------------------------------------------------------

# Run only the HF Space Dockerfile locally on port 7860.
# This lets you test the EXACT container HF would run, without pushing to main.
# After starting, point your frontend at it:
#   NEXT_PUBLIC_SEGMENTATION_API_URL=http://localhost:7860  (in frontend/.env.local)
run-hf-local:
	@echo "Building and starting HF Space mirror on http://localhost:7860 ..."
	@echo "Set OCT_LOCAL_DEVICE=cpu|mps|cuda to control the device (default: cpu)."
	docker build \
		-t oct-hf-space-local \
		hf_space
	docker run --rm \
		-p 7860:7860 \
		-v "$(PWD)/hf_space/weights:/app/weights" \
		-e OCT_LOCAL_DEVICE=$${OCT_LOCAL_DEVICE:-cpu} \
		-e KMP_DUPLICATE_LIB_OK=TRUE \
		oct-hf-space-local

# Start backend + HF Space mirror together via docker-compose.
docker-up:
	docker-compose up --build

# Tear down all docker-compose services.
docker-down:
	docker-compose down

# ---------------------------------------------------------------------------
# Training targets
# ---------------------------------------------------------------------------

train-convnext:
	@echo "Training Multi-Head ConvNeXt model..."
	export PYTHONPATH=$$(pwd)/training/classification:$$PYTHONPATH && \
	KMP_DUPLICATE_LIB_OK=TRUE $(VENV_PYTHON) training/classification/scripts/train_convnext.py \
		--config training/classification/config/hierarchy.yaml

smoke-test:
	@echo "Running smoke test on Multi-Head ConvNeXt pipeline..."
	export PYTHONPATH=$$(pwd)/training/classification:$$PYTHONPATH && \
	KMP_DUPLICATE_LIB_OK=TRUE $(VENV_PYTHON) training/classification/scripts/train_convnext.py \
		--config training/classification/config/hierarchy.yaml \
		--smoke-test --epochs-warmup 1 --epochs-finetune 1 --batch-size 8

