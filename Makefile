# Makefile for TensorGuardFlow
# Automation for build, test, and security verification

.PHONY: install test agent bench clean reports lint setup ci typecheck

# Default target
all: test

# Installation (uses pyproject.toml for dependency management)
install:
	pip install -e ".[all]"

# Install core only (minimal dependencies)
install-core:
	pip install -e .

# Install with specific extras
install-dev:
	pip install -e ".[dev]"

install-bench:
	pip install -e ".[bench]"

# Project Setup
setup:
	mkdir -p keys/identity keys/inference keys/aggregation artifacts
	python scripts/setup_env.py

# Testing
test:
	@echo "--- Running Holistic Security Fabric Tests ---"
	python -m pytest tests/

# Agent Orchestration
agent:
	@echo "--- Starting TensorGuard Unified Agent ---"
	export PYTHONPATH=src && python -m tensorguard.agent.daemon

# Benchmarking Subsystem
bench:
	@echo "--- Running TensorGuard Microbenchmarks ---"
	export PYTHONPATH=src && python -m tensorguard.bench.cli micro
	@echo "--- Running Privacy Eval ---"
	export PYTHONPATH=src && python -m tensorguard.bench.cli privacy
	@echo "--- Generating Benchmarking Report ---"
	export PYTHONPATH=src && python -m tensorguard.bench.cli report

# Linting
lint:
	@echo "--- Running Linter (ruff) ---"
	ruff check src/

# Type checking
typecheck:
	@echo "--- Running Type Checker (mypy) ---"
	mypy src/

# CI target: install, lint, (optional type check), tests
ci: install lint typecheck test
	@echo "--- CI checks completed ---"

# Cleanup
clean:
	@echo "--- Cleaning temporary files ---"
	rm -rf .pytest_cache
	rm -rf artifacts/metrics artifacts/privacy artifacts/robustness artifacts/evidence_pack
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f artifacts/report.html
