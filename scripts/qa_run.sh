#!/bin/bash
# QA Run Script for TensorGuardFlow

MODE=$1

case $MODE in
    "unit")
        echo "Running Unit Tests..."
        pytest -m "unit" -q
        ;;
    "integration")
        echo "Running Integration Tests..."
        pytest -m "integration" -q
        ;;
    "e2e")
        echo "Running End-to-End Tests..."
        pytest -m "e2e" -q
        ;;
    "security")
        echo "Running Security Tests..."
        pytest -m "security" -q
        ;;
    "perf")
        echo "Running Performance Sanity Checks..."
        pytest -m "perf" -q
        ;;
    "full")
        echo "Running Full Test Suite..."
        pytest -q
        ;;
    *)
        echo "Usage: $0 {unit|integration|e2e|security|perf|full}"
        exit 1
        ;;
esac
