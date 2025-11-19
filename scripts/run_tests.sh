#!/bin/bash
# Run all tests

echo "=================================="
echo "Running 3D Diffusion Model Tests"
echo "=================================="

# Run all tests with coverage
pytest src/tests/ \
    --verbose \
    --tb=short \
    --cov=src \
    --cov-report=html \
    --cov-report=term-missing

echo ""
echo "=================================="
echo "Test Results Summary"
echo "=================================="
echo "HTML coverage report: htmlcov/index.html"
