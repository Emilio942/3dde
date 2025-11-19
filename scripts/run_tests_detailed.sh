#!/bin/bash
# Run specific test categories

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function run_test_category() {
    local test_file=$1
    local description=$2
    
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing: $description${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    if pytest "$test_file" -v --tb=short; then
        echo -e "${GREEN}✓ $description: PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ $description: FAILED${NC}"
        return 1
    fi
}

# Run test categories
echo -e "${GREEN}=================================="
echo "3D Diffusion Model - Test Suite"
echo -e "==================================${NC}"

run_test_category "src/tests/test_precompute.py" "Precompute (Φ, Σ)"
run_test_category "src/tests/test_forward.py" "Forward Noising"
run_test_category "src/tests/test_sampling.py" "Reverse Sampling"
run_test_category "src/tests/test_models.py" "Model Architecture"
run_test_category "src/tests/test_training.py" "Training Pipeline"
run_test_category "src/tests/test_integration.py" "Integration Tests"

echo ""
echo -e "${GREEN}=================================="
echo "All Tests Completed!"
echo -e "==================================${NC}"
