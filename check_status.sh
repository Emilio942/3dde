#!/bin/bash
# Quick Test Status Check

echo "========================================"
echo "3D Diffusion Model - Test Status"
echo "========================================"
echo ""

# Check Python
python_version=$(python --version 2>&1)
echo "✓ Python: $python_version"

# Check pip
pip_version=$(python -m pip --version | head -1)
echo "✓ pip: $pip_version"

# Check pytest
if python -c "import pytest" 2>/dev/null; then
    pytest_version=$(python -c "import pytest; print(pytest.__version__)")
    echo "✓ pytest: $pytest_version"
else
    echo "✗ pytest: Not installed"
    echo "  Install with: pip install pytest pytest-cov"
fi

echo ""
echo "========================================"
echo "File Checks"
echo "========================================"

# Count files
py_files=$(find src -name "*.py" -type f | wc -l)
test_files=$(find src/tests -name "test_*.py" -type f | wc -l)
config_files=$(find experiments/configs -name "*.yaml" -type f 2>/dev/null | wc -l)

echo "✓ Python modules: $py_files"
echo "✓ Test files: $test_files"
echo "✓ Config files: $config_files"

echo ""
echo "========================================"
echo "Syntax Validation"
echo "========================================"

# Check Python syntax
syntax_errors=0
for file in $(find src -name "*.py" -type f); do
    if ! python -m py_compile "$file" 2>/dev/null; then
        echo "✗ Syntax error: $file"
        ((syntax_errors++))
    fi
done

if [ $syntax_errors -eq 0 ]; then
    echo "✓ All Python files have valid syntax ($py_files files)"
else
    echo "✗ Found $syntax_errors files with syntax errors"
fi

echo ""
echo "========================================"
echo "Dependencies Status"
echo "========================================"

# Check for PyTorch
if python -c "import torch" 2>/dev/null; then
    torch_version=$(python -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch: $torch_version"
    torch_installed=true
else
    echo "✗ PyTorch: Not installed"
    echo "  Install with: pip install torch"
    torch_installed=false
fi

# Check for PyTorch Geometric
if python -c "import torch_geometric" 2>/dev/null; then
    pyg_version=$(python -c "import torch_geometric; print(torch_geometric.__version__)")
    echo "✓ PyTorch Geometric: $pyg_version"
else
    echo "✗ PyTorch Geometric: Not installed"
    echo "  Install with: pip install torch-geometric"
fi

# Check for other dependencies
for pkg in numpy scipy matplotlib pyyaml; do
    if python -c "import $pkg" 2>/dev/null; then
        version=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "installed")
        echo "✓ $pkg: $version"
    else
        echo "✗ $pkg: Not installed"
    fi
done

echo ""
echo "========================================"
echo "Next Steps"
echo "========================================"

if [ "$torch_installed" = false ]; then
    echo ""
    echo "To install all dependencies:"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "To run tests (after installing dependencies):"
    echo "  pytest src/tests/ -v"
    echo ""
    echo "To train a model:"
    echo "  python train_example.py"
else
    echo ""
    echo "✅ Core dependencies installed!"
    echo ""
    echo "Run tests:"
    echo "  pytest src/tests/ -v"
    echo ""
    echo "Run quick example:"
    echo "  python train_example.py"
fi

echo ""
echo "========================================"
