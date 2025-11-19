#!/bin/bash
# Quick Test Status Check

echo "========================================"
echo "3D Diffusion Model - Test Status"
echo "========================================"
echo ""

# Check Python
# Try to find the virtual environment python first
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

python_version=$($PYTHON_CMD --version 2>&1)
echo "✓ Python: $python_version ($PYTHON_CMD)"

# Check pip
pip_version=$($PYTHON_CMD -m pip --version | head -1)
echo "✓ pip: $pip_version"

# Check pytest
if $PYTHON_CMD -c "import pytest" 2>/dev/null; then
    pytest_version=$($PYTHON_CMD -c "import pytest; print(pytest.__version__)")
    echo "✓ pytest: $pytest_version"
else
    echo "✗ pytest: Not installed"
    echo "  Install with: $PYTHON_CMD -m pip install pytest pytest-cov"
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
    if ! $PYTHON_CMD -m py_compile "$file" 2>/dev/null; then
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
if $PYTHON_CMD -c "import torch" 2>/dev/null; then
    torch_version=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch: $torch_version"
    torch_installed=true
else
    echo "✗ PyTorch: Not installed"
    echo "  Install with: $PYTHON_CMD -m pip install torch"
    torch_installed=false
fi

# Check for PyTorch Geometric
if $PYTHON_CMD -c "import torch_geometric" 2>/dev/null; then
    pyg_version=$($PYTHON_CMD -c "import torch_geometric; print(torch_geometric.__version__)")
    echo "✓ PyTorch Geometric: $pyg_version"
else
    echo "✗ PyTorch Geometric: Not installed"
    echo "  Install with: $PYTHON_CMD -m pip install torch-geometric"
fi

# Check for other dependencies
for pkg in numpy scipy matplotlib yaml; do
    # Handle pyyaml import name
    import_name=$pkg
    if [ "$pkg" == "yaml" ]; then
        import_name="yaml"
    elif [ "$pkg" == "pyyaml" ]; then
        import_name="yaml"
    fi
    
    if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        version=$($PYTHON_CMD -c "import $import_name; print($import_name.__version__ if hasattr($import_name, '__version__') else 'installed')" 2>/dev/null || echo "installed")
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
