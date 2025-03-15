#!/usr/bin/env python
"""
Setup script for mandatory validation before committing changes.
This script creates a pre-commit hook that runs tests before allowing commits.
"""

import os
import stat

# Define the pre-commit hook content
PRE_COMMIT_HOOK = """#!/bin/sh
#
# Pre-commit hook to run tests before allowing commits
# This ensures validation is mandatory before committing changes

echo "Running pre-commit validation tests..."

# Store the current state to return to it after tests
STASH_NAME="pre-commit-$(date +%s)"
git stash save -q --keep-index $STASH_NAME

# Run the tests
python -m pytest

# Store the test result
RESULT=$?

# Restore the previous state
STASH_LIST=$(git stash list)
if [[ $STASH_LIST == *"$STASH_NAME"* ]]; then
    git stash pop -q
fi

# If tests failed, prevent the commit
if [ $RESULT -ne 0 ]; then
    echo "❌ Tests failed. Commit aborted. Please fix the failing tests before committing."
    exit 1
fi

echo "✅ All tests passed. Proceeding with commit."
exit 0
"""

# Define the advanced pre-commit hook content
ADVANCED_PRE_COMMIT_HOOK = """#!/bin/sh
#
# Pre-commit hook to run tests before allowing commits
# This ensures validation is mandatory before committing changes

# Colors for better output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[0;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

echo "${BLUE}Running pre-commit validation tests...${NC}"

# Store the current state to return to it after tests
STASH_NAME="pre-commit-$(date +%s)"
git stash save -q --keep-index $STASH_NAME

# Check if only specific files have changed
PYTHON_FILES_CHANGED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\\.py$')

# Determine which tests to run
if [ -n "$PYTHON_FILES_CHANGED" ]; then
    echo "${YELLOW}Python files changed:${NC}"
    echo "$PYTHON_FILES_CHANGED"
    
    # Check if any test files were modified
    TEST_FILES_CHANGED=$(echo "$PYTHON_FILES_CHANGED" | grep -E '^tests/')
    
    if [ -n "$TEST_FILES_CHANGED" ]; then
        echo "${YELLOW}Running only modified test files:${NC}"
        echo "$TEST_FILES_CHANGED"
        python -m pytest $TEST_FILES_CHANGED -v
    else
        # Check if any validation-related files were modified
        VALIDATION_FILES_CHANGED=$(echo "$PYTHON_FILES_CHANGED" | grep -E 'validation|metrics')
        
        if [ -n "$VALIDATION_FILES_CHANGED" ]; then
            echo "${YELLOW}Validation files changed, running validation tests:${NC}"
            python -m pytest tests/unit/test_validation_module.py tests/unit/test_metrics.py -v
        else
            # Run all tests by default
            echo "${YELLOW}Running all tests:${NC}"
            python -m pytest
        fi
    fi
else
    # No Python files changed, run a quick test suite
    echo "${YELLOW}No Python files changed, running quick validation:${NC}"
    python -m pytest tests/unit/test_metrics.py -v
fi

# Store the test result
RESULT=$?

# Restore the previous state
STASH_LIST=$(git stash list)
if [[ $STASH_LIST == *"$STASH_NAME"* ]]; then
    git stash pop -q
fi

# If tests failed, prevent the commit
if [ $RESULT -ne 0 ]; then
    echo "${RED}❌ Tests failed. Commit aborted.${NC}"
    echo "${YELLOW}Please fix the failing tests before committing.${NC}"
    exit 1
fi

echo "${GREEN}✅ All tests passed. Proceeding with commit.${NC}"
exit 0
"""

# Define the pre-commit config content
PRE_COMMIT_CONFIG = """# .pre-commit-config.yaml
repos:
-   repo: local
    hooks:
    -   id: pytest
        name: pytest
        entry: python -m pytest
        language: system
        pass_filenames: false
        always_run: true
        
    -   id: pytest-validation
        name: pytest-validation
        entry: python -m pytest tests/unit/test_validation_module.py tests/unit/test_metrics.py -v
        language: system
        files: ^(backtester/validation/|tests/unit/test_validation_module.py|tests/unit/test_metrics.py)
        pass_filenames: false
        
    -   id: flake8
        name: flake8
        entry: flake8
        language: system
        types: [python]
        exclude: ^(venv/|\\.venv/|\\.git/)
"""

def setup_git_hooks():
    """Set up Git hooks for mandatory validation."""
    # Check if .git directory exists
    if not os.path.exists('.git'):
        print("Initializing Git repository...")
        os.system('git init')
    
    # Create hooks directory if it doesn't exist
    hooks_dir = '.git/hooks'
    if not os.path.exists(hooks_dir):
        os.makedirs(hooks_dir)
    
    # Create pre-commit hook
    pre_commit_path = os.path.join(hooks_dir, 'pre-commit')
    with open(pre_commit_path, 'w') as f:
        f.write(ADVANCED_PRE_COMMIT_HOOK)
    
    # Make pre-commit hook executable
    os.chmod(pre_commit_path, os.stat(pre_commit_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    
    print(f"Pre-commit hook created at {pre_commit_path}")
    
    # Create pre-commit config
    with open('.pre-commit-config.yaml', 'w') as f:
        f.write(PRE_COMMIT_CONFIG)
    
    print("Pre-commit config created at .pre-commit-config.yaml")
    
    # Check if pre-commit is installed
    if os.system('which pre-commit > /dev/null 2>&1') != 0:
        print("pre-commit is not installed. You may want to install it with:")
        print("pip install pre-commit")
    else:
        # Install pre-commit hooks
        os.system('pre-commit install')
        print("pre-commit hooks installed")

if __name__ == '__main__':
    setup_git_hooks()
    print("\nMandatory validation before committing is now set up!")
    print("Tests will run automatically before each commit.")
    print("Commits will be blocked if tests fail.") 