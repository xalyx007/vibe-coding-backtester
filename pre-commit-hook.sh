#!/bin/sh
#
# Pre-commit hook to run tests before allowing commits
# This ensures validation is mandatory before committing changes

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "${BLUE}Running pre-commit validation tests...${NC}"

# Store the current state to return to it after tests
STASH_NAME="pre-commit-$(date +%s)"
git stash save -q --keep-index $STASH_NAME

# Check if only specific files have changed
PYTHON_FILES_CHANGED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.py$')

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