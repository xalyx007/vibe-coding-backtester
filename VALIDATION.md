# Mandatory Validation Process

This document describes the mandatory validation process that is enforced before committing changes to the backtester project.

## Overview

To ensure code quality and prevent regressions, all changes must pass validation tests before they can be committed. This is enforced using Git pre-commit hooks that automatically run tests when you attempt to commit changes.

## How It Works

1. When you run `git commit`, the pre-commit hook is triggered automatically.
2. The hook stashes any unstaged changes to ensure only the changes you're committing are tested.
3. It then runs the appropriate tests based on the files you've modified:
   - If you modified test files, it runs only those test files.
   - If you modified validation-related files, it runs the validation tests.
   - Otherwise, it runs all tests.
4. If the tests pass, your commit proceeds normally.
5. If any tests fail, your commit is blocked, and you'll need to fix the failing tests before you can commit.

## Setting Up the Validation Process

The validation process is set up using the `setup_validation.py` script. This script:

1. Creates a pre-commit hook in the `.git/hooks` directory.
2. Makes the hook executable.
3. Creates a `.pre-commit-config.yaml` file for additional configuration.

To set up the validation process, run:

```bash
python setup_validation.py
```

## Pre-commit Tool (Optional)

For more advanced pre-commit functionality, you can install the `pre-commit` tool:

```bash
pip install pre-commit
pre-commit install
```

This will use the configuration in `.pre-commit-config.yaml` to set up additional hooks.

## Bypassing Validation (Not Recommended)

In rare cases, you may need to bypass the validation process. This should be done only in exceptional circumstances:

```bash
git commit --no-verify
```

**Note:** Bypassing validation is strongly discouraged as it may lead to broken code being committed.

## Troubleshooting

If you encounter issues with the validation process:

1. Make sure your tests are passing locally by running `python -m pytest`.
2. Check that the pre-commit hook is executable: `ls -la .git/hooks/pre-commit`.
3. If the hook isn't working, try reinstalling it: `python setup_validation.py`.

## Best Practices

1. Run tests locally before committing to catch issues early.
2. Keep your commits small and focused to make validation faster.
3. If a test is failing, fix the test or the code, but don't bypass validation.
4. Update tests when you change functionality to ensure validation remains effective. 