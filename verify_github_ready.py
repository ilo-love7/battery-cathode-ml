#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" GitHub


"""

import os
from pathlib import Path
from typing import List, Tuple

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    ".pre-commit-config.yaml",
]

REQUIRED_DIRS = [
    "src",
    "tests",
    "examples",
    ".github/ISSUE_TEMPLATE",
    ".github/workflows",
    "data",
    "results",
    "docs",
]

CHECKS = []


def check_file_exists(filepath: str) -> Tuple[bool, str]:
    """"""
    path = Path(filepath)
    if path.exists():
        return True, f" {filepath}"
    return False, f" {filepath} (missing)"


def check_directory_exists(dirpath: str) -> Tuple[bool, str]:
    """"""
    path = Path(dirpath)
    if path.is_dir():
        return True, f" {dirpath}/"
    return False, f" {dirpath}/ (missing)"


def check_python_syntax() -> Tuple[bool, List[str]]:
    """ Python """
    import py_compile
    
    errors = []
    src_dir = Path("src")
    
    if not src_dir.exists():
        return False, ["src directory not found"]
    
    for py_file in src_dir.glob("*.py"):
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"Syntax error in {py_file}: {e}")
    
    if errors:
        return False, errors
    return True, ["All Python files have valid syntax"]


def check_requirements() -> Tuple[bool, str]:
    """ requirements.txt """
    req_file = Path("requirements.txt")
    if not req_file.exists():
        return False, "requirements.txt not found"
    
    with open(req_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    if len(lines) == 0:
        return False, "requirements.txt is empty"
    
    return True, f"Found {len(lines)} dependencies"


def check_git_config() -> Tuple[bool, str]:
    """ .git """
    git_dir = Path(".git")
    if git_dir.exists():
        return True, "Git repository initialized"
    return None, "  Not a git repository yet (run 'git init')"


def run_checks() -> bool:
    """"""
    print("=" * 70)
    print("GitHub Repository Readiness Check")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # 
    print("Checking required files...")
    for filepath in REQUIRED_FILES:
        passed, message = check_file_exists(filepath)
        CHECKS.append((filepath, passed))
        print(f"  {message}")
        if not passed:
            all_passed = False
    print()
    
    # 
    print(" Checking required directories...")
    for dirpath in REQUIRED_DIRS:
        passed, message = check_directory_exists(dirpath)
        CHECKS.append((dirpath, passed))
        print(f"  {message}")
        if not passed:
            all_passed = False
    print()
    
    #  Python 
    print(" Checking Python syntax...")
    passed, messages = check_python_syntax()
    for msg in messages:
        print(f"  {'' if passed else ''} {msg}")
    if not passed:
        all_passed = False
    print()
    
    #  requirements
    print(" Checking requirements.txt...")
    passed, message = check_requirements()
    print(f"  {'' if passed else ''} {message}")
    if not passed:
        all_passed = False
    print()
    
    #  Git 
    print(" Checking Git configuration...")
    status, message = check_git_config()
    if status is not None:
        print(f"  {'' if status else ' '} {message}")
    else:
        print(f"  {message}")
    print()
    
    # 
    print("=" * 70)
    if all_passed:
        print(" All checks passed! Your project is ready for GitHub!")
    else:
        print("  Some checks failed. Please fix the issues above.")
    print("=" * 70)
    print()
    
    # 
    print(" Next Steps:")
    print("  1. Review the code and remove any sensitive information")
    print("  2. Replace YOUR_USERNAME in README.md with your actual username")
    print("  3. Initialize git repository: git init")
    print("  4. Add remote: git remote add origin https://github.com/YOUR_USER/repo.git")
    print("  5. Commit: git add . && git commit -m 'Initial commit'")
    print("  6. Push: git push -u origin main")
    print()
    
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    exit(0 if success else 1)
