#!/bin/bash
set -euo pipefail

# Script Description: Run local CI pipeline (fix, format, type-check, test, coverage)
# Author: elvee
# Version: 0.2.0
# License: MIT
# Creation Date: 03/12/2025
# Last Modified: 15/01/2026
# Usage: local-ci.sh

# Constants
DEFAULT_OUTPUT_FILE="${PWD}/ci-output.log"

# ASCII Art (Calvin font)
print_ascii_art() {
  echo "
╦    ╔═╗  ╔═╗  ╔═╗  ╦         ╔═╗  ╦
║    ║ ║  ║    ╠═╣  ║    ───  ║    ║
╩═╝  ╚═╝  ╚═╝  ╩ ╩  ╩═╝       ╚═╝  ╩
"
}

# Help
show_help() {
  echo "
Usage: $0 [OPTIONS]

Options:
  -o, --output_file FILE     Write CI logs to file (default: $DEFAULT_OUTPUT_FILE)
  -h, --help                 Show help

This script performs:
  • uv run ruff check --fix .
  • uv run ruff format .
  • uv run ty check insanely_fast_whisper_rocm
  • uv run pytest -q
  • uv run pytest --cov=insanely_fast_whisper_rocm --cov-report=term-missing:skip-covered --cov-report=xml
  • uv run interrogate . --fail-under=85 -vvvv --style=google
"
}

# Error handling
error_exit() {
  echo "Error: $1" >&2
  exit 1
}

# Main logic
main_logic() {
  echo "[+] The following tasks will be executed:"
  echo "    • uv run ruff check --fix ."
  echo "    • uv run ruff format ."
  echo "    • uv run ty check insanely_fast_whisper_rocm"
  echo "    • uv run pytest -q"
  echo "    • uv run pytest --cov=insanely_fast_whisper_rocm --cov-report=term-missing:skip-covered --cov-report=xml"
  echo "    • uv run interrogate . --fail-under=85 -vvvv --style=google"
  echo ""

  local output_file="$1"

  {
    echo "[+] Running fix..."
    echo ""
    uv run ruff check --fix .
    echo ""
    echo "[+] Running format..."
    uv run ruff format .
    echo ""
    echo "[+] Running type checks..."
    uv run ty check insanely_fast_whisper_rocm
    echo ""
    echo "[+] Running tests..."
    uv run pytest -q
    echo ""
    echo "[+] Running test coverage..."
    uv run pytest --cov=insanely_fast_whisper_rocm --cov-report=term-missing:skip-covered --cov-report=xml
    echo ""
    echo "[+] Running interrogate to check docstring coverage..."
    uv run interrogate . --fail-under=85 -vvvv --style=google
    echo ""
    echo "[+] Local CI check successful. You can commit these changes."
  } | tee "${output_file}"
}

# Main
main() {
  local output_file="$DEFAULT_OUTPUT_FILE"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -o|--output_file)
        output_file="$2"
        shift 2
        ;;
      -h|--help)
        show_help
        exit 0
        ;;
      *)
        error_exit "Invalid option: $1"
        ;;
    esac
  done

  main_logic "$output_file"
}

# Header ASCII art
print_ascii_art

# Execute
main "$@"
