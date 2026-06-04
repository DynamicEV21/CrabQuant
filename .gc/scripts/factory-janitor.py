#!/usr/bin/env python3
"""Python wrapper for factory-janitor.sh — Hermes Script field requires Python."""
import subprocess
import sys
import os

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factory-janitor.sh")

def main():
    result = subprocess.run(
        ["bash", SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Output stdout (JSON report) so Hermes injects it into the prompt
    if result.stdout:
        print(result.stdout, end="")
    # If the script failed, exit with error so Hermes knows
    if result.returncode != 0:
        # Print stderr to stderr for debugging
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
