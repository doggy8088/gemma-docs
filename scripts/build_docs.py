#!/usr/bin/env python3
"""Build the Zensical site and add its agent-facing raw Markdown routes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import agent_docs


def main() -> int:
    agent_docs.write_generated()
    command = [sys.executable, "-m", "zensical", "build", *sys.argv[1:]]
    result = subprocess.run(command, cwd=agent_docs.REPO_ROOT, check=False)
    if result.returncode:
        return result.returncode
    agent_docs.publish(agent_docs.REPO_ROOT / "site")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
