#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the Zensical site and add its agent-facing raw Markdown routes."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import agent_docs

ANALYTICS_PROPERTY_RE = re.compile(
    r'(\[project\.extra\.analytics\][\s\S]*?\bproperty\s*=\s*)"[^"]*"'
)


def inject_analytics_property(config_text: str, measurement_id: str) -> str:
    """Return config_text with project.extra.analytics.property set to measurement_id."""
    if not ANALYTICS_PROPERTY_RE.search(config_text):
        raise ValueError("Could not locate [project.extra.analytics] property in zensical.toml")
    return ANALYTICS_PROPERTY_RE.sub(rf'\g<1>"{measurement_id}"', config_text, count=1)


@contextmanager
def build_config(measurement_id: str) -> Iterator[Path | None]:
    """Yield a temporary Zensical config path with GA4 measurement ID injected if set."""
    if not measurement_id:
        yield None
        return

    source_config = agent_docs.REPO_ROOT / "zensical.toml"
    temp_config = agent_docs.REPO_ROOT / ".zensical.build.toml"
    content = source_config.read_text(encoding="utf-8")
    temp_config.write_text(inject_analytics_property(content, measurement_id), encoding="utf-8")
    try:
        yield temp_config
    finally:
        temp_config.unlink(missing_ok=True)


def main() -> int:
    agent_docs.write_generated()
    measurement_id = os.environ.get("GA_MEASUREMENT_ID", "").strip()
    args = sys.argv[1:]
    with build_config(measurement_id) as temp_config:
        if temp_config and "-f" not in args and "--config-file" not in args:
            args = ["-f", str(temp_config), *args]
        command = [sys.executable, "-m", "zensical", "build", *args]
        result = subprocess.run(command, cwd=agent_docs.REPO_ROOT, check=False)
        if result.returncode:
            return result.returncode
    agent_docs.publish(agent_docs.REPO_ROOT / "site")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
