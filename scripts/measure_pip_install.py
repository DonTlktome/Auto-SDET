"""Time the OLD hard-coded pip command vs the new one, 3 fresh sandboxes each."""
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")

from e2b_code_interpreter import Sandbox  # noqa: E402

from auto_sdet.config import get_settings  # noqa: E402

OLD = "pip install pytest pytest-cov langgraph langchain-openai langchain-core pydantic pydantic-settings"
NEW = "pip install -q --disable-pip-version-check pytest pytest-cov"
key = get_settings().e2b_api_key.get_secret_value()

for label, cmd in [("old", OLD), ("new", NEW)]:
    for i in range(3):
        sbx = Sandbox.create(timeout=300, api_key=key)
        try:
            t0 = time.perf_counter()
            try:
                sbx.commands.run(cmd, timeout=240)
                ok = True
            except Exception as e:  # noqa: BLE001
                ok = f"{type(e).__name__}"
            print(f"{label} run{i + 1}: {time.perf_counter() - t0:.1f}s ok={ok}", flush=True)
        finally:
            sbx.kill()
