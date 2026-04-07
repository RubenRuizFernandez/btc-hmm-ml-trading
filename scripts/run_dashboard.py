"""Launch the Streamlit dashboard."""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
APP = ROOT / "src" / "dashboard" / "app.py"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.port", "8501",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT),
    )
