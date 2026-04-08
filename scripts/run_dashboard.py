"""Launch the Streamlit dashboard."""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
APP = ROOT / "src" / "dashboard" / "app.py"
sys.path.insert(0, str(ROOT))

from src.data.loader import ensure_raw_data
from src.mt5 import load_dotenv_file

if __name__ == "__main__":
    load_dotenv_file(ROOT / ".env")
    ensure_raw_data()
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.port", "8501",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT),
    )
