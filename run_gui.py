
# 启动图形界面

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.app import run_app

if __name__ == "__main__":
    run_app()
