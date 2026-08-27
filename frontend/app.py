"""Compatibility launcher for Streamlit deployments that use frontend/app.py."""

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_ROOT / "app.py"), run_name="__main__")
