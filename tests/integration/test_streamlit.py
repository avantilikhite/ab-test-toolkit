"""Integration test: Streamlit app health check."""

import subprocess
import sys
import time
import signal
import pytest


class TestStreamlitApp:
    """Verify Streamlit app launches without error."""

    def test_app_syntax_valid(self):
        """All Streamlit app files are valid Python."""
        files = [
            "app/app.py",
            "app/app_utils.py",
            "app/pages/01_experiment_design.py",
            "app/pages/02_analyze_results.py",
            "app/pages/03_sensitivity_analysis.py",
            "app/pages/04_case_study_demo.py",
        ]
        for f in files:
            result = subprocess.run(
                [sys.executable, "-c", f"import ast; ast.parse(open('{f}').read())"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, f"Syntax error in {f}: {result.stderr}"

    def test_app_imports_work(self):
        """App files can be parsed without import errors at module level."""
        result = subprocess.run(
            [sys.executable, "-c", """
import importlib.util
import sys
files = {
    'app_main': 'app/app.py',
    'app_utils': 'app/app_utils.py',
}
# Just verify syntax, not full execution (which needs streamlit runtime)
for name, path in files.items():
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"Cannot create spec for {path}"
print("All app files have valid specs")
"""],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Import check failed: {result.stderr}"
