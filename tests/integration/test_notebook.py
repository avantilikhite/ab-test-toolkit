"""Integration test: notebook execution."""

import subprocess
import sys
import pytest


class TestNotebookExecution:
    """Verify case study notebook executes without errors."""

    def test_case_study_notebook_executes(self):
        """case_study.ipynb runs top-to-bottom with zero errors."""
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=120",
                "notebooks/case_study.ipynb",
                "--output", "/tmp/test_executed.ipynb",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, f"Notebook execution failed:\n{result.stderr}"
