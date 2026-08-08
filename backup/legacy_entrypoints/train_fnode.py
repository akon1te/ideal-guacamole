"""Deprecated compatibility entry point for the public ``fnode`` model."""
import runpy


if __name__ == "__main__":
    runpy.run_path("train.py", run_name="__main__")
