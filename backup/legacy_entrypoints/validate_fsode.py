"""Deprecated compatibility entry point for FNODE validation."""
import runpy


if __name__ == "__main__":
    runpy.run_path("validate.py", run_name="__main__")
