import sys
import pytest

if __name__ == "__main__":
    print(f"Running test suite with Python {sys.version} ...", flush=True)
    exit_code = pytest.main(["-v", "-s", "tests/"])
    print(f"\nPytest finished with exit code: {exit_code}", flush=True)
    sys.exit(exit_code)
