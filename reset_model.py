"""
Restore model.py to the canonical baseline specification.
"""

from pathlib import Path


BASELINE_MODEL = Path("baseline_model.py")
LIVE_MODEL = Path("model.py")


def main():
    LIVE_MODEL.write_text(BASELINE_MODEL.read_text())
    print(f"Restored {LIVE_MODEL} from {BASELINE_MODEL}")


if __name__ == "__main__":
    main()
