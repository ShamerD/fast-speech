from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent
DATA_DIR = ROOT_PATH / "data"
CHECKPOINT_DIR = ROOT_PATH / "resources"
