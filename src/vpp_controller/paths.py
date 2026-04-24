from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
DATA_NETWORKS_DIR = DATA_DIR / "networks"
DATA_PRICES_DIR = DATA_DIR / "clean_price_data"


OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_PATH = ROOT_DIR / "figures"
