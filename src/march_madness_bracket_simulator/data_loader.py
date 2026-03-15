"""Data loading utilities for March Madness datasets."""
from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _load_csv(filename: str, data_dir: Path | None = None) -> pd.DataFrame:
    base_dir = Path(data_dir) if data_dir is not None else RAW_DATA_DIR
    path = base_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")

    return pd.read_csv(path)


def load_m_teams(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_csv("MTeams.csv", data_dir)


def load_m_regular_season_results(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_csv("MRegularSeasonCompactResults.csv", data_dir)


def load_m_tourney_results(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_csv("MNCAATourneyCompactResults.csv", data_dir)


def load_m_tourney_seeds(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_csv("MNCAATourneySeeds.csv", data_dir)


def load_m_tourney_slots(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_csv("MNCAATourneySlots.csv", data_dir)


def load_core_march_madness_data(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    return {
        "teams": load_m_teams(data_dir),
        "regular_season": load_m_regular_season_results(data_dir),
        "tourney": load_m_tourney_results(data_dir),
        "seeds": load_m_tourney_seeds(data_dir),
        "slots": load_m_tourney_slots(data_dir),
    }

