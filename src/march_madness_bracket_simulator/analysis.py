"""Analysis utilities for tournament simulation outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from march_madness_bracket_simulator.simulator import (
    simulate_full_tournament_details_once,
    simulate_full_tournament_once,
)


def simulate_tournament_champions(
    east_round1_ids: pd.DataFrame,
    west_round1_ids: pd.DataFrame,
    south_round1_ids: pd.DataFrame,
    midwest_round1_ids: pd.DataFrame,
    features_2026: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    second_round_pairs: Sequence[tuple[int, int]],
    n_simulations: int = 1000,
    random_seed: int | None = None,
) -> pd.Series:
    """Run repeated full-tournament simulations and return champion counts."""
    rng = np.random.default_rng(random_seed)
    champions: list[str] = []

    for _ in range(n_simulations):
        champions.append(
            simulate_full_tournament_once(
                east_round1_ids,
                west_round1_ids,
                south_round1_ids,
                midwest_round1_ids,
                features_2026,
                feature_cols,
                model,
                second_round_pairs,
                rng=rng,
            )
        )

    return pd.Series(champions, name="champion").value_counts()


def summarize_champion_odds(champion_counts: pd.Series, top_n: int | None = None) -> pd.DataFrame:
    """Convert champion counts into a sorted odds table."""
    total = champion_counts.sum()
    odds = (
        champion_counts.rename_axis("team")
        .reset_index(name="titles_won")
        .assign(championship_odds_pct=lambda df: (df["titles_won"] / total * 100).round(1))
    )

    if top_n is not None:
        odds = odds.head(top_n)

    return odds.reset_index(drop=True)


def simulate_consensus_bracket(
    east_round1_ids: pd.DataFrame,
    west_round1_ids: pd.DataFrame,
    south_round1_ids: pd.DataFrame,
    midwest_round1_ids: pd.DataFrame,
    features_2026: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    second_round_pairs: Sequence[tuple[int, int]],
    n_simulations: int = 1000,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Aggregate repeated simulations into a slot-by-slot consensus bracket."""
    rng = np.random.default_rng(random_seed)
    region_rounds: dict[str, dict[str, list[pd.DataFrame]]] = {
        region: {"round1": [], "round2": [], "round3": [], "final": []}
        for region in ["East", "West", "South", "Midwest"]
    }
    final_four_results: list[pd.DataFrame] = []
    championship_results: list[pd.DataFrame] = []

    for _ in range(n_simulations):
        simulation = simulate_full_tournament_details_once(
            east_round1_ids,
            west_round1_ids,
            south_round1_ids,
            midwest_round1_ids,
            features_2026,
            feature_cols,
            model,
            second_round_pairs,
            rng=rng,
        )
        for region_name, region_data in simulation["regions"].items():
            for round_name in ["round1", "round2", "round3", "final"]:
                region_rounds[region_name][round_name].append(region_data[round_name].copy())
        final_four_results.append(simulation["final_four"].copy())
        championship_results.append(simulation["championship"].copy())

    def _consensus_round(round_frames: list[pd.DataFrame], winner_col: str = "simulated_winner") -> pd.DataFrame:
        template = round_frames[0].copy()
        winners_by_slot = pd.DataFrame(
            {idx: frame[winner_col].reset_index(drop=True) for idx, frame in enumerate(round_frames)}
        ).T
        consensus_rows = []

        for slot_idx in range(template.shape[0]):
            counts = winners_by_slot[slot_idx].value_counts()
            winner = counts.index[0]
            share = counts.iloc[0] / len(round_frames)
            row = template.iloc[slot_idx].copy()
            row["consensus_winner"] = winner
            row["consensus_share"] = share

            if "winner_seed" in template.columns:
                matching = [frame.iloc[slot_idx] for frame in round_frames if frame.iloc[slot_idx][winner_col] == winner]
                if matching:
                    row["consensus_seed"] = matching[0].get("winner_seed", np.nan)
                else:
                    row["consensus_seed"] = np.nan
            else:
                row["consensus_seed"] = np.nan

            consensus_rows.append(row)

        return pd.DataFrame(consensus_rows)

    regions = {
        region: {
            round_name: _consensus_round(round_frames)
            for round_name, round_frames in rounds.items()
        }
        for region, rounds in region_rounds.items()
    }
    final_four = _consensus_round(final_four_results)
    championship = _consensus_round(championship_results)

    return {
        "regions": regions,
        "final_four": final_four,
        "championship": championship,
        "champion": championship.iloc[0]["consensus_winner"],
        "champion_share": championship.iloc[0]["consensus_share"],
    }


def simulate_tournament_summary(
    east_round1_ids: pd.DataFrame,
    west_round1_ids: pd.DataFrame,
    south_round1_ids: pd.DataFrame,
    midwest_round1_ids: pd.DataFrame,
    features_2026: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    second_round_pairs: Sequence[tuple[int, int]],
    n_simulations: int = 1000,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Run repeated simulations and collect champion, regional, and Final Four counts."""
    rng = np.random.default_rng(random_seed)
    champion_counts: dict[str, int] = {}
    final_four_counts: dict[str, int] = {}
    regional_counts: dict[str, dict[str, int]] = {
        region: {} for region in ["East", "West", "South", "Midwest"]
    }

    for _ in range(n_simulations):
        simulation = simulate_full_tournament_details_once(
            east_round1_ids,
            west_round1_ids,
            south_round1_ids,
            midwest_round1_ids,
            features_2026,
            feature_cols,
            model,
            second_round_pairs,
            rng=rng,
        )

        champion = simulation["champion"]
        champion_counts[champion] = champion_counts.get(champion, 0) + 1

        for region_name, region_data in simulation["regions"].items():
            region_champ = region_data["champion"]
            regional_counts[region_name][region_champ] = regional_counts[region_name].get(region_champ, 0) + 1
            final_four_counts[region_champ] = final_four_counts.get(region_champ, 0) + 1

    return {
        "n_simulations": n_simulations,
        "champion_counts": pd.Series(champion_counts).sort_values(ascending=False),
        "final_four_counts": pd.Series(final_four_counts).sort_values(ascending=False),
        "regional_counts": {
            region: pd.Series(counts).sort_values(ascending=False)
            for region, counts in regional_counts.items()
        },
    }


def summarize_round_odds(
    counts: pd.Series,
    simulations: int,
    odds_col_name: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Convert repeated simulation counts into a sorted odds table."""
    odds = (
        counts.rename_axis("team")
        .reset_index(name="appearances")
        .assign(**{odds_col_name: lambda df: (df["appearances"] / simulations * 100).round(1)})
    )
    if top_n is not None:
        odds = odds.head(top_n)
    return odds.reset_index(drop=True)


def _frame_to_payload(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": frame.columns.tolist(),
        "rows": frame.to_dict(orient="records"),
    }


def _frame_from_payload(payload: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(payload["rows"], columns=payload["columns"])


def save_simulation_outputs(
    output_dir: str | Path,
    *,
    champion_odds: pd.DataFrame,
    final_four_odds: pd.DataFrame,
    regional_odds: dict[str, pd.DataFrame],
    consensus_bracket: dict[str, object],
    n_simulations: int,
    random_seed: int | None,
) -> None:
    """Persist app-facing simulation outputs for fast reloads."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    champion_odds.to_csv(output_path / "champion_odds.csv", index=False)
    final_four_odds.to_csv(output_path / "final_four_odds.csv", index=False)

    for region, df in regional_odds.items():
        region_slug = region.lower().replace(" ", "_")
        df.to_csv(output_path / f"regional_odds_{region_slug}.csv", index=False)

    consensus_payload = {
        "n_simulations": n_simulations,
        "random_seed": random_seed,
        "champion": consensus_bracket["champion"],
        "champion_share": consensus_bracket["champion_share"],
        "final_four": _frame_to_payload(consensus_bracket["final_four"]),
        "championship": _frame_to_payload(consensus_bracket["championship"]),
        "regions": {
            region: {
                round_name: _frame_to_payload(frame)
                for round_name, frame in rounds.items()
            }
            for region, rounds in consensus_bracket["regions"].items()
        },
    }

    (output_path / "consensus_bracket.json").write_text(
        json.dumps(consensus_payload, indent=2),
        encoding="utf-8",
    )


def load_simulation_outputs(output_dir: str | Path) -> dict[str, object] | None:
    """Load persisted simulation outputs if they exist."""
    output_path = Path(output_dir)
    champion_path = output_path / "champion_odds.csv"
    final_four_path = output_path / "final_four_odds.csv"
    consensus_path = output_path / "consensus_bracket.json"
    region_paths = {
        region: output_path / f"regional_odds_{region.lower().replace(' ', '_')}.csv"
        for region in ["East", "West", "South", "Midwest"]
    }

    required_paths = [champion_path, final_four_path, consensus_path, *region_paths.values()]
    if not all(path.exists() for path in required_paths):
        return None

    consensus_payload = json.loads(consensus_path.read_text(encoding="utf-8"))
    consensus_bracket = {
        "champion": consensus_payload["champion"],
        "champion_share": consensus_payload["champion_share"],
        "final_four": _frame_from_payload(consensus_payload["final_four"]),
        "championship": _frame_from_payload(consensus_payload["championship"]),
        "regions": {
            region: {
                round_name: _frame_from_payload(frame_payload)
                for round_name, frame_payload in rounds.items()
            }
            for region, rounds in consensus_payload["regions"].items()
        },
    }

    return {
        "n_simulations": consensus_payload.get("n_simulations"),
        "random_seed": consensus_payload.get("random_seed"),
        "champion_odds": pd.read_csv(champion_path),
        "final_four_odds": pd.read_csv(final_four_path),
        "regional_odds": {
            region: pd.read_csv(path)
            for region, path in region_paths.items()
        },
        "consensus_bracket": consensus_bracket,
    }
