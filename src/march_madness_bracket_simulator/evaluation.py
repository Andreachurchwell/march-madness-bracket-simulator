"""Helpers for comparing predicted 2026 bracket paths against actual results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ACTUAL_TO_BRACKET_NAME_MAP = {
    "Iowa State": "Iowa St.",
    "Kennesaw State": "Kennesaw St.",
    "Miami (OH)": "Miami OH",
    "Michigan State": "Michigan St.",
    "North Dakota State": "North Dakota St.",
    "Ohio State": "Ohio St",
    "Prairie View A&M": "PVAMU",
    "Queens (NC)": "Queens (N.C.)",
    "St John's": "St. John's",
    "Tennessee State": "Tennessee St.",
    "Utah State": "Utah St.",
    "Wright State": "Wright St.",
}

ROUND_NAME_MAP = {
    "round1": "First Round",
    "round2": "Second Round",
    "round3": "Sweet 16",
    "final": "Elite 8",
}


def normalize_team_name(team_name: str) -> str:
    return ACTUAL_TO_BRACKET_NAME_MAP.get(team_name, team_name)


def load_actual_results(results_path: str | Path) -> pd.DataFrame:
    results_df = pd.read_csv(results_path)
    return normalize_results(results_df)


def normalize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    normalized = results_df.copy()
    for col in ["team1", "team2", "winner"]:
        normalized[col] = normalized[col].map(normalize_team_name)
    normalized["region"] = normalized["region"].fillna("")
    normalized["slot_index"] = normalized.groupby(["round", "region"]).cumcount()
    return normalized


def flatten_bracket_summary(
    bracket_summary: dict[str, object],
    *,
    winner_field: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for region_name, region_data in bracket_summary["regions"].items():
        for round_key, round_name in ROUND_NAME_MAP.items():
            round_df = region_data[round_key]
            for slot_index, (_, game) in enumerate(round_df.iterrows()):
                rows.append(
                    {
                        "round": round_name,
                        "region": region_name,
                        "slot_index": slot_index,
                        "team1": game["team_a"],
                        "team2": game["team_b"],
                        "predicted_winner": game[winner_field],
                    }
                )

    for slot_index, (_, game) in enumerate(bracket_summary["final_four"].iterrows()):
        rows.append(
            {
                "round": "Final Four",
                "region": "",
                "slot_index": slot_index,
                "team1": game["team_a"],
                "team2": game["team_b"],
                "predicted_winner": game[winner_field],
            }
        )

    championship = bracket_summary["championship"].iloc[0]
    rows.append(
        {
            "round": "Championship",
            "region": "",
            "slot_index": 0,
            "team1": championship["team_a"],
            "team2": championship["team_b"],
            "predicted_winner": championship[winner_field],
        }
    )
    return pd.DataFrame(rows)


def validate_results_teams(
    bracket_teams: set[str],
    results_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    results_teams = set(pd.concat([results_df["team1"], results_df["team2"], results_df["winner"]]).dropna())
    unknown_in_results = sorted(results_teams - bracket_teams)
    missing_from_results = sorted(bracket_teams - results_teams)
    return unknown_in_results, missing_from_results


def evaluate_bracket_summary(
    bracket_summary: dict[str, object],
    actual_results: pd.DataFrame,
    *,
    winner_field: str,
    champion_field: str,
) -> dict[str, object]:
    predicted_games = flatten_bracket_summary(bracket_summary, winner_field=winner_field)
    comparison = predicted_games.merge(
        actual_results[["round", "region", "slot_index", "winner", "team1", "team2"]],
        on=["round", "region", "slot_index"],
        how="left",
        validate="one_to_one",
        suffixes=("_predicted", "_actual"),
    )
    comparison["correct"] = comparison["predicted_winner"] == comparison["winner"]

    missing_actual = comparison[comparison["winner"].isna()].copy()
    round_summary = (
        comparison.groupby("round", sort=False)["correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "correct_picks", "count": "games"})
        .reset_index()
    )
    round_summary["accuracy"] = (round_summary["correct_picks"] / round_summary["games"]).round(3)

    actual_champion = actual_results[actual_results["round"] == "Championship"]["winner"].iloc[0]
    predicted_champion = bracket_summary[champion_field]

    missed_games = comparison.loc[
        ~comparison["correct"],
        [
            "round",
            "region",
            "team1_predicted",
            "team2_predicted",
            "team1_actual",
            "team2_actual",
            "predicted_winner",
            "winner",
        ],
    ].reset_index(drop=True)

    return {
        "comparison": comparison,
        "missing_actual": missing_actual,
        "round_summary": round_summary,
        "missed_games": missed_games,
        "overall_correct": int(comparison["correct"].sum()),
        "overall_games": int(len(comparison)),
        "overall_accuracy": float(comparison["correct"].mean()),
        "predicted_champion": predicted_champion,
        "actual_champion": actual_champion,
        "champion_correct": bool(predicted_champion == actual_champion),
    }


def summarize_monte_carlo(
    *,
    consensus_bracket: dict[str, object],
    champion_odds: pd.DataFrame,
    actual_results: pd.DataFrame,
) -> dict[str, object]:
    actual_champion = actual_results[actual_results["round"] == "Championship"]["winner"].iloc[0]
    consensus_eval = evaluate_bracket_summary(
        consensus_bracket,
        actual_results,
        winner_field="consensus_winner",
        champion_field="champion",
    )

    actual_champion_row = champion_odds[champion_odds["team"] == actual_champion].copy()
    champion_rank = None
    champion_odds_pct = None
    if not actual_champion_row.empty:
        champion_rank = int(actual_champion_row.index[0]) + 1
        champion_odds_pct = float(actual_champion_row.iloc[0]["championship_odds_pct"])

    sim_favorite = champion_odds.iloc[0]

    return {
        "consensus_eval": consensus_eval,
        "simulation_favorite": sim_favorite["team"],
        "simulation_favorite_odds_pct": float(sim_favorite["championship_odds_pct"]),
        "actual_champion_rank": champion_rank,
        "actual_champion_odds_pct": champion_odds_pct,
    }
