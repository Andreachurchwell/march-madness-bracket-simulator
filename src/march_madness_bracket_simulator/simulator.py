"""Bracket simulation logic."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_NAME_MAP = {
    "St. John's": "St John's",
    "Michigan St.": "Michigan St",
    "North Dakota St.": "N Dakota St",
    "UConn": "Connecticut",
    "Ole Miss": "Mississippi",
    "Saint Mary's": "St Mary's CA",
    "Utah St.": "Utah St",
    "Miami (FL)": "Miami FL",
    "Queens (N.C.)": "Queens NC",
    "Saint Louis": "St Louis",
    "Iowa St.": "Iowa St",
    "Tennessee St.": "Tennessee St",
    "Wright St.": "Wright St",
    "Kennesaw St.": "Kennesaw",
    "McNeese": "McNeese St",
    "Long Island": "LIU Brooklyn",
    "PVAMU": "Prairie View",
}


def _sample_winner(team_a: str, team_b: str, team_a_win_prob: float, rng: np.random.Generator) -> str:
    return team_a if rng.random() < team_a_win_prob else team_b


def _attach_team_features(
    matchup_df: pd.DataFrame,
    features_2026: pd.DataFrame,
) -> pd.DataFrame:
    matchup_df = matchup_df.merge(
        features_2026.add_prefix("team_a_"),
        left_on="team_a_id",
        right_on="team_a_TeamID",
        how="left",
    )
    matchup_df = matchup_df.merge(
        features_2026.add_prefix("team_b_"),
        left_on="team_b_id",
        right_on="team_b_TeamID",
        how="left",
    )
    return matchup_df


def _compute_diff_features(matchup_df: pd.DataFrame) -> pd.DataFrame:
    matchup_df["win_pct_diff"] = matchup_df["team_a_win_pct"] - matchup_df["team_b_win_pct"]
    matchup_df["points_for_diff"] = matchup_df["team_a_avg_points_for"] - matchup_df["team_b_avg_points_for"]
    matchup_df["points_against_diff"] = (
        matchup_df["team_a_avg_points_against"] - matchup_df["team_b_avg_points_against"]
    )
    matchup_df["scoring_margin_diff"] = (
        matchup_df["team_a_avg_scoring_margin"] - matchup_df["team_b_avg_scoring_margin"]
    )
    return matchup_df


def _simulate_matchup_round(
    matchup_df: pd.DataFrame,
    features_2026: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    rng: np.random.Generator,
) -> pd.DataFrame:
    matchup_df = _attach_team_features(matchup_df.copy(), features_2026)
    matchup_df = _compute_diff_features(matchup_df)

    X = matchup_df[list(feature_cols)]
    matchup_df["team_a_win_prob"] = model.predict_proba(X)[:, 1]
    matchup_df["simulated_winner"] = matchup_df.apply(
        lambda row: _sample_winner(row["team_a"], row["team_b"], row["team_a_win_prob"], rng),
        axis=1,
    )
    if {"seed_a", "seed_b"}.issubset(matchup_df.columns):
        matchup_df["winner_seed"] = matchup_df.apply(
            lambda row: row["seed_a"] if row["simulated_winner"] == row["team_a"] else row["seed_b"],
            axis=1,
        )
    else:
        matchup_df["winner_seed"] = np.nan
    matchup_df["winner_team_id"] = matchup_df.apply(
        lambda row: row["team_a_id"] if row["simulated_winner"] == row["team_a"] else row["team_b_id"],
        axis=1,
    )
    return matchup_df


def simulate_region_once(
    region_round1_ids: pd.DataFrame,
    features_2026: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    second_round_pairs: Sequence[tuple[int, int]],
    rng: np.random.Generator | None = None,
) -> str:
    """Simulate one full region and return the simulated regional champion."""
    return simulate_region_details_once(
        region_round1_ids,
        features_2026,
        feature_cols,
        model,
        second_round_pairs,
        rng=rng,
    )["champion"]


def simulate_region_details_once(
    region_round1_ids: pd.DataFrame,
    features_2026: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    second_round_pairs: Sequence[tuple[int, int]],
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    """Simulate one full region and return round-by-round results."""
    rng = rng or np.random.default_rng()

    round1 = region_round1_ids.copy()
    round1["simulated_winner"] = round1.apply(
        lambda row: _sample_winner(row["team_a"], row["team_b"], row["team_a_win_prob"], rng),
        axis=1,
    )
    round1["winner_seed"] = round1.apply(
        lambda row: row["seed_a"] if row["simulated_winner"] == row["team_a"] else row["seed_b"],
        axis=1,
    )
    round1["winner_team_id"] = round1.apply(
        lambda row: row["team_a_id"] if row["simulated_winner"] == row["team_a"] else row["team_b_id"],
        axis=1,
    )

    round2_rows = []
    for idx_a, idx_b in second_round_pairs:
        game_a = round1.iloc[idx_a]
        game_b = round1.iloc[idx_b]
        round2_rows.append(
            {
                "team_a": game_a["simulated_winner"],
                "seed_a": game_a["winner_seed"],
                "team_a_id": game_a["winner_team_id"],
                "team_b": game_b["simulated_winner"],
                "seed_b": game_b["winner_seed"],
                "team_b_id": game_b["winner_team_id"],
            }
        )

    round2 = _simulate_matchup_round(pd.DataFrame(round2_rows), features_2026, feature_cols, model, rng)

    round3_rows = [
        {
            "team_a": round2.iloc[0]["simulated_winner"],
            "seed_a": round2.iloc[0]["winner_seed"],
            "team_a_id": round2.iloc[0]["winner_team_id"],
            "team_b": round2.iloc[1]["simulated_winner"],
            "seed_b": round2.iloc[1]["winner_seed"],
            "team_b_id": round2.iloc[1]["winner_team_id"],
        },
        {
            "team_a": round2.iloc[2]["simulated_winner"],
            "seed_a": round2.iloc[2]["winner_seed"],
            "team_a_id": round2.iloc[2]["winner_team_id"],
            "team_b": round2.iloc[3]["simulated_winner"],
            "seed_b": round2.iloc[3]["winner_seed"],
            "team_b_id": round2.iloc[3]["winner_team_id"],
        },
    ]
    round3 = _simulate_matchup_round(pd.DataFrame(round3_rows), features_2026, feature_cols, model, rng)

    final_rows = [
        {
            "team_a": round3.iloc[0]["simulated_winner"],
            "seed_a": round3.iloc[0]["winner_seed"],
            "team_a_id": round3.iloc[0]["winner_team_id"],
            "team_b": round3.iloc[1]["simulated_winner"],
            "seed_b": round3.iloc[1]["winner_seed"],
            "team_b_id": round3.iloc[1]["winner_team_id"],
        }
    ]
    final = _simulate_matchup_round(pd.DataFrame(final_rows), features_2026, feature_cols, model, rng)
    return {
        "round1": round1,
        "round2": round2,
        "round3": round3,
        "final": final,
        "champion": final.iloc[0]["simulated_winner"],
        "champion_seed": final.iloc[0]["winner_seed"],
        "champion_team_id": final.iloc[0]["winner_team_id"],
    }


def simulate_full_tournament_once(
    east_round1_ids: pd.DataFrame,
    west_round1_ids: pd.DataFrame,
    south_round1_ids: pd.DataFrame,
    midwest_round1_ids: pd.DataFrame,
    features_2026: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    second_round_pairs: Sequence[tuple[int, int]],
    name_map: dict[str, str] | None = None,
    rng: np.random.Generator | None = None,
) -> str:
    """Simulate one full tournament and return the simulated national champion."""
    return simulate_full_tournament_details_once(
        east_round1_ids,
        west_round1_ids,
        south_round1_ids,
        midwest_round1_ids,
        features_2026,
        feature_cols,
        model,
        second_round_pairs,
        name_map=name_map,
        rng=rng,
    )["champion"]


def simulate_full_tournament_details_once(
    east_round1_ids: pd.DataFrame,
    west_round1_ids: pd.DataFrame,
    south_round1_ids: pd.DataFrame,
    midwest_round1_ids: pd.DataFrame,
    features_2026: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    second_round_pairs: Sequence[tuple[int, int]],
    name_map: dict[str, str] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    """Simulate one full tournament and return round-level results."""
    rng = rng or np.random.default_rng()
    name_map = name_map or DEFAULT_NAME_MAP

    east = simulate_region_details_once(
        east_round1_ids, features_2026, feature_cols, model, second_round_pairs, rng=rng
    )
    west = simulate_region_details_once(
        west_round1_ids, features_2026, feature_cols, model, second_round_pairs, rng=rng
    )
    south = simulate_region_details_once(
        south_round1_ids, features_2026, feature_cols, model, second_round_pairs, rng=rng
    )
    midwest = simulate_region_details_once(
        midwest_round1_ids, features_2026, feature_cols, model, second_round_pairs, rng=rng
    )
    east_champ = east["champion"]
    west_champ = west["champion"]
    south_champ = south["champion"]
    midwest_champ = midwest["champion"]

    champion_lookup = features_2026[
        ["TeamID", "TeamName", "win_pct", "avg_points_for", "avg_points_against", "avg_scoring_margin"]
    ].copy()

    def _lookup_team(team_name: str) -> pd.Series:
        clean_name = name_map.get(team_name, team_name)
        matches = champion_lookup[champion_lookup["TeamName"] == clean_name]
        if matches.empty:
            raise KeyError(f"Could not find simulated team '{team_name}' as '{clean_name}' in features_2026")
        return matches.iloc[0]

    south_team = _lookup_team(south_champ)
    west_team = _lookup_team(west_champ)
    east_team = _lookup_team(east_champ)
    midwest_team = _lookup_team(midwest_champ)

    final_four = pd.DataFrame(
        [
            {
                "team_a": south_champ,
                "seed_a": south["champion_seed"],
                "team_a_id": south_team["TeamID"],
                "team_b": west_champ,
                "seed_b": west["champion_seed"],
                "team_b_id": west_team["TeamID"],
            },
            {
                "team_a": east_champ,
                "seed_a": east["champion_seed"],
                "team_a_id": east_team["TeamID"],
                "team_b": midwest_champ,
                "seed_b": midwest["champion_seed"],
                "team_b_id": midwest_team["TeamID"],
            },
        ]
    )
    final_four = _simulate_matchup_round(final_four, features_2026, feature_cols, model, rng)

    champ_a = final_four.iloc[0]["simulated_winner"]
    champ_b = final_four.iloc[1]["simulated_winner"]
    champ_a_row = _lookup_team(champ_a)
    champ_b_row = _lookup_team(champ_b)

    championship = pd.DataFrame(
        [
            {
                "team_a": champ_a,
                "seed_a": final_four.iloc[0]["winner_seed"],
                "team_a_id": champ_a_row["TeamID"],
                "team_b": champ_b,
                "seed_b": final_four.iloc[1]["winner_seed"],
                "team_b_id": champ_b_row["TeamID"],
            }
        ]
    )
    championship = _simulate_matchup_round(championship, features_2026, feature_cols, model, rng)
    return {
        "regions": {
            "East": east,
            "West": west,
            "South": south,
            "Midwest": midwest,
        },
        "final_four": final_four,
        "championship": championship,
        "champion": championship.iloc[0]["simulated_winner"],
    }
