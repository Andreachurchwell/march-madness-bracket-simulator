"""Feature engineering logic for matchup-based team comparisons."""
import pandas as pd


def build_team_season_features(regular_season_results: pd.DataFrame) -> pd.DataFrame:
    wins = (
        regular_season_results.groupby(["Season", "WTeamID"])
        .agg(
            wins=("WTeamID", "size"),
            points_for_win=("WScore", "mean"),
            points_against_win=("LScore", "mean"),
        )
        .reset_index()
        .rename(columns={"WTeamID": "TeamID"})
    )

    losses = (
        regular_season_results.groupby(["Season", "LTeamID"])
        .agg(
            losses=("LTeamID", "size"),
            points_for_loss=("LScore", "mean"),
            points_against_loss=("WScore", "mean"),
        )
        .reset_index()
        .rename(columns={"LTeamID": "TeamID"})
    )

    team_features = wins.merge(losses, on=["Season", "TeamID"], how="outer").fillna(0)

    team_features["games"] = team_features["wins"] + team_features["losses"]
    team_features["win_pct"] = team_features["wins"] / team_features["games"]

    team_features["avg_points_for"] = (
        (team_features["points_for_win"] * team_features["wins"])
        + (team_features["points_for_loss"] * team_features["losses"])
    ) / team_features["games"]

    team_features["avg_points_against"] = (
        (team_features["points_against_win"] * team_features["wins"])
        + (team_features["points_against_loss"] * team_features["losses"])
    ) / team_features["games"]

    team_features["avg_scoring_margin"] = (
        team_features["avg_points_for"] - team_features["avg_points_against"]
    )

    return team_features[
        [
            "Season",
            "TeamID",
            "wins",
            "losses",
            "games",
            "win_pct",
            "avg_points_for",
            "avg_points_against",
            "avg_scoring_margin",
        ]
    ]
