from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.streamlit_app import (  # noqa: E402
    build_deterministic_bracket_summary,
    build_round1_id_tables,
    build_round1_predictions,
    load_core_march_madness_data,
)
from march_madness_bracket_simulator.evaluation import (  # noqa: E402
    evaluate_bracket_summary,
    load_actual_results,
    validate_results_teams,
)
from march_madness_bracket_simulator.feature_engineering import build_team_season_features  # noqa: E402


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_PATH = RAW_DATA_DIR / "2026_bracket_results.csv"


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Could not find results file: {RESULTS_PATH}")

    data = load_core_march_madness_data()
    team_features = build_team_season_features(data["regular_season"])
    bracket_2026 = data["bracket_2026"].copy()
    bracket_2026["is_play_in"] = bracket_2026["team"].str.contains("/", regex=False)

    _, _, model, bracket_with_features, first_round_2026 = build_round1_predictions(
        data,
        team_features,
        bracket_2026,
    )
    round1_id_tables = build_round1_id_tables(first_round_2026, bracket_with_features)
    latest_season = int(team_features["Season"].max())
    features_latest = team_features[team_features["Season"] == latest_season].copy()
    deterministic_bracket = build_deterministic_bracket_summary(
        round1_id_tables,
        features_latest,
        model,
    )

    actual_results = load_actual_results(RESULTS_PATH)
    bracket_teams = set(bracket_2026["team"].dropna())
    unknown_in_results, missing_from_results = validate_results_teams(bracket_teams, actual_results)

    if unknown_in_results or missing_from_results:
        print("Team-name validation found issues.")
        if unknown_in_results:
            print("\nTeams in 2026_bracket_results.csv that do not match bracket_2026.csv:")
            for team in unknown_in_results:
                print(f"- {team}")
        if missing_from_results:
            print("\nTeams in bracket_2026.csv that never appear in 2026_bracket_results.csv:")
            for team in missing_from_results:
                print(f"- {team}")
        print("\nPlease fix the team names in data/raw/2026_bracket_results.csv before trusting the evaluation.")
        return

    evaluation = evaluate_bracket_summary(
        deterministic_bracket,
        actual_results,
        winner_field="predicted_winner",
        champion_field="champion",
    )
    missing_actual = evaluation["missing_actual"]
    if not missing_actual.empty:
        print("\nSome predicted games could not be matched to actual results:")
        print(missing_actual[["round", "region", "slot_index", "team1_predicted", "team2_predicted"]].to_string(index=False))
        return

    print("\nBaseline deterministic bracket evaluation")
    print("----------------------------------------")
    print(evaluation["round_summary"].to_string(index=False))
    print(f"\nOverall correct picks: {evaluation['overall_correct']} / {evaluation['overall_games']}")
    print(f"Overall accuracy: {evaluation['overall_accuracy']:.3f}")
    print(f"Predicted champion: {evaluation['predicted_champion']}")
    print(f"Actual champion: {evaluation['actual_champion']}")
    print(f"Champion correct: {evaluation['champion_correct']}")

    print("\nMissed games")
    print("-----------")
    missed = evaluation["missed_games"]
    if missed.empty:
        print("None. The baseline got every game right.")
    else:
        print(missed.to_string(index=False))


if __name__ == "__main__":
    main()
