from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.streamlit_app import (
    FEATURE_COLS,
    MONTE_CARLO_SIMULATIONS,
    SECOND_ROUND_PAIRS,
    annotate_champion_odds,
    annotate_team_odds,
    build_round1_id_tables,
    build_round1_predictions,
)
from march_madness_bracket_simulator.analysis import (
    load_simulation_outputs,
    save_simulation_outputs,
    simulate_consensus_bracket,
    simulate_tournament_summary,
    summarize_champion_odds,
    summarize_round_odds,
)
from march_madness_bracket_simulator.data_loader import load_core_march_madness_data
from march_madness_bracket_simulator.feature_engineering import build_team_season_features


OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "simulation_cache"
SIM_RANDOM_SEED = 42


def main() -> None:
    data = load_core_march_madness_data()
    team_features = build_team_season_features(data["regular_season"])
    team_features = team_features.merge(
        data["teams"][["TeamID", "TeamName"]],
        on="TeamID",
        how="left",
    )

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

    consensus_bracket = simulate_consensus_bracket(
        round1_id_tables["East"],
        round1_id_tables["West"],
        round1_id_tables["South"],
        round1_id_tables["Midwest"],
        features_latest,
        FEATURE_COLS,
        model,
        SECOND_ROUND_PAIRS,
        n_simulations=MONTE_CARLO_SIMULATIONS,
        random_seed=SIM_RANDOM_SEED,
    )
    simulation_summary = simulate_tournament_summary(
        round1_id_tables["East"],
        round1_id_tables["West"],
        round1_id_tables["South"],
        round1_id_tables["Midwest"],
        features_latest,
        FEATURE_COLS,
        model,
        SECOND_ROUND_PAIRS,
        n_simulations=MONTE_CARLO_SIMULATIONS,
        random_seed=SIM_RANDOM_SEED,
    )

    champion_odds = annotate_champion_odds(
        summarize_champion_odds(simulation_summary["champion_counts"], top_n=10),
        bracket_2026,
    )
    final_four_odds = annotate_team_odds(
        summarize_round_odds(
            simulation_summary["final_four_counts"],
            simulation_summary["n_simulations"],
            "final_four_odds_pct",
            top_n=10,
        ),
        bracket_2026,
    )
    regional_odds = {
        region: annotate_team_odds(
            summarize_round_odds(
                counts,
                simulation_summary["n_simulations"],
                "regional_win_odds_pct",
                top_n=5,
            ),
            bracket_2026,
        )
        for region, counts in simulation_summary["regional_counts"].items()
    }

    save_simulation_outputs(
        OUTPUT_DIR,
        champion_odds=champion_odds,
        final_four_odds=final_four_odds,
        regional_odds=regional_odds,
        consensus_bracket=consensus_bracket,
        n_simulations=MONTE_CARLO_SIMULATIONS,
        random_seed=SIM_RANDOM_SEED,
    )

    loaded = load_simulation_outputs(OUTPUT_DIR)
    if loaded is None:
        raise RuntimeError("Simulation outputs were not saved correctly.")

    print(f"Saved simulation outputs to {OUTPUT_DIR}")
    print(loaded["champion_odds"][["team", "championship_odds_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
