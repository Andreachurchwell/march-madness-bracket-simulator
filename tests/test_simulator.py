import numpy as np
import pandas as pd

from march_madness_bracket_simulator.analysis import (
    simulate_consensus_bracket,
    simulate_tournament_champions,
    summarize_champion_odds,
)
from march_madness_bracket_simulator.simulator import (
    DEFAULT_NAME_MAP,
    simulate_full_tournament_once,
    simulate_region_once,
)


class DummyModel:
    def predict_proba(self, X):
        probs = np.where(X["win_pct_diff"].to_numpy() >= 0, 0.75, 0.25)
        return np.column_stack([1 - probs, probs])


def make_features(team_ids):
    rows = []
    for idx, (team_id, team_name) in enumerate(team_ids, start=1):
        rows.append(
            {
                "Season": 2026,
                "TeamID": team_id,
                "TeamName": team_name,
                "win_pct": 0.5 + idx / 100,
                "avg_points_for": 70 + idx,
                "avg_points_against": 60 + idx / 2,
                "avg_scoring_margin": 10 + idx / 2,
            }
        )
    return pd.DataFrame(rows)


def make_region_round1(prefix, team_ids):
    rows = []
    for i in range(8):
        a_name, a_id = team_ids[2 * i]
        b_name, b_id = team_ids[2 * i + 1]
        rows.append(
            {
                "region": prefix,
                "team_a": a_name,
                "seed_a": i + 1,
                "team_a_id": a_id,
                "team_b": b_name,
                "seed_b": i + 9,
                "team_b_id": b_id,
                "team_a_win_prob": 0.8,
            }
        )
    return pd.DataFrame(rows)


def test_simulate_region_once_returns_valid_team():
    team_ids = [(f"Team{i}", 1000 + i) for i in range(16)]
    features = make_features([(team_id, name) for name, team_id in team_ids])
    round1 = make_region_round1("East", team_ids)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

    winner = simulate_region_once(
        round1,
        features,
        ["win_pct_diff", "points_for_diff", "points_against_diff", "scoring_margin_diff"],
        DummyModel(),
        pairs,
        rng=np.random.default_rng(0),
    )

    assert winner in {name for name, _ in team_ids}


def test_simulate_full_tournament_once_returns_valid_champion():
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    east_ids = [(f"EastTeam{i}", 1000 + i) for i in range(16)]
    west_ids = [(f"WestTeam{i}", 2000 + i) for i in range(16)]
    south_ids = [(f"SouthTeam{i}", 3000 + i) for i in range(16)]
    midwest_ids = [(f"MidwestTeam{i}", 4000 + i) for i in range(16)]

    all_teams = east_ids + west_ids + south_ids + midwest_ids
    features = make_features([(team_id, name) for name, team_id in all_teams])

    east = make_region_round1("East", east_ids)
    west = make_region_round1("West", west_ids)
    south = make_region_round1("South", south_ids)
    midwest = make_region_round1("Midwest", midwest_ids)

    champion = simulate_full_tournament_once(
        east,
        west,
        south,
        midwest,
        features,
        ["win_pct_diff", "points_for_diff", "points_against_diff", "scoring_margin_diff"],
        DummyModel(),
        pairs,
        name_map=DEFAULT_NAME_MAP,
        rng=np.random.default_rng(1),
    )

    assert champion in {name for name, _ in all_teams}


def test_simulate_tournament_champions_and_summary():
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    east_ids = [(f"EastTeam{i}", 1000 + i) for i in range(16)]
    west_ids = [(f"WestTeam{i}", 2000 + i) for i in range(16)]
    south_ids = [(f"SouthTeam{i}", 3000 + i) for i in range(16)]
    midwest_ids = [(f"MidwestTeam{i}", 4000 + i) for i in range(16)]

    all_teams = east_ids + west_ids + south_ids + midwest_ids
    features = make_features([(team_id, name) for name, team_id in all_teams])

    champion_counts = simulate_tournament_champions(
        make_region_round1("East", east_ids),
        make_region_round1("West", west_ids),
        make_region_round1("South", south_ids),
        make_region_round1("Midwest", midwest_ids),
        features,
        ["win_pct_diff", "points_for_diff", "points_against_diff", "scoring_margin_diff"],
        DummyModel(),
        pairs,
        n_simulations=25,
        random_seed=7,
    )

    odds = summarize_champion_odds(champion_counts, top_n=5)

    assert champion_counts.sum() == 25
    assert set(champion_counts.index).issubset({name for name, _ in all_teams})
    assert list(odds.columns) == ["team", "titles_won", "championship_odds_pct"]
    assert len(odds) <= 5


def test_simulate_consensus_bracket_returns_expected_shape():
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    east_ids = [(f"EastTeam{i}", 1000 + i) for i in range(16)]
    west_ids = [(f"WestTeam{i}", 2000 + i) for i in range(16)]
    south_ids = [(f"SouthTeam{i}", 3000 + i) for i in range(16)]
    midwest_ids = [(f"MidwestTeam{i}", 4000 + i) for i in range(16)]
    all_teams = east_ids + west_ids + south_ids + midwest_ids
    features = make_features([(team_id, name) for name, team_id in all_teams])

    consensus = simulate_consensus_bracket(
        make_region_round1("East", east_ids),
        make_region_round1("West", west_ids),
        make_region_round1("South", south_ids),
        make_region_round1("Midwest", midwest_ids),
        features,
        ["win_pct_diff", "points_for_diff", "points_against_diff", "scoring_margin_diff"],
        DummyModel(),
        pairs,
        n_simulations=20,
        random_seed=5,
    )

    assert set(consensus["regions"].keys()) == {"East", "West", "South", "Midwest"}
    assert consensus["regions"]["East"]["round1"].shape[0] == 8
    assert consensus["regions"]["East"]["round2"].shape[0] == 4
    assert consensus["regions"]["East"]["round3"].shape[0] == 2
    assert consensus["regions"]["East"]["final"].shape[0] == 1
    assert 0 <= consensus["champion_share"] <= 1
