from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

from march_madness_bracket_simulator.data_loader import load_core_march_madness_data
from march_madness_bracket_simulator.feature_engineering import build_team_season_features


st.set_page_config(
    page_title="March Madness Bracket Simulator",
    page_icon="🏀",
    layout="wide",
)

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
LOGO_PATH = ASSETS_DIR / "logo.webp"
HERO_PATH = ASSETS_DIR / "marchM.png"

FIRST_ROUND_PAIRS = [
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
]

FEATURE_COLS = [
    "win_pct_diff",
    "points_for_diff",
    "points_against_diff",
    "scoring_margin_diff",
]

BRACKET_NAME_MAP = {
    "Ole Miss": "Mississippi",
    "St. John's": "St John's",
    "Michigan St.": "Michigan St",
    "North Dakota St.": "N Dakota St",
    "UConn": "Connecticut",
    "McNeese": "McNeese St",
    "Saint Mary's": "St Mary's CA",
    "Long Island": "LIU Brooklyn",
    "Utah St.": "Utah St",
    "Kennesaw St.": "Kennesaw",
    "Miami (FL)": "Miami FL",
    "Queens (N.C.)": "Queens NC",
    "Saint Louis": "St Louis",
    "Wright St.": "Wright St",
    "Iowa St.": "Iowa St",
    "Tennessee St.": "Tennessee St",
}


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(37, 99, 235, 0.20), transparent 30%),
                    linear-gradient(180deg, #0a0f1a 0%, #121926 100%);
                color: #f5f7fb;
            }
            .hero-card, .panel-card {
                background: rgba(16, 24, 38, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.20);
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.25);
            }
            .hero-title {
                font-size: 2.2rem;
                font-weight: 800;
                letter-spacing: -0.03em;
                color: #f8fafc;
                margin-bottom: 0.2rem;
            }
            .hero-subtitle {
                color: #cbd5e1;
                font-size: 1rem;
                margin-bottom: 0;
            }
            .metric-label {
                color: #94a3b8;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .metric-value {
                color: #eff6ff;
                font-size: 1.8rem;
                font-weight: 700;
            }
            .section-label {
                color: #93c5fd;
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                margin-bottom: 0.4rem;
            }
            .mini-list {
                color: #cbd5e1;
                margin: 0;
                padding-left: 1.1rem;
                line-height: 1.7;
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 14px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str, label: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="section-label">{label}</div>
            <h3 style="margin-top:0;color:#f8fafc;">{title}</h3>
            <p style="color:#cbd5e1; margin-bottom:0;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_bracket_features(
    bracket_2026: pd.DataFrame,
    teams: pd.DataFrame,
    team_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bracket = bracket_2026.copy()
    bracket["team_clean"] = bracket["team"].replace(BRACKET_NAME_MAP)
    merged = bracket.merge(
        teams[["TeamID", "TeamName"]],
        left_on="team_clean",
        right_on="TeamName",
        how="left",
    )

    latest_season = int(team_features["Season"].max())
    features_latest = team_features[team_features["Season"] == latest_season].copy()
    bracket_main = merged[~merged["is_play_in"]].copy()
    bracket_with_features = bracket_main.merge(
        features_latest,
        on="TeamID",
        how="left",
        suffixes=("", "_feature"),
    )
    return merged, bracket_with_features


def build_first_round_matchups(region_df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    rows: list[dict] = []

    for seed_a, seed_b in FIRST_ROUND_PAIRS:
        team_a_df = region_df[region_df["seed"] == seed_a]
        team_b_df = region_df[region_df["seed"] == seed_b]

        if team_a_df.empty or team_b_df.empty:
            continue

        team_a = team_a_df.iloc[0]
        team_b = team_b_df.iloc[0]

        rows.append(
            {
                "region": region_name,
                "seed_a": team_a["seed"],
                "team_a": team_a["team"],
                "seed_b": team_b["seed"],
                "team_b": team_b["team"],
                "seed_diff": team_a["seed"] - team_b["seed"],
                "win_pct_diff": team_a["win_pct"] - team_b["win_pct"],
                "points_for_diff": team_a["avg_points_for"] - team_b["avg_points_for"],
                "points_against_diff": (
                    team_a["avg_points_against"] - team_b["avg_points_against"]
                ),
                "scoring_margin_diff": (
                    team_a["avg_scoring_margin"] - team_b["avg_scoring_margin"]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_historical_matchups(
    tourney_df: pd.DataFrame, team_features_df: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []

    for _, game in tourney_df.iterrows():
        season = game["Season"]
        team_a_id = min(game["WTeamID"], game["LTeamID"])
        team_b_id = max(game["WTeamID"], game["LTeamID"])
        team_a_won = int(team_a_id == game["WTeamID"])

        team_a_features = team_features_df[
            (team_features_df["Season"] == season)
            & (team_features_df["TeamID"] == team_a_id)
        ]
        team_b_features = team_features_df[
            (team_features_df["Season"] == season)
            & (team_features_df["TeamID"] == team_b_id)
        ]

        if team_a_features.empty or team_b_features.empty:
            continue

        team_a_features = team_a_features.iloc[0]
        team_b_features = team_b_features.iloc[0]

        rows.append(
            {
                "Season": season,
                "team_a_id": team_a_id,
                "team_b_id": team_b_id,
                "team_a_won": team_a_won,
                "win_pct_diff": team_a_features["win_pct"] - team_b_features["win_pct"],
                "points_for_diff": (
                    team_a_features["avg_points_for"] - team_b_features["avg_points_for"]
                ),
                "points_against_diff": (
                    team_a_features["avg_points_against"]
                    - team_b_features["avg_points_against"]
                ),
                "scoring_margin_diff": (
                    team_a_features["avg_scoring_margin"]
                    - team_b_features["avg_scoring_margin"]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_round1_predictions(
    data: dict[str, pd.DataFrame],
    team_features: pd.DataFrame,
    bracket_2026: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, bracket_with_features = prepare_bracket_features(
        bracket_2026, data["teams"], team_features
    )

    region_frames = []
    for region_name in ["East", "South", "West", "Midwest"]:
        region_df = bracket_with_features[bracket_with_features["region"] == region_name]
        region_frames.append(build_first_round_matchups(region_df, region_name))

    first_round_2026 = pd.concat(region_frames, ignore_index=True)

    historical_matchups = build_historical_matchups(data["tourney"], team_features)
    X = historical_matchups[FEATURE_COLS]
    y = historical_matchups["team_a_won"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    X_2026 = first_round_2026[FEATURE_COLS]
    first_round_2026["team_a_win_prob"] = model.predict_proba(X_2026)[:, 1]
    first_round_2026["team_b_win_prob"] = 1 - first_round_2026["team_a_win_prob"]
    first_round_2026["predicted_winner"] = first_round_2026.apply(
        lambda row: row["team_a"] if row["team_a_win_prob"] >= 0.5 else row["team_b"],
        axis=1,
    )
    first_round_2026["favorite"] = first_round_2026.apply(
        lambda row: row["team_a"]
        if row["team_a_win_prob"] >= row["team_b_win_prob"]
        else row["team_b"],
        axis=1,
    )
    first_round_2026["favorite_win_prob"] = first_round_2026[
        ["team_a_win_prob", "team_b_win_prob"]
    ].max(axis=1)
    first_round_2026["underdog_pick"] = first_round_2026["predicted_winner"] != first_round_2026[
        "team_a"
    ]

    round1_predictions = first_round_2026[
        [
            "region",
            "team_a",
            "seed_a",
            "team_b",
            "seed_b",
            "team_a_win_prob",
            "team_b_win_prob",
            "predicted_winner",
            "favorite",
            "favorite_win_prob",
            "underdog_pick",
        ]
    ].copy()
    round1_predictions["team_a_win_prob"] = round1_predictions["team_a_win_prob"].round(3)
    round1_predictions["team_b_win_prob"] = round1_predictions["team_b_win_prob"].round(3)
    round1_predictions["favorite_win_prob"] = round1_predictions["favorite_win_prob"].round(3)

    return round1_predictions, historical_matchups


@st.cache_data
def load_app_data():
    data = load_core_march_madness_data()
    team_features = build_team_season_features(data["regular_season"])
    team_features = team_features.merge(
        data["teams"][["TeamID", "TeamName"]],
        on="TeamID",
        how="left",
    )

    bracket_2026 = data["bracket_2026"].copy()
    bracket_2026["is_play_in"] = bracket_2026["team"].str.contains("/", regex=False)

    round1_predictions, historical_matchups = build_round1_predictions(
        data, team_features, bracket_2026
    )

    return data, team_features, bracket_2026, round1_predictions, historical_matchups


def main() -> None:
    apply_styles()
    data, team_features, bracket_2026, round1_predictions, historical_matchups = load_app_data()

    latest_season = int(team_features["Season"].max())
    latest_features = team_features[team_features["Season"] == latest_season].copy()
    latest_features = latest_features.sort_values("avg_scoring_margin", ascending=False)
    play_in_rows = bracket_2026[bracket_2026["is_play_in"]].copy()
    resolved_rows = bracket_2026[~bracket_2026["is_play_in"]].copy()
    _, bracket_with_features = prepare_bracket_features(bracket_2026, data["teams"], team_features)
    top_2026 = bracket_with_features.sort_values("avg_scoring_margin", ascending=False)

    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=170)
        st.markdown("### What This App Shows")
        st.markdown("- Current 2026 bracket input")
        st.markdown("- Baseline round-1 predictions")
        st.markdown("- Closest games and underdog leans")
        st.markdown("- Unresolved play-in rows")

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">Project Shell</div>
            <div class="hero-title">March Madness Bracket Simulator</div>
            <p class="hero-subtitle">
                A simple app view for the project as it exists right now: bracket input,
                baseline model output, and unresolved play-in status.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if HERO_PATH.exists():
        hero_left, hero_center, hero_right = st.columns([0.24, 0.52, 0.24])
        with hero_center:
            st.image(str(HERO_PATH), use_container_width=True)

    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Resolved Teams", f"{resolved_rows.shape[0]}"),
        ("Play-In Rows", f"{play_in_rows.shape[0]}"),
        ("Training Matchups", f"{historical_matchups.shape[0]:,}"),
        ("Round 1 Predictions", f"{round1_predictions.shape[0]}"),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4], metrics):
        col.markdown(
            f"""
            <div class="panel-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    top_left, top_right = st.columns([1.15, 1])

    with top_left:
        panel(
            "2026 Bracket Input",
            "Current Focus",
            "This is the current tournament field being fed into the project. Play-in "
            "rows are still separated because they do not represent finalized teams yet.",
        )
        st.dataframe(
            bracket_2026[["region", "seed", "team", "is_play_in"]],
            use_container_width=True,
            hide_index=True,
        )

    with top_right:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-label">Baseline Model</div>
                <h3 style="margin-top:0;color:#f8fafc;">Current Feature Set</h3>
                <ul class="mini-list">
                    <li><code>win_pct_diff</code></li>
                    <li><code>points_for_diff</code></li>
                    <li><code>points_against_diff</code></li>
                    <li><code>scoring_margin_diff</code></li>
                </ul>
                <p style="color:#cbd5e1; margin-bottom:0;">
                    These are the matchup-difference features currently used in the
                    baseline logistic regression model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        panel(
            "What This Means",
            "App Context",
            "This app is not the full simulator yet. Right now it is mainly a clearer "
            "way to see the current bracket input, the baseline model output, and "
            "which pieces still need to be finished.",
        )

    st.write("")

    predictions_left, predictions_right = st.columns([1.3, 1])

    with predictions_left:
        panel(
            "2026 Round 1 Baseline Predictions",
            "Current Output",
            "These are the non-play-in first-round predictions generated by the same "
            "baseline logistic regression workflow you built in the notebook.",
        )
        st.dataframe(round1_predictions, use_container_width=True, hide_index=True)

    with predictions_right:
        panel(
            "Closest First-Round Games",
            "Upset Watch",
            "These are the least certain round-1 games by favorite win probability. "
            "They are the natural places to look first for toss-ups and upset cases.",
        )
        st.dataframe(
            round1_predictions.sort_values("favorite_win_prob").head(8),
            use_container_width=True,
            hide_index=True,
        )

        st.write("")
        panel(
            "Underdog-Style Picks",
            "Model Leans",
            "These are the games where the model currently picks the team on the "
            "right side of the matchup row.",
        )
        st.dataframe(
            round1_predictions[round1_predictions["underdog_pick"]],
            use_container_width=True,
            hide_index=True,
        )

    st.write("")

    bottom_left, bottom_right = st.columns([1.2, 1])

    with bottom_left:
        panel(
            "Current 2026 Teams by Scoring Margin",
            "Quick Read",
            "A simple latest-season view of bracket teams sorted by scoring margin. "
            "This is a lightweight way to spot strong profiles before full simulation.",
        )
        st.dataframe(
            top_2026[
                [
                    "team",
                    "seed",
                    "wins",
                    "losses",
                    "win_pct",
                    "avg_points_for",
                    "avg_points_against",
                    "avg_scoring_margin",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with bottom_right:
        panel(
            "Unresolved Play-In Rows",
            "Still Pending",
            "These rows still need explicit play-in handling before the full 2026 "
            "bracket can be advanced round by round in the app.",
        )
        st.dataframe(
            play_in_rows[["region", "seed", "team"]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
