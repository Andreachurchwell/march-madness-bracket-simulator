from pathlib import Path

import streamlit as st

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
            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 14px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_app_data():
    data = load_core_march_madness_data()
    team_features = build_team_season_features(data["regular_season"])
    team_features = team_features.merge(
        data["teams"][["TeamID", "TeamName"]],
        on="TeamID",
        how="left",
    )
    team_features = team_features.merge(
        data["seeds"],
        on=["Season", "TeamID"],
        how="left",
    )
    return data, team_features


def main() -> None:
    apply_styles()
    data, team_features = load_app_data()

    latest_season = int(team_features["Season"].max())
    latest_features = team_features[team_features["Season"] == latest_season].copy()
    latest_features = latest_features.sort_values("avg_scoring_margin", ascending=False)

    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=170)
        st.markdown("### Project Areas")
        st.markdown("- Data exploration")
        st.markdown("- Feature engineering")
        st.markdown("- Sleeper-team analysis")
        st.markdown("- Bracket simulation")

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">Project Shell</div>
            <div class="hero-title">March Madness Bracket Simulator</div>
            <p class="hero-subtitle">
                A working project shell for data exploration, team-strength features,
                future matchup prediction, and bracket simulation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if HERO_PATH.exists():
        st.image(str(HERO_PATH), use_container_width=True)

    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Teams", f"{data['teams'].shape[0]}"),
        ("Regular Season Games", f"{data['regular_season'].shape[0]:,}"),
        ("Tournament Games", f"{data['tourney'].shape[0]:,}"),
        ("Latest Season", str(latest_season)),
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

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-label">Current Focus</div>
                <h3 style="margin-top:0;color:#f8fafc;">Team-Season Feature Table</h3>
                <p style="color:#cbd5e1;">
                    This view shows the latest season's team-level features built from
                    regular season results. It is the base layer for later matchup
                    prediction and sleeper-team analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        season_choice = st.selectbox(
            "Season",
            options=sorted(team_features["Season"].unique(), reverse=True),
            index=0,
        )

        display_df = team_features[team_features["Season"] == season_choice].copy()
        display_df = display_df.sort_values("avg_scoring_margin", ascending=False)
        st.dataframe(
            display_df[
                [
                    "TeamName",
                    "Seed",
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

    with right:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-label">Roadmap</div>
                <h3 style="margin-top:0;color:#f8fafc;">What This App Will Hold</h3>
                <ul style="color:#cbd5e1; line-height:1.8;">
                    <li>Core NCAA men’s dataset summaries</li>
                    <li>Season-level feature engineering outputs</li>
                    <li>Matchup comparison tools</li>
                    <li>Sleeper-team indicators</li>
                    <li>Bracket simulation results and upset paths</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        seeded_latest = latest_features[latest_features["Seed"].notna()].copy()
        seeded_latest["seed_num"] = (
            seeded_latest["Seed"].str.extract(r"(\d{2})")[0].astype(int)
        )
        top_sleepers = seeded_latest[
            seeded_latest["seed_num"].ge(8)
        ][["TeamName", "Seed", "avg_scoring_margin", "win_pct"]].head(8)

        st.markdown(
            """
            <div class="panel-card">
                <div class="section-label">Early Signal</div>
                <h3 style="margin-top:0;color:#f8fafc;">Potential Sleeper Candidates</h3>
                <p style="color:#cbd5e1;">
                    This is a simple placeholder view: higher-seeded teams from the
                    latest season sorted by scoring margin.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(top_sleepers, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
