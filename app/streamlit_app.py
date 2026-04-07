import base64
from pathlib import Path
from textwrap import dedent

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

from march_madness_bracket_simulator.analysis import (
    load_simulation_outputs,
    simulate_consensus_bracket,
    simulate_tournament_summary,
    summarize_round_odds,
    summarize_champion_odds,
)
from march_madness_bracket_simulator.data_loader import load_core_march_madness_data
from march_madness_bracket_simulator.evaluation import (
    evaluate_bracket_summary,
    load_actual_results,
    summarize_monte_carlo,
    validate_results_teams,
)
from march_madness_bracket_simulator.feature_engineering import build_team_season_features
from march_madness_bracket_simulator.simulator import simulate_full_tournament_once


st.set_page_config(
    page_title="Andrea's Bracket Breakdown",
    page_icon="🏀",
    layout="wide",
)

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
LOGO_PATH = ASSETS_DIR / "logo.webp"
HERO_PATH = ASSETS_DIR / "marchM.png"
SIMULATION_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "simulation_cache"
ACTUAL_RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "2026_bracket_results.csv"
TEAM_LOGO_MAP = {
    "Akron": ASSETS_DIR / "akron.png",
    "Alabama": ASSETS_DIR / "bama.png",
    "Arizona": ASSETS_DIR / "arizona.png",
    "Arkansas": ASSETS_DIR / "arkansas.png",
    "BYU": ASSETS_DIR / "byu.png",
    "Duke": ASSETS_DIR / "duke.png",
    "Florida": ASSETS_DIR / "florida.png",
    "Gonzaga": ASSETS_DIR / "gonzaga.png",
    "High Point": ASSETS_DIR / "highpoint.png",
    "Houston": ASSETS_DIR / "houston.png",
    "Illinois": ASSETS_DIR / "illinois.png",
    "Iowa": ASSETS_DIR / "iowa.png",
    "Iowa St.": ASSETS_DIR / "iowastate.png",
    "Kansas": ASSETS_DIR / "kansas.png",
    "Louisville": ASSETS_DIR / "louisville.png",
    "Miami (FL)": ASSETS_DIR / "miami.png",
    "Miami OH": ASSETS_DIR / "miamioh.png",
    "Michigan": ASSETS_DIR / "michigan.png",
    "Michigan St.": ASSETS_DIR / "mstate.png",
    "Nebraska": ASSETS_DIR / "nebraska.png",
    "Purdue": ASSETS_DIR / "purdue.png",
    "Saint Mary's": ASSETS_DIR / "stmary.png",
    "Saint Louis": ASSETS_DIR / "stlouis.png",
    "Santa Clara": ASSETS_DIR / "santaclara.png",
    "St. John's": ASSETS_DIR / "stjohns.png",
    "TCU": ASSETS_DIR / "tcu.png",
    "UCLA": ASSETS_DIR / "ucla.png",
    "UConn": ASSETS_DIR / "uconn.png",
    "Utah St.": ASSETS_DIR / "utah.png",
    "Vanderbilt": ASSETS_DIR / "vandy.png",
    "VCU": ASSETS_DIR / "vcu.png",
    "Virginia": ASSETS_DIR / "virginia.png",
}

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
SECOND_ROUND_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
MONTE_CARLO_SIMULATIONS = 1000

BRACKET_NAME_MAP = {
    "Ole Miss": "Mississippi",
    "St. John's": "St John's",
    "Michigan St.": "Michigan St",
    "North Dakota St.": "N Dakota St",
    "UConn": "Connecticut",
    "McNeese": "McNeese St",
    "Saint Mary's": "St Mary's CA",
    "Long Island": "LIU Brooklyn",
    "PVAMU": "Prairie View",
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
            .hero-inner {
                display: flex;
                align-items: center;
                gap: 1rem;
                justify-content: space-between;
            }
            .hero-logo {
                width: 132px;
                height: 132px;
                object-fit: contain;
                flex: 0 0 auto;
            }
            .hero-copy {
                flex: 1 1 auto;
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
            .metric-top {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.55rem;
            }
            .metric-logo {
                width: 34px;
                height: 34px;
                object-fit: contain;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.92);
                padding: 0.22rem;
                flex: 0 0 auto;
            }
            .odds-list {
                display: grid;
                gap: 0.75rem;
            }
            .odds-row {
                display: grid;
                grid-template-columns: minmax(180px, 0.9fr) 1.8fr 64px;
                gap: 0.8rem;
                align-items: center;
            }
            .odds-team {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                color: #f8fafc;
                font-weight: 700;
            }
            .odds-bar {
                height: 12px;
                border-radius: 999px;
                background: rgba(30, 41, 59, 0.9);
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.12);
            }
            .odds-bar-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, #60a5fa 0%, #2563eb 100%);
            }
            .odds-pct {
                color: #cbd5e1;
                font-size: 0.9rem;
                text-align: right;
            }
            .section-label {
                color: #93c5fd;
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                margin-bottom: 0.4rem;
            }
            .panel-head {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.7rem;
            }
            .panel-logo {
                width: 34px;
                height: 34px;
                object-fit: contain;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.92);
                padding: 0.22rem;
                flex: 0 0 auto;
            }
            .inline-logo-pair {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                margin-bottom: 0.8rem;
            }
            .mini-list {
                color: #cbd5e1;
                margin: 0;
                padding-left: 1.1rem;
                line-height: 1.7;
            }
            .bracket-board {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 1rem;
                align-items: start;
            }
            .bracket-column {
                background: rgba(16, 24, 38, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.20);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.2);
            }
            .bracket-heading {
                color: #93c5fd;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                margin-bottom: 0.75rem;
            }
            .bracket-region {
                color: #f8fafc;
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.75rem;
            }
            .bracket-subhead {
                color: #94a3b8;
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin: 1rem 0 0.45rem 0;
            }
            .matchup-card {
                background: rgba(30, 41, 59, 0.78);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 14px;
                padding: 0.7rem 0.8rem;
                margin-bottom: 0.55rem;
            }
            .matchup-winner {
                color: #f8fafc;
                font-weight: 700;
                font-size: 0.98rem;
                line-height: 1.35;
            }
            .matchup-meta {
                color: #cbd5e1;
                font-size: 0.82rem;
                margin-top: 0.2rem;
            }
            .champion-card {
                background: linear-gradient(180deg, rgba(37, 99, 235, 0.28), rgba(15, 23, 42, 0.88));
                border: 1px solid rgba(147, 197, 253, 0.32);
            }
            .bracket-center {
                align-self: stretch;
            }
            .ff-wrap {
                display: grid;
                gap: 1rem;
            }
            .ff-top {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                align-items: start;
            }
            .ff-bottom {
                display: grid;
                grid-template-columns: 1fr 0.8fr;
                gap: 1rem;
                align-items: start;
            }
            .ff-col {
                display: grid;
                gap: 1rem;
            }
            .ff-stage {
                background: rgba(16, 24, 38, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.20);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.2);
            }
            .ff-label {
                color: #93c5fd;
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                margin-bottom: 0.75rem;
            }
            .ff-matchup {
                background: rgba(30, 41, 59, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 14px;
                padding: 0.8rem;
                margin-bottom: 0.7rem;
            }
            .ff-team {
                color: #f8fafc;
                font-size: 1rem;
                font-weight: 700;
                line-height: 1.35;
            }
            .ff-team-row {
                display: flex;
                align-items: center;
                gap: 0.6rem;
            }
            .ff-team-inline {
                color: #f8fafc;
                font-size: 1rem;
                font-weight: 700;
                line-height: 1.35;
                display: inline-block;
            }
            .ff-team-logo {
                width: 28px;
                height: 28px;
                object-fit: contain;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.92);
                padding: 0.18rem;
                flex: 0 0 auto;
            }
            .team-chip-wrap {
                display: flex;
                flex-wrap: wrap;
                gap: 0.8rem;
                margin-top: 0.9rem;
            }
            .team-chip {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                background: rgba(30, 41, 59, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 999px;
                padding: 0.5rem 0.8rem;
            }
            .team-chip img {
                width: 26px;
                height: 26px;
                object-fit: contain;
            }
            .stButton > button {
                background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
                color: #eff6ff;
                border: 1px solid rgba(147, 197, 253, 0.35);
                border-radius: 12px;
                font-weight: 700;
                box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
            }
            .stButton > button:hover {
                background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
                color: #ffffff;
                border-color: rgba(191, 219, 254, 0.55);
            }
            .stButton > button:focus:not(:active) {
                border-color: rgba(191, 219, 254, 0.65);
                color: #ffffff;
            }
            .ff-meta {
                color: #cbd5e1;
                font-size: 0.82rem;
                margin-top: 0.2rem;
            }
            .ff-title {
                background: linear-gradient(180deg, rgba(37, 99, 235, 0.32), rgba(15, 23, 42, 0.92));
                border: 1px solid rgba(147, 197, 253, 0.35);
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 14px;
                overflow: hidden;
            }
            @media (max-width: 900px) {
                .hero-inner {
                    flex-direction: column;
                    align-items: flex-start;
                }
                .hero-logo {
                    width: 96px;
                    height: 96px;
                }
                .bracket-board {
                    grid-template-columns: 1fr;
                }
                .ff-top,
                .ff-bottom {
                    grid-template-columns: 1fr;
                }
                .odds-row {
                    grid-template-columns: 1fr;
                    gap: 0.45rem;
                }
                .odds-pct {
                    text-align: left;
                }
            }
            @media (max-width: 640px) {
                .hero-card, .panel-card, .bracket-column, .ff-stage {
                    padding: 0.9rem 1rem;
                    border-radius: 16px;
                }
                .hero-title {
                    font-size: 1.7rem;
                    line-height: 1.15;
                }
                .hero-subtitle {
                    font-size: 0.95rem;
                }
                .metric-value {
                    font-size: 1.45rem;
                }
                .matchup-card,
                .ff-matchup,
                .team-chip {
                    padding: 0.65rem 0.75rem;
                }
                .bracket-board,
                .ff-wrap,
                .ff-col,
                .team-chip-wrap {
                    gap: 0.75rem;
                }
                .odds-team {
                    font-size: 0.95rem;
                }
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


def team_logo_html(team_name: str, css_class: str = "ff-team-logo") -> str:
    logo_path = TEAM_LOGO_MAP.get(team_name)
    if logo_path is None or not logo_path.exists():
        return ""
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    suffix = logo_path.suffix.lower().lstrip(".") or "png"
    mime_type = "image/png" if suffix == "png" else "image/webp"
    return f'<img src="data:{mime_type};base64,{encoded}" class="{css_class}" alt="{team_name} logo" />'


def _render_matchup_cards(matchups: list[dict[str, object]], champion: bool = False) -> str:
    cards = []
    extra_class = " champion-card" if champion else ""
    for matchup in matchups:
        meta = []
        if matchup.get("seed") is not None:
            meta.append(f"Seed {matchup['seed']}")
        if matchup.get("probability") is not None:
            meta.append(f"{matchup['probability']:.1%} win prob")
        meta_html = " · ".join(meta)
        cards.append(
            dedent(
                f"""
                <div class="matchup-card{extra_class}">
                    <div class="ff-team-row">{team_logo_html(matchup['team'])}<span class="matchup-winner">{matchup['team']}</span></div>
                    <div class="matchup-meta">{meta_html}</div>
                </div>
                """
            ).strip()
        )
    return "".join(cards)


def render_bracket_board(
    bracket_data: dict[str, object],
    winner_field: str,
    seed_field: str,
    probability_field: str,
) -> None:
    regions = bracket_data["regions"]
    final_four = bracket_data["final_four"]
    championship = bracket_data["championship"]

    def region_block(region_name: str) -> str:
        region_data = regions[region_name]
        round1_cards = [
            {
                "team": row[winner_field],
                "seed": int(row[seed_field]) if pd.notna(row[seed_field]) else None,
                "probability": float(row[probability_field]),
            }
            for _, row in region_data["round1"].iterrows()
        ]
        round2_cards = [
            {"team": row[winner_field], "seed": int(row[seed_field]) if pd.notna(row[seed_field]) else None, "probability": float(row[probability_field])}
            for _, row in region_data["round2"].iterrows()
        ]
        round3_cards = [
            {"team": row[winner_field], "seed": int(row[seed_field]) if pd.notna(row[seed_field]) else None, "probability": float(row[probability_field])}
            for _, row in region_data["round3"].iterrows()
        ]
        final_card = [
            {
                "team": region_data["final"].iloc[0][winner_field],
                "seed": int(region_data["final"].iloc[0][seed_field]) if pd.notna(region_data["final"].iloc[0][seed_field]) else None,
                "probability": float(region_data["final"].iloc[0][probability_field]),
            }
        ]
        return dedent(
            f"""
            <div class="bracket-column">
                <div class="bracket-heading">Deterministic Path</div>
                <div class="bracket-region">{region_name}</div>
                <div class="bracket-subhead">Round 1 Winners</div>
                {_render_matchup_cards(round1_cards)}
                <div class="bracket-subhead">Round 2 Winners</div>
                {_render_matchup_cards(round2_cards)}
                <div class="bracket-subhead">Round 3 Winners</div>
                {_render_matchup_cards(round3_cards)}
                <div class="bracket-subhead">Regional Champion</div>
                {_render_matchup_cards(final_card, champion=True)}
            </div>
            """
        ).strip()

    final_four_cards = [
        {
            "team": row[winner_field],
            "seed": int(row[seed_field]) if pd.notna(row[seed_field]) else None,
            "probability": float(row[probability_field]),
        }
        for _, row in final_four.iterrows()
    ]
    championship_card = [
        {
            "team": championship.iloc[0][winner_field],
            "seed": int(championship.iloc[0][seed_field]) if pd.notna(championship.iloc[0][seed_field]) else None,
            "probability": float(championship.iloc[0][probability_field]),
        }
    ]

    board_html = dedent(
        f"""
        <div class="bracket-board">
            {region_block("East")}
            {region_block("South")}
            <div class="bracket-column bracket-center">
                <div class="bracket-heading">Finish</div>
                <div class="bracket-region">Final Four</div>
                {_render_matchup_cards(final_four_cards)}
                <div class="bracket-subhead">National Champion</div>
                {_render_matchup_cards(championship_card, champion=True)}
            </div>
            {region_block("West")}
            {region_block("Midwest")}
        </div>
        """
    ).strip()
    st.markdown(board_html, unsafe_allow_html=True)


def render_final_four_bracket(
    bracket_data: dict[str, object],
    winner_field: str,
    probability_field: str,
    title: str,
    label: str,
) -> None:
    final_four = bracket_data["final_four"]
    championship = bracket_data["championship"]

    left = final_four.iloc[0]
    right = final_four.iloc[1]
    champ = championship.iloc[0]

    def _seed_text(row: pd.Series, key: str) -> str:
        value = row.get(key)
        return str(int(value)) if pd.notna(value) else "-"

    html = dedent(
        f"""
        <div class="panel-card">
            <div class="section-label">{label}</div>
            <h3 style="margin-top:0;color:#f8fafc;">{title}</h3>
            <div class="ff-wrap">
                <div class="ff-top">
                    <div class="ff-stage">
                        <div class="ff-label">Semifinal 1</div>
                        <div class="ff-matchup">
                            <div class="ff-team-row">{team_logo_html(left['team_a'])}<span class="ff-team-inline">{left['team_a']}</span></div>
                            <div class="ff-meta">Seed {_seed_text(left, 'seed_a')}</div>
                        </div>
                        <div class="ff-matchup">
                            <div class="ff-team-row">{team_logo_html(left['team_b'])}<span class="ff-team-inline">{left['team_b']}</span></div>
                            <div class="ff-meta">Seed {_seed_text(left, 'seed_b')}</div>
                        </div>
                        <div class="ff-matchup ff-title">
                            <div class="ff-team-row">{team_logo_html(left[winner_field])}<span class="ff-team-inline">{left[winner_field]}</span></div>
                            <div class="ff-meta">Advanced with {left[probability_field]:.1%}</div>
                        </div>
                    </div>
                    <div class="ff-stage">
                        <div class="ff-label">Semifinal 2</div>
                        <div class="ff-matchup">
                            <div class="ff-team-row">{team_logo_html(right['team_a'])}<span class="ff-team-inline">{right['team_a']}</span></div>
                            <div class="ff-meta">Seed {_seed_text(right, 'seed_a')}</div>
                        </div>
                        <div class="ff-matchup">
                            <div class="ff-team-row">{team_logo_html(right['team_b'])}<span class="ff-team-inline">{right['team_b']}</span></div>
                            <div class="ff-meta">Seed {_seed_text(right, 'seed_b')}</div>
                        </div>
                        <div class="ff-matchup ff-title">
                            <div class="ff-team-row">{team_logo_html(right[winner_field])}<span class="ff-team-inline">{right[winner_field]}</span></div>
                            <div class="ff-meta">Advanced with {right[probability_field]:.1%}</div>
                        </div>
                    </div>
                </div>
                <div class="ff-bottom">
                    <div class="ff-stage">
                        <div class="ff-label">Championship</div>
                        <div class="ff-matchup">
                            <div class="ff-team-row">{team_logo_html(champ['team_a'])}<span class="ff-team-inline">{champ['team_a']}</span></div>
                            <div class="ff-meta">Seed {_seed_text(champ, 'seed_a')}</div>
                        </div>
                        <div class="ff-matchup">
                            <div class="ff-team-row">{team_logo_html(champ['team_b'])}<span class="ff-team-inline">{champ['team_b']}</span></div>
                            <div class="ff-meta">Seed {_seed_text(champ, 'seed_b')}</div>
                        </div>
                    </div>
                    <div class="ff-stage">
                        <div class="ff-label">Champion</div>
                        <div class="ff-matchup ff-title">
                            <div class="ff-team-row">{team_logo_html(champ[winner_field])}<span class="ff-team-inline">{champ[winner_field]}</span></div>
                            <div class="ff-meta">Champion at {champ[probability_field]:.1%}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    ).strip()
    st.markdown(html, unsafe_allow_html=True)


def render_featured_team_chips(teams: list[str]) -> None:
    chips = []
    for team in teams:
        chips.append(
            dedent(
                f"""
                <div class="team-chip">
                    {team_logo_html(team, css_class="ff-team-logo")}
                    <span>{team}</span>
                </div>
                """
            ).strip()
        )
    st.markdown(
        dedent(f"""
        <div class="team-chip-wrap">{"".join(chips)}</div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, team_name: str | None = None) -> str:
    logo_html = team_logo_html(team_name, css_class="metric-logo") if team_name else ""
    return dedent(
        f"""
        <div class="panel-card">
            <div class="metric-top">
                {logo_html}
                <div class="metric-label">{label}</div>
            </div>
            <div class="metric-value">{value}</div>
        </div>
        """
    ).strip()


def render_logo_panel(
    title: str,
    label: str,
    body: str,
    teams: list[str],
) -> str:
    logos = "".join(team_logo_html(team, css_class="panel-logo") for team in teams)
    logo_row = f'<div class="inline-logo-pair">{logos}</div>' if logos else ""
    return dedent(
        f"""
        <div class="panel-card">
            <div class="section-label">{label}</div>
            {logo_row}
            <h3 style="margin-top:0;color:#f8fafc;">{title}</h3>
            <p style="color:#cbd5e1; margin-bottom:0;">{body}</p>
        </div>
        """
    ).strip()


def render_odds_chart(champion_odds: pd.DataFrame) -> None:
    rows = []
    max_pct = float(champion_odds["championship_odds_pct"].max()) if not champion_odds.empty else 1.0
    for _, row in champion_odds.iterrows():
        fill_pct = (float(row["championship_odds_pct"]) / max_pct * 100) if max_pct else 0
        rows.append(
            dedent(
                f"""
                <div class="odds-row">
                    <div class="odds-team">
                        {team_logo_html(row['team'])}
                        <span>{row['team']}</span>
                    </div>
                    <div class="odds-bar">
                        <div class="odds-bar-fill" style="width:{fill_pct:.1f}%;"></div>
                    </div>
                    <div class="odds-pct">{row['championship_odds_pct']:.1f}%</div>
                </div>
                """
            ).strip()
        )
    st.markdown(
        dedent(f"""
        <div class="odds-list">{"".join(rows)}</div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_seed_odds_chart(champion_odds: pd.DataFrame) -> None:
    chart_data = champion_odds.copy()
    chart = (
        alt.Chart(chart_data)
        .mark_circle(opacity=0.85, stroke="#dbeafe", strokeWidth=1)
        .encode(
            x=alt.X("seed:Q", title="Seed"),
            y=alt.Y("championship_odds_pct:Q", title="Title Odds (%)"),
            size=alt.Size("championship_odds_pct:Q", legend=None),
            color=alt.Color("region:N", legend=alt.Legend(title="Region")),
            tooltip=["team", "seed", "region", "championship_odds_pct"],
        )
        .properties(height=280, background="#121926")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#cbd5e1",
            titleColor="#f8fafc",
            gridColor="rgba(148,163,184,0.15)",
            domainColor="rgba(148,163,184,0.25)",
            tickColor="rgba(148,163,184,0.25)",
        )
        .configure_legend(
            labelColor="#cbd5e1",
            titleColor="#f8fafc",
        )
        .configure_title(color="#f8fafc")
    )
    st.altair_chart(chart, width="stretch", theme=None)


def render_region_odds_chart(regional_odds: dict[str, pd.DataFrame]) -> None:
    chart_data = pd.concat(
        [df.assign(region_name=region) for region, df in regional_odds.items()],
        ignore_index=True,
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("regional_win_odds_pct:Q", title="Regional Win Odds (%)"),
            y=alt.Y("team:N", sort="-x", title=None),
            color=alt.Color("region_name:N", legend=alt.Legend(title="Region")),
            tooltip=["team", "region_name", "regional_win_odds_pct", "seed"],
            row=alt.Row("region_name:N", title=None),
        )
        .properties(height=150, background="#121926")
        .resolve_scale(y="independent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#cbd5e1",
            titleColor="#f8fafc",
            gridColor="rgba(148,163,184,0.15)",
            domainColor="rgba(148,163,184,0.25)",
            tickColor="rgba(148,163,184,0.25)",
        )
        .configure_legend(labelColor="#cbd5e1", titleColor="#f8fafc")
        .configure_header(labelColor="#f8fafc", titleColor="#93c5fd")
    )
    st.altair_chart(chart, width="stretch", theme=None)


def render_final_four_odds_chart(final_four_odds: pd.DataFrame) -> None:
    chart = (
        alt.Chart(final_four_odds)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("final_four_odds_pct:Q", title="Final Four Odds (%)"),
            y=alt.Y("team:N", sort="-x", title=None),
            color=alt.Color("region:N", legend=alt.Legend(title="Region")),
            tooltip=["team", "region", "final_four_odds_pct", "seed"],
        )
        .properties(height=320, background="#121926")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#cbd5e1",
            titleColor="#f8fafc",
            gridColor="rgba(148,163,184,0.15)",
            domainColor="rgba(148,163,184,0.25)",
            tickColor="rgba(148,163,184,0.25)",
        )
        .configure_legend(labelColor="#cbd5e1", titleColor="#f8fafc")
    )
    st.altair_chart(chart, width="stretch", theme=None)


def render_upset_watch_chart(round1_predictions: pd.DataFrame) -> None:
    chart_data = (
        round1_predictions.sort_values("favorite_win_prob")
        .head(8)
        .assign(
            matchup=lambda df: df["team_a"] + " vs " + df["team_b"],
            upset_window_pct=lambda df: ((1 - df["favorite_win_prob"]) * 100).round(1),
        )
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("upset_window_pct:Q", title="Upset Window (%)"),
            y=alt.Y("matchup:N", sort="-x", title=None),
            color=alt.Color("region:N", legend=alt.Legend(title="Region")),
            tooltip=["region", "matchup", "favorite", "favorite_win_prob", "predicted_winner", "upset_window_pct"],
        )
        .properties(height=300, background="#121926")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#cbd5e1",
            titleColor="#f8fafc",
            gridColor="rgba(148,163,184,0.15)",
            domainColor="rgba(148,163,184,0.25)",
            tickColor="rgba(148,163,184,0.25)",
        )
        .configure_legend(
            labelColor="#cbd5e1",
            titleColor="#f8fafc",
        )
        .configure_title(color="#f8fafc")
    )
    st.altair_chart(chart, width="stretch", theme=None)


def render_round_comparison_chart(
    baseline_round_summary: pd.DataFrame,
    consensus_round_summary: pd.DataFrame,
) -> None:
    baseline_chart_df = baseline_round_summary[["round", "correct_picks"]].copy()
    baseline_chart_df["approach"] = "Baseline"

    consensus_chart_df = consensus_round_summary[["round", "correct_picks"]].copy()
    consensus_chart_df["approach"] = "Monte Carlo Consensus"

    chart_df = pd.concat([baseline_chart_df, consensus_chart_df], ignore_index=True)
    round_order = ["First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("round:N", sort=round_order, title="Round", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("approach:N"),
            y=alt.Y("correct_picks:Q", title="Correct Picks"),
            color=alt.Color(
                "approach:N",
                scale=alt.Scale(
                    domain=["Baseline", "Monte Carlo Consensus"],
                    range=["#60a5fa", "#f59e0b"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("round:N", title="Round"),
                alt.Tooltip("approach:N", title="Approach"),
                alt.Tooltip("correct_picks:Q", title="Correct picks"),
            ],
        )
        .properties(height=320)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#cbd5e1",
            titleColor="#e2e8f0",
            gridColor="rgba(148, 163, 184, 0.18)",
        )
        .configure_legend(labelColor="#e2e8f0")
    )
    st.altair_chart(chart, width="stretch", theme=None)


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
) -> tuple[pd.DataFrame, pd.DataFrame, LogisticRegression, pd.DataFrame, pd.DataFrame]:
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

    return round1_predictions, historical_matchups, model, bracket_with_features, first_round_2026


def build_round1_id_tables(
    first_round_2026: pd.DataFrame,
    bracket_with_features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    team_lookup = bracket_with_features[["region", "seed", "team", "TeamID"]].copy()
    region_tables: dict[str, pd.DataFrame] = {}

    for region_name in ["East", "West", "South", "Midwest"]:
        region_round1 = first_round_2026[first_round_2026["region"] == region_name].copy()
        region_round1 = region_round1.merge(
            team_lookup.rename(
                columns={"team": "team_a", "seed": "seed_a", "TeamID": "team_a_id"}
            ),
            on=["region", "team_a", "seed_a"],
            how="left",
        )
        region_round1 = region_round1.merge(
            team_lookup.rename(
                columns={"team": "team_b", "seed": "seed_b", "TeamID": "team_b_id"}
            ),
            on=["region", "team_b", "seed_b"],
            how="left",
        )
        region_tables[region_name] = region_round1

    return region_tables


def _attach_features_for_round(matchup_df: pd.DataFrame, features_latest: pd.DataFrame) -> pd.DataFrame:
    matchup_df = matchup_df.merge(
        features_latest.add_prefix("team_a_"),
        left_on="team_a_id",
        right_on="team_a_TeamID",
        how="left",
    )
    matchup_df = matchup_df.merge(
        features_latest.add_prefix("team_b_"),
        left_on="team_b_id",
        right_on="team_b_TeamID",
        how="left",
    )
    matchup_df["win_pct_diff"] = matchup_df["team_a_win_pct"] - matchup_df["team_b_win_pct"]
    matchup_df["points_for_diff"] = matchup_df["team_a_avg_points_for"] - matchup_df["team_b_avg_points_for"]
    matchup_df["points_against_diff"] = matchup_df["team_a_avg_points_against"] - matchup_df["team_b_avg_points_against"]
    matchup_df["scoring_margin_diff"] = (
        matchup_df["team_a_avg_scoring_margin"] - matchup_df["team_b_avg_scoring_margin"]
    )
    return matchup_df


def _score_deterministic_round(
    matchup_df: pd.DataFrame,
    features_latest: pd.DataFrame,
    model: LogisticRegression,
) -> pd.DataFrame:
    matchup_df = _attach_features_for_round(matchup_df.copy(), features_latest)
    X_round = matchup_df[FEATURE_COLS]
    matchup_df["team_a_win_prob"] = model.predict_proba(X_round)[:, 1]
    matchup_df["team_b_win_prob"] = 1 - matchup_df["team_a_win_prob"]
    matchup_df["predicted_winner"] = matchup_df.apply(
        lambda row: row["team_a"] if row["team_a_win_prob"] >= 0.5 else row["team_b"],
        axis=1,
    )
    matchup_df["winner_seed"] = matchup_df.apply(
        lambda row: row["seed_a"] if row["predicted_winner"] == row["team_a"] else row["seed_b"],
        axis=1,
    )
    matchup_df["winner_team_id"] = matchup_df.apply(
        lambda row: row["team_a_id"] if row["predicted_winner"] == row["team_a"] else row["team_b_id"],
        axis=1,
    )
    return matchup_df


def _build_next_round_rows(previous_round: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for idx in range(0, len(previous_round), 2):
        game_a = previous_round.iloc[idx]
        game_b = previous_round.iloc[idx + 1]
        rows.append(
            {
                "team_a": game_a["predicted_winner"],
                "seed_a": game_a["winner_seed"],
                "team_a_id": game_a["winner_team_id"],
                "team_b": game_b["predicted_winner"],
                "seed_b": game_b["winner_seed"],
                "team_b_id": game_b["winner_team_id"],
            }
        )
    return pd.DataFrame(rows)


def build_deterministic_region_summary(
    region_round1_ids: pd.DataFrame,
    features_latest: pd.DataFrame,
    model: LogisticRegression,
) -> dict[str, pd.DataFrame | str]:
    round1 = region_round1_ids.copy()
    round1["predicted_winner"] = round1.apply(
        lambda row: row["team_a"] if row["team_a_win_prob"] >= 0.5 else row["team_b"],
        axis=1,
    )
    round1["winner_seed"] = round1.apply(
        lambda row: row["seed_a"] if row["predicted_winner"] == row["team_a"] else row["seed_b"],
        axis=1,
    )
    round1["winner_team_id"] = round1.apply(
        lambda row: row["team_a_id"] if row["predicted_winner"] == row["team_a"] else row["team_b_id"],
        axis=1,
    )

    round2 = _score_deterministic_round(_build_next_round_rows(round1), features_latest, model)
    round3 = _score_deterministic_round(_build_next_round_rows(round2), features_latest, model)
    final = _score_deterministic_round(_build_next_round_rows(round3), features_latest, model)

    return {
        "round1": round1,
        "round2": round2,
        "round3": round3,
        "final": final,
        "champion": final.iloc[0]["predicted_winner"],
        "champion_seed": int(final.iloc[0]["winner_seed"]),
        "champion_team_id": int(final.iloc[0]["winner_team_id"]),
    }


def build_deterministic_bracket_summary(
    round1_id_tables: dict[str, pd.DataFrame],
    features_latest: pd.DataFrame,
    model: LogisticRegression,
) -> dict[str, object]:
    east = build_deterministic_region_summary(round1_id_tables["East"], features_latest, model)
    west = build_deterministic_region_summary(round1_id_tables["West"], features_latest, model)
    south = build_deterministic_region_summary(round1_id_tables["South"], features_latest, model)
    midwest = build_deterministic_region_summary(round1_id_tables["Midwest"], features_latest, model)

    final_four = pd.DataFrame(
        [
            {
                "team_a": south["champion"],
                "seed_a": south["champion_seed"],
                "team_a_id": south["champion_team_id"],
                "team_b": west["champion"],
                "seed_b": west["champion_seed"],
                "team_b_id": west["champion_team_id"],
            },
            {
                "team_a": east["champion"],
                "seed_a": east["champion_seed"],
                "team_a_id": east["champion_team_id"],
                "team_b": midwest["champion"],
                "seed_b": midwest["champion_seed"],
                "team_b_id": midwest["champion_team_id"],
            },
        ]
    )
    final_four = _score_deterministic_round(final_four, features_latest, model)
    championship = _score_deterministic_round(_build_next_round_rows(final_four), features_latest, model)

    return {
        "regions": {
            "East": east,
            "West": west,
            "South": south,
            "Midwest": midwest,
        },
        "final_four": final_four,
        "championship": championship,
        "champion": championship.iloc[0]["predicted_winner"],
    }


def annotate_champion_odds(champion_odds: pd.DataFrame, bracket_2026: pd.DataFrame) -> pd.DataFrame:
    seed_lookup = (
        bracket_2026[~bracket_2026["is_play_in"]][["team", "seed", "region"]]
        .drop_duplicates()
        .rename(columns={"team": "team"})
    )
    return champion_odds.merge(seed_lookup, on="team", how="left")


def annotate_team_odds(odds_df: pd.DataFrame, bracket_2026: pd.DataFrame) -> pd.DataFrame:
    seed_lookup = (
        bracket_2026[~bracket_2026["is_play_in"]][["team", "seed", "region"]]
        .drop_duplicates()
        .rename(columns={"team": "team"})
    )
    return odds_df.merge(seed_lookup, on="team", how="left")


@st.cache_data
def load_actual_results_data() -> pd.DataFrame | None:
    if not ACTUAL_RESULTS_PATH.exists():
        return None
    return load_actual_results(ACTUAL_RESULTS_PATH)


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

    round1_predictions, historical_matchups, model, bracket_with_features, first_round_2026 = build_round1_predictions(
        data, team_features, bracket_2026
    )
    round1_id_tables = build_round1_id_tables(first_round_2026, bracket_with_features)
    latest_season = int(team_features["Season"].max())
    features_latest = team_features[team_features["Season"] == latest_season].copy()
    deterministic_bracket = build_deterministic_bracket_summary(
        round1_id_tables,
        features_latest,
        model,
    )
    cached_simulation = load_simulation_outputs(SIMULATION_CACHE_DIR)
    if cached_simulation is not None:
        consensus_bracket = cached_simulation["consensus_bracket"]
        champion_odds = cached_simulation["champion_odds"]
        final_four_odds = cached_simulation["final_four_odds"]
        regional_odds = cached_simulation["regional_odds"]
    else:
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
            random_seed=42,
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
            random_seed=42,
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

    return (
        data,
        team_features,
        bracket_2026,
        round1_predictions,
        historical_matchups,
        deterministic_bracket,
        consensus_bracket,
        champion_odds,
        final_four_odds,
        regional_odds,
        round1_id_tables,
        features_latest,
        model,
    )


def render_evaluation_section(
    *,
    bracket_2026: pd.DataFrame,
    deterministic_bracket: dict[str, object],
    consensus_bracket: dict[str, object],
    champion_odds: pd.DataFrame,
) -> None:
    actual_results = load_actual_results_data()
    if actual_results is None:
        st.info("Add `data/raw/2026_bracket_results.csv` to show the post-tournament evaluation section.")
        return

    bracket_teams = set(bracket_2026["team"].dropna())
    unknown_in_results, missing_from_results = validate_results_teams(bracket_teams, actual_results)
    if unknown_in_results or missing_from_results:
        st.warning("The 2026 results file has team names that do not line up with the bracket input yet.")
        if unknown_in_results:
            st.write("Results-only names:", ", ".join(unknown_in_results))
        if missing_from_results:
            st.write("Bracket-only names:", ", ".join(missing_from_results))
        return

    baseline_eval = evaluate_bracket_summary(
        deterministic_bracket,
        actual_results,
        winner_field="predicted_winner",
        champion_field="champion",
    )
    monte_carlo_summary = summarize_monte_carlo(
        consensus_bracket=consensus_bracket,
        champion_odds=champion_odds,
        actual_results=actual_results,
    )
    consensus_eval = monte_carlo_summary["consensus_eval"]

    overview_left, overview_right = st.columns(2)

    with overview_left:
        panel(
            "Baseline Result Check",
            "After The Tournament",
            f"The deterministic bracket finished {baseline_eval['overall_correct']} for {baseline_eval['overall_games']} "
            f"for an accuracy of {baseline_eval['overall_accuracy']:.1%}. "
            f"It picked {baseline_eval['predicted_champion']} and the actual champion was {baseline_eval['actual_champion']}.",
        )
        st.dataframe(baseline_eval["round_summary"], width="stretch", hide_index=True)

    with overview_right:
        actual_rank = monte_carlo_summary["actual_champion_rank"]
        actual_rank_text = (
            f"The actual champion ranked #{actual_rank} in the Monte Carlo title odds at about "
            f"{monte_carlo_summary['actual_champion_odds_pct']:.1f}%."
            if actual_rank is not None
            else "The actual champion did not appear in the top Monte Carlo title odds table."
        )
        panel(
            "Monte Carlo Check",
            "After The Tournament",
            f"The consensus Monte Carlo bracket also finished {consensus_eval['overall_correct']} for "
            f"{consensus_eval['overall_games']}, but its projected champion was "
            f"{monte_carlo_summary['simulation_favorite']} instead of {consensus_eval['actual_champion']}. "
            f"{actual_rank_text}",
        )
        st.dataframe(consensus_eval["round_summary"], width="stretch", hide_index=True)

    st.markdown("### What Changed Once Games Were Played")
    st.write(
        f"The baseline held up best in the First Round at {baseline_eval['round_summary'].iloc[0]['accuracy']:.1%}, "
        f"then got less stable in later rounds. The consensus Monte Carlo bracket finished "
        f"{consensus_eval['overall_correct']} for {consensus_eval['overall_games']} "
        f"for an accuracy of {consensus_eval['overall_accuracy']:.1%}."
    )
    st.write(
        f"The deterministic bracket correctly landed the national champion with {baseline_eval['predicted_champion']}. "
        f"The consensus Monte Carlo bracket tied the baseline on total correct picks, but it missed the champion. "
        f"At the same time, the full simulation still treated {consensus_eval['actual_champion']} as a real contender, "
        f"ranking them #{monte_carlo_summary['actual_champion_rank']} in title odds."
    )

    st.markdown("#### Round-By-Round Comparison")
    render_round_comparison_chart(
        baseline_eval["round_summary"],
        consensus_eval["round_summary"],
    )

    missed_left, missed_right = st.columns(2)
    with missed_left:
        st.markdown("#### Baseline Misses")
        st.dataframe(baseline_eval["missed_games"], width="stretch", hide_index=True)
    with missed_right:
        st.markdown("#### Consensus Misses")
        st.dataframe(consensus_eval["missed_games"], width="stretch", hide_index=True)


def main() -> None:
    apply_styles()
    (
        _data,
        _team_features,
        bracket_2026,
        round1_predictions,
        _historical_matchups,
        deterministic_bracket,
        consensus_bracket,
        champion_odds,
        final_four_odds,
        regional_odds,
        round1_id_tables,
        features_latest,
        model,
    ) = load_app_data()

    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=170)
        st.markdown("### What This App Shows")
        st.markdown("- Deterministic bracket board")
        st.markdown("- 1000-simulation title odds")
        st.markdown("- Sleeper and takeaway cards")
        st.markdown("- One random tournament run")

    hero_logo_html = ""
    if HERO_PATH.exists():
        hero_logo_html = (
            "<img src='data:image/png;base64,"
            + base64.b64encode(HERO_PATH.read_bytes()).decode("ascii")
            + "' class='hero-logo' alt=\"Andrea's Bracket Breakdown logo\" />"
        )

    st.markdown(
        dedent(
            f"""
            <div class="hero-card">
                <div class="hero-inner">
                    <div class="hero-copy">
                        <div class="section-label">Bracket Dashboard</div>
                        <div class="hero-title">Andrea's Bracket Breakdown</div>
                        <p class="hero-subtitle">
                            A cleaner view of the project: one deterministic bracket, one Monte Carlo picture,
                            and the teams the model thinks matter most.
                        </p>
                    </div>
                    {hero_logo_html}
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.write("")

    simulation_favorite = champion_odds.iloc[0]
    sleeper_candidates = champion_odds[champion_odds["seed"].fillna(0) >= 8]
    sleeper_pick = sleeper_candidates.iloc[0] if not sleeper_candidates.empty else champion_odds.iloc[min(1, len(champion_odds) - 1)]

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Baseline Champion", f"{deterministic_bracket['champion']}", deterministic_bracket["champion"]),
        ("Sim Favorite", f"{simulation_favorite['team']}", simulation_favorite["team"]),
        ("Top Odds", f"{simulation_favorite['championship_odds_pct']}%", simulation_favorite["team"]),
        ("Sleeper Pick", f"{sleeper_pick['team']}", sleeper_pick["team"]),
    ]
    for col, (label, value, team_name) in zip([col1, col2, col3, col4], metrics):
        col.markdown(render_metric_card(label, value, team_name=team_name), unsafe_allow_html=True)

    st.write("")

    render_featured_team_chips(["Duke", "Florida", "Gonzaga", "Saint Louis"])

    st.write("")

    ff_left, ff_right = st.columns(2)

    with ff_left:
        render_final_four_bracket(
            deterministic_bracket,
            winner_field="predicted_winner",
            probability_field="team_a_win_prob",
            title="Baseline Final Four Bracket",
            label="Before Monte Carlo",
        )

    with ff_right:
        render_final_four_bracket(
            consensus_bracket,
            winner_field="consensus_winner",
            probability_field="consensus_share",
            title="Consensus Simulation Final Four",
            label="After 1000 Simulations",
        )

    st.write("")

    panel(
        "Deterministic Bracket",
        "Before Monte Carlo",
        "This board shows the single baseline tournament path when the higher-probability team advances in every game.",
    )
    render_bracket_board(
        deterministic_bracket,
        winner_field="predicted_winner",
        seed_field="winner_seed",
        probability_field="team_a_win_prob",
    )

    st.write("")

    panel(
        "Consensus Simulation Bracket",
        "After 1000 Simulations",
        "This board shows the team that advanced most often in each bracket slot across the 1000 Monte Carlo runs.",
    )
    render_bracket_board(
        consensus_bracket,
        winner_field="consensus_winner",
        seed_field="consensus_seed",
        probability_field="consensus_share",
    )

    st.write("")

    insights_left, insights_center, insights_right = st.columns(3)

    with insights_left:
        st.markdown(
            render_logo_panel(
                "Baseline vs Simulation",
                "Key Takeaway",
                f"The deterministic bracket champion is {deterministic_bracket['champion']}, but the most common simulated champion is {simulation_favorite['team']} at about {simulation_favorite['championship_odds_pct']}%.",
                [deterministic_bracket["champion"], simulation_favorite["team"]],
            ),
            unsafe_allow_html=True,
        )

    with insights_center:
        sleeper_region = f" from the {sleeper_pick['region']} region" if pd.notna(sleeper_pick.get("region")) else ""
        st.markdown(
            render_logo_panel(
                "Sleeper Signal",
                "Watch List",
                f"{sleeper_pick['team']} is a lower-seeded team{sleeper_region} that still shows up with meaningful title odds in the current simulation output.",
                [sleeper_pick["team"]],
            ),
            unsafe_allow_html=True,
        )

    with insights_right:
        if st.button("Run One Simulated Tournament"):
            sim_champion = simulate_full_tournament_once(
                round1_id_tables["East"],
                round1_id_tables["West"],
                round1_id_tables["South"],
                round1_id_tables["Midwest"],
                features_latest,
                FEATURE_COLS,
                model,
                SECOND_ROUND_PAIRS,
            )
            st.session_state["sim_champion"] = sim_champion

        sim_champion = st.session_state.get("sim_champion", "Not run yet")
        sim_teams = [] if sim_champion == "Not run yet" else [sim_champion]
        st.markdown('<div class="section-label">Interactive</div>', unsafe_allow_html=True)
        if sim_teams:
            render_featured_team_chips(sim_teams)
        st.markdown("### One Random Run")
        st.write(f"Use the button to sample one random tournament path. Current simulated champion: {sim_champion}.")

    st.write("")

    odds_extra_left, odds_extra_right = st.columns(2)

    with odds_extra_left:
        panel(
            "Regional Win Odds",
            "Simulation Chart",
            "This view breaks the simulations back down by region so you can see who wins East, West, South, and Midwest most often.",
        )
        render_region_odds_chart(regional_odds)

    with odds_extra_right:
        panel(
            "Final Four Odds",
            "Simulation Chart",
            "This chart shows which teams make the Final Four most often across the full Monte Carlo run, not just who wins the title most often.",
        )
        render_final_four_odds_chart(final_four_odds)

    st.write("")

    odds_left, odds_right = st.columns([1.15, 1])

    with odds_left:
        panel(
            "Monte Carlo Championship Odds",
            "After 1000 Simulations",
            "These are the top title odds from repeated full-tournament simulations. This is the probability view that sits next to the single deterministic bracket.",
        )
        render_odds_chart(champion_odds)

    with odds_right:
        panel(
            "Title Odds By Seed",
            "Simulation Chart",
            "This view shows where the strongest title paths sit across seeds and regions, which makes it easier to spot both contenders and overperforming teams.",
        )
        render_seed_odds_chart(champion_odds)

    st.write("")

    panel(
        "Round 1 Upset Watch",
        "Useful Extra",
        "These are the round-1 games where the favorite is most vulnerable, so this is the quickest place to look for toss-ups and upset chances.",
    )
    render_upset_watch_chart(round1_predictions)

    st.write("")

    if st.button("Show 2026 Post-Tournament Evaluation"):
        st.session_state["show_2026_evaluation"] = not st.session_state.get("show_2026_evaluation", False)

    if st.session_state.get("show_2026_evaluation", False):
        st.write("")
        panel(
            "2026 Results Check",
            "Post-Tournament Review",
            "This section compares the deterministic bracket and the consensus Monte Carlo picture against the actual 2026 tournament results.",
        )
        render_evaluation_section(
            bracket_2026=bracket_2026,
            deterministic_bracket=deterministic_bracket,
            consensus_bracket=consensus_bracket,
            champion_odds=champion_odds,
        )


if __name__ == "__main__":
    main()
