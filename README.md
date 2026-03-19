# Andrea's Bracket Breakdown

March Madness bracket simulation project using historical NCAA data, matchup-based feature engineering, logistic regression, and Monte Carlo simulation.

## Overview

This project predicts NCAA tournament games in two ways:

- a **deterministic baseline bracket** that always advances the higher-probability team
- a **Monte Carlo simulation view** that runs the full tournament many times and estimates how often different outcomes happen

The goal is not just to pick one bracket. The project is meant to answer questions like:

- Who is the most likely national champion?
- Which teams reach the Final Four most often?
- Which lower-seeded teams look dangerous?
- Which first-round games are the most upset-prone?

## Data

Main source:

- Kaggle `march-machine-learning-mania-2026`

Main men's files used:

- `MTeams.csv`
- `MRegularSeasonCompactResults.csv`
- `MNCAATourneyCompactResults.csv`
- `MNCAATourneySeeds.csv`
- `MNCAATourneySlots.csv`

Project-specific bracket file:

- `data/raw/bracket_2026.csv`

## Current Model

The baseline model is a logistic regression trained on historical NCAA tournament matchups.

Current matchup-difference features:

- `win_pct_diff`
- `points_for_diff`
- `points_against_diff`
- `scoring_margin_diff`

Baseline held-out performance:

- Accuracy: about `0.69`
- Log loss: about `0.58`

## Current Results

Deterministic bracket:

- East champion: `Duke`
- West champion: `Gonzaga`
- South champion: `Florida`
- Midwest champion: `Michigan`
- National champion: `Michigan`

Monte Carlo summary from `1000` full tournament simulations:

- `Duke`: about `17.5%`
- `Gonzaga`: about `14.0%`
- `Michigan`: about `13.7%`
- `Arizona`: about `10.1%`
- `Iowa St.`: about `6.8%`
- `Saint Louis`: about `6.2%`

This is one of the main takeaways of the project: the deterministic bracket picked `Michigan`, but the most common simulated champion was `Duke`.

## App

The Streamlit app now shows:

- a deterministic bracket view
- a consensus simulation Final Four view
- championship odds from Monte Carlo simulation
- round-1 upset watch
- one random tournament run

Branding:

- **Andrea's Bracket Breakdown**

## Project Structure

```text
march-madness-bracket-simulator/
|- app/
|- assets/
|- data/
|- notebooks/
|- src/
|  \- march_madness_bracket_simulator/
|     |- analysis.py
|     |- data_loader.py
|     |- feature_engineering.py
|     |- model.py
|     |- simulator.py
|     \- __init__.py
|- tests/
|- README.md
|- projectnotes.md
|- andrea.md
|- pyproject.toml
\- uv.lock
```

## Setup

Using `uv`:

```bash
uv sync
```

Activate the environment:

```bash
source .venv/Scripts/activate
```

Run tests:

```bash
./.venv/Scripts/python.exe -m pytest tests/test_simulator.py
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```

## Next Steps

- add Final Four odds and regional win odds
- define sleeper teams more explicitly using simulation outputs
- keep improving the app layout and bracket presentation
- record a walkthrough video

## Author

Andrea Churchwell
