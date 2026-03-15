# Project Notes

## Focus
- Men's March Madness bracket simulator
- Use historical NCAA data to build matchup-based predictions and simulate tournament outcomes

## Setup Status
- `uv` environment is working
- `.venv` is active
- notebook kernel is using project `.venv`
- package structure is under `src/march_madness_bracket_simulator/`

## Core Datasets
- `MTeams.csv`: team lookup table
- `MRegularSeasonCompactResults.csv`: regular-season game results
- `MNCAATourneyCompactResults.csv`: historical NCAA tournament results
- `MNCAATourneySeeds.csv`: tournament seed assignments by season
- `MNCAATourneySlots.csv`: bracket slot structure and round progression

## What I Confirmed
- `TeamID` is the team join key
- `Season` is the season join key
- regular season and tournament results share the same game structure
- first-round slots are seed vs seed
- later-round slots depend on winners of previous slots

## Next Steps
- Map seeds to team names for one sample season
- Build team-season summary stats from regular season results
- Create matchup-level features
- Train a win probability model
- Simulate the bracket
