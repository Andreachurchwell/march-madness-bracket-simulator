## Day 1 Progress

### What I finished
- Fixed the project package structure so `uv` works normally
- Confirmed the `.venv` and notebook kernel are using the correct environment
- Identified the core men's NCAA datasets for the simulator
- Built working data loader functions for the main datasets
- Built the first team-season feature table from regular season results
- Created an initial Streamlit app shell for the project

### What I learned
- `TeamID` and `Season` are the main join keys
- `MNCAATourneySlots.csv` controls bracket progression across rounds
- `MNCAATourneySeeds.csv` maps teams into the bracket structure
- The regular season results can be used to build team strength features before the bracket is released

### Next session
- Merge team names and seeds into the feature table more cleanly
- Start defining sleeper-team signals
- Build matchup-level features for two teams in the same game


## Day 2 Progress

### What I finished
- Created `bracket_2026.csv` from the released 2026 bracket
- Matched the 2026 bracket teams to Kaggle `MTeams.csv`
- Built a name-cleaning step to fix bracket team names that did not match Kaggle names directly
- Identified and separated the 3 play-in placeholder rows from the main bracket teams
- Confirmed that all non-play-in 2026 bracket teams map to valid `TeamID` values
- Confirmed that all non-play-in 2026 bracket teams have 2026 season feature values
- Built first-round matchup feature rows for the non-play-in games

### What I learned
- The released bracket team names do not always match Kaggle team names exactly
- Play-in games need separate handling because they leave some seed slots temporarily unresolved
- One game can be represented as one matchup row using feature differences like seed difference, win percentage difference, and scoring margin difference
- The current bracket can now be connected to the season feature table, which means the project is ready to move toward prediction

### Next session
- Add a few more matchup features such as points-for difference and points-against difference
- Build historical tournament matchup rows to create training data for logistic regression
- Train a first logistic regression model for win probabilities
- Handle play-in winners so the full first round can be represented
- Use Monte Carlo simulation to advance teams through the bracket
