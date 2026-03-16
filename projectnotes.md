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
- Loaded and cleaned the 2026 bracket into `bracket_2026.csv`
- Matched the released bracket teams to Kaggle `MTeams.csv`
- Built a manual name-cleaning step for team names that did not match directly
- Identified and separated the 3 play-in placeholder rows from the main bracket
- Confirmed all non-play-in 2026 bracket teams map to valid `TeamID` values
- Confirmed all non-play-in 2026 bracket teams have 2026 season features
- Built first-round 2026 matchup rows for the main bracket teams
- Built a historical matchup dataset from past NCAA tournament games using neutral team ordering
- Created training-style feature differences and a valid binary target column `team_a_won`

### What I learned
- Bracket team names often need normalization before they can match Kaggle team names
- Play-in games need separate handling because some seed slots are not resolved yet
- One game can be represented as one row using feature differences between two teams
- Historical tournament games can be transformed into supervised learning data for logistic regression
- The training data currently has both classes in the target, which means the label setup is valid

### Current Project State
- Setup and environment are working
- Core men's data is loaded and understood
- Team-season feature engineering is working
- 2026 bracket teams are matched and mostly ready
- Historical training rows are built
- Next major step is training the first logistic regression baseline

### Next session
- Split the historical matchup dataset into train/test sets
- Train a baseline logistic regression model
- Evaluate model performance
- Apply the model to 2026 first-round matchup rows
- Later handle play-in winners and full bracket simulation

