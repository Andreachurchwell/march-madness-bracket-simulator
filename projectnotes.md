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

## Day 3 Progress

### What I finished
- Built a historical tournament matchup training dataset from past NCAA tournament games
- Converted historical tournament games into neutral team-vs-team rows with a valid binary target column `team_a_won`
- Trained a baseline logistic regression model using matchup feature differences
- Evaluated the baseline model on a held-out test set
- Applied the trained model to the 2026 first-round bracket matchups
- Generated first-round win probabilities and predicted winners for the non-play-in games
- Identified some early upset-style and sleeper-style picks from the model output

### Baseline Model Result
- Accuracy: about `0.69`
- Log loss: about `0.58`

### What I learned
- The current feature set already contains real predictive signal
- A simple logistic regression baseline is enough to produce meaningful first-round probabilities
- Historical tournament games can be turned into a usable supervised learning dataset with neutral team ordering
- The model is already surfacing possible upset picks where the lower-seeded or less expected team has a stronger probability than I might expect from seed alone

### Current Project State
- Setup is working
- Core men's data is loaded
- Team-season feature engineering is working
- 2026 bracket teams are matched
- Historical training data is built
- Baseline model is trained
- 2026 round 1 predictions are generated

### Next Steps
- Clean up and summarize the first-round prediction outputs
- Handle the play-in matchups that were skipped
- Build bracket advancement logic for later rounds
- Run Monte Carlo simulations across the bracket
- Expand sleeper-team analysis based on prediction and simulation results

## Day 4 Progress

### What I finished
- Restored the notebook environment after the kernel broke following `uv sync`
- Added notebook dependencies back into `pyproject.toml`
- Re-synced the environment and re-registered the project Jupyter kernel
- Improved the Streamlit app so it shows the current 2026 round-1 baseline predictions
- Added a more meaningful app view for closest games, underdog-style picks, and unresolved play-in rows

### What I learned
- If notebook dependencies are not tracked in `pyproject.toml`, `uv sync` can remove them from the project environment
- The Streamlit app makes more sense when it shows one real output from the project instead of only placeholder sections
- The current pipeline is strong enough to surface real first-round probabilities, but play-in handling and bracket advancement still need to be built

### Current Project State
- Team-season feature engineering is working
- Historical matchup training data is built
- Baseline logistic regression is working
- 2026 round-1 predictions are working for non-play-in games
- Streamlit now shows a useful baseline prediction view
- Play-in logic and round advancement are still next

### Next Steps
- Handle the play-in teams
- Build bracket advancement logic for round 2 and later rounds
- Run Monte Carlo simulation across the full bracket

## Day 5 Progress

### What I finished
- Updated the 2026 bracket after one play-in result was finalized
- Confirmed that only 2 unresolved play-in rows remain in `bracket_2026.csv`
- Rebuilt the first-round matchup pipeline with the updated bracket
- Learned how to advance winners from one round to the next using `TeamID` instead of relying only on display names
- Built and scored later-round matchups for the East region through the regional final
- Built and scored later-round matchups for the West region through the regional final

### Regional Results So Far
- East champion: `Duke`
- West champion: `Gonzaga`

### What I learned
- Round 1 can look simpler because the teams are still directly tied to the cleaned bracket table, but later rounds need `TeamID` to be carried forward or the pipeline falls back into team-name matching problems
- `TeamID` is the more reliable identifier for bracket advancement than display names
- Once winners carry their seeds and `TeamID`, the same modeling pattern can be reused for round 2, round 3, and the regional final
- Some model picks feel more intuitive than others, which is a useful reminder that this is still a baseline model and not a finished simulator

### Current Project State
- East region is fully advanced and scored
- West region is fully advanced and scored
- South and Midwest are still waiting on the remaining play-in teams
- The bracket advancement pattern is now understood well enough to repeat for the rest of the tournament

### Next Steps
- Update the bracket once the last 2 play-in games are finalized
- Build and score the South and Midwest regions
- Build the Final Four and championship matchups
- Move toward full Monte Carlo simulation once the full bracket flow is in place

