## Project Walkthrough Script

This project is called **Andrea's Bracket Breakdown**. It's basically my March Madness bracket simulator. The whole idea was to see if I could use real NCAA data, a baseline machine learning model, and Monte Carlo simulation to make smarter bracket predictions instead of just guessing.

The main dataset I used came from Kaggle's `march-machine-learning-mania-2026` dataset. From that, I used the men's team file, regular season results, tournament results, seeds, and bracket slot structure. I also made my own `bracket_2026.csv` file so I could line up the actual current bracket with the Kaggle data.

The first thing I had to do was make sure the environment and project structure worked. After that, I spent time just understanding the data, especially what keys connected everything. The main ones I used were `TeamID` and `Season`.

Once I understood that, I built a team-season feature table from regular season results. The features I picked were:

- wins and losses
- win percentage
- average points scored
- average points allowed
- average scoring margin

I picked those because they were simple, interpretable, and gave me a basic way to measure team strength without getting into really advanced stats yet.

After that, I matched the current 2026 bracket teams to Kaggle team IDs. Some team names didn't match perfectly, so I had to manually normalize a bunch of names. That turned out to matter a lot later.

Then I changed the problem from "what does one team look like?" to "how do two teams compare in one matchup?" So instead of feeding the model raw team stats, I built matchup-difference features. The four main ones I actually used in the baseline model were:

- win percentage difference
- average points scored difference
- average points allowed difference
- scoring margin difference

I picked those because they were the clearest direct comparisons between two teams. For example, instead of just saying one team scores 78 points per game, I wanted the model to see whether Team A scores more or less than Team B.

For the model, I used **logistic regression** as a baseline. I trained it on historical NCAA tournament games. I built those historical games into neutral Team A vs Team B rows so the winner would not always be on the same side.

The baseline model got about **69% accuracy** and a **log loss around 0.58** on held-out historical tournament games.

Accuracy is the easy part. It just means the model picked the winner correctly about 69% of the time.

Log loss is a little different. It measures how good the model's **probabilities** were, not just whether the final pick was right. So if the model says a team has a 95% chance to win and that team loses, log loss penalizes that a lot. Lower log loss is better. So for this project, accuracy told me the model was making decent picks, and log loss told me the probabilities were also reasonably useful.

After I trained the model, I used it to predict the 2026 bracket. At first that was just round 1, but then I started building the bracket forward round by round.

One of the biggest things I learned was that later rounds really needed me to carry `TeamID` forward, not just the team names. Round 1 was easier because it still matched directly to the cleaned bracket table, but once I started advancing winners, using only display names caused too many problems. Carrying `TeamID` forward made later-round feature rebuilding actually work.

Using that process, I built the full deterministic bracket. In the deterministic version, the model always advances the higher-probability team. That gave me these regional champions:

- East: Duke
- West: Gonzaga
- South: Florida
- Midwest: Michigan

And in that full deterministic bracket, the national champion came out as **Michigan**.

After that, I moved on to **Monte Carlo simulation**, which was really the point where the project got more interesting. Instead of always advancing the higher-probability team, Monte Carlo uses the model's win probabilities to randomly simulate the whole tournament over and over.

So if one team has a 55% win probability, Monte Carlo doesn't just automatically advance them. It lets that team win about 55% of the time and lose about 45% of the time across many simulations.

I ended up running **1000 full tournament simulations**.

That gave me a more realistic picture than one single bracket. The deterministic bracket picked **Michigan** as champion, but across the 1000 simulations, the most common champion was **Duke at about 18.5%**. After that, Gonzaga and Michigan were both around **13.9%**, and Arizona was around **8.5%**.

That was one of the biggest project takeaways for me: the single bracket and the simulation do not always tell the exact same story. One bracket path said Michigan, but the broader probability view said Duke showed up most often.

The simulations also helped highlight teams like **Saint Louis** as sleeper-style teams, because they showed up more often than I would expect just from looking at seed alone.

After that, I moved the reusable simulation logic out of the notebook and into the Python package so the project was cleaner and the app could use the same logic. Then I updated the Streamlit app to show:

- a deterministic bracket view
- a consensus simulation view
- championship odds
- Final Four odds
- regional win odds
- a round-1 upset watch
- one random simulated tournament run

So at this point, the project does both sides of the job. It can show one clean baseline bracket, and it can also show the bigger Monte Carlo picture of how often different outcomes happen.

If I kept going, the next things I'd improve would be defining sleepers more clearly, saving the simulation outputs so the app loads faster, and polishing the app even more. But overall, the main goal is already there: using real data and simulation to make bracket predictions in a way that is more thoughtful than just picking teams by seed.

## Short Version

I built a March Madness bracket simulator using Kaggle NCAA data, team-season features, matchup-difference features, logistic regression, and Monte Carlo simulation. The baseline model got about 69% accuracy with a log loss around 0.58. I used it to build a full deterministic bracket where Michigan was the champion, then ran 1000 tournament simulations, where Duke showed up most often as champion at about 18.5%. The final project includes reusable simulation code and a Streamlit app that shows the bracket, title odds, Final Four odds, regional win odds, and upset-focused views.

## What I Need To Understand To Talk About It

### Why these features?
I picked them because they were:
- easy to understand
- available for every team
- good first indicators of team strength

### Why feature differences instead of raw stats?
Because the model predicts **games**, not teams by themselves. A game is about how Team A compares to Team B.

### Why logistic regression?
Because it's a strong, simple baseline for binary outcomes like win/loss, and it gives probabilities.

### What does accuracy mean?
How often the model picked the winner correctly.

### What does log loss mean?
How good the model's probabilities were. Lower is better.

### What is the deterministic bracket?
A single bracket where the higher-probability team always advances.

### What is Monte Carlo?
Running the whole tournament many times using the model probabilities, so I can estimate odds instead of pretending one bracket path is certain.

### Why did the champion change from Michigan to Duke?
Because:
- deterministic bracket = one path
- Monte Carlo = many paths

Michigan won the one baseline path, but Duke appeared most often across many simulated tournaments.
