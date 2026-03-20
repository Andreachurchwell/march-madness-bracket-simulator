## Project Walkthrough Script

This project is called **Andrea's Bracket Breakdown**. It is a men's March Madness bracket simulator built with historical NCAA data, feature engineering, logistic regression, and Monte Carlo simulation.

The main dataset comes from the Kaggle dataset `march-machine-learning-mania-2026`. From that dataset, I used the men's team file, regular season results, historical tournament results, tournament seeds, and bracket slots. I also created my own `bracket_2026.csv` file so I could map the current tournament field into the project.

The first thing I did was make sure the environment, package structure, and notebook setup all worked. After that, I explored the Kaggle files to understand what each one was doing and what keys I would need to join them together. The main join keys I ended up using were `TeamID` and `Season`.

From there, I built a team-season feature table using regular season results. The main features I created were wins, losses, win percentage, average points scored, average points allowed, and average scoring margin. Then I merged team names into that table so it was easier to read and use.

After that, I cleaned the 2026 bracket teams and matched them to Kaggle team IDs. Some team names did not match exactly, so I had to normalize several names manually. Once the teams were matched, I started turning games into matchup rows by building feature differences between the two teams in each game.

For the model, I used logistic regression as a baseline. I built historical matchup rows from past NCAA tournament games using a neutral team A and team B ordering so the model would not always see the winner on the same side. The baseline model uses four matchup-difference features:

- win percentage difference
- average points scored difference
- average points allowed difference
- scoring margin difference

That baseline logistic regression got about 69% accuracy on held-out historical tournament games, with a log loss of about 0.58. I used that model to generate first-round win probabilities for the 2026 bracket and then started advancing winners through later rounds.

One of the biggest things I learned was that later rounds need to carry `TeamID` forward, not just display names. Round 1 worked more easily because it was still directly tied to the cleaned bracket table, but once I started advancing winners, I had to keep the winner `TeamID` attached so I could rebuild features correctly in the next round.

Using that process, I advanced all four regions through their regional finals, then built the Final Four and national championship game. The current deterministic baseline bracket has:

- East champion: `Duke`
- West champion: `Gonzaga`
- South champion: `Florida`
- Midwest champion: `Michigan`

In that deterministic bracket, the final national champion was `Michigan`.

After that, I moved on to Monte Carlo simulation. Instead of always advancing the higher-probability team, Monte Carlo uses the model's probabilities to simulate the entire tournament many times. I built reusable simulation functions in the project package and ran 1000 full tournament simulations.

Those simulations showed a different picture than the single deterministic bracket. Even though the deterministic bracket picked `Michigan`, the most common simulated champion was `Duke` at about 17.5%, followed by `Gonzaga`, `Michigan`, and `Arizona`. The simulations also highlighted teams like `Saint Louis` as interesting sleeper-style outcomes because they appeared more often than I would expect from seed alone.

I also moved the reusable simulation logic out of the notebook and into the Python package so the app and the rest of the project could use the same code. Then I updated the Streamlit app to show a deterministic bracket view, a consensus simulation view, championship odds, Final Four odds, regional win odds, upset-watch charts, and a one-click random tournament run.

At this point, the project can show both a single baseline bracket path and a probability-based view of the tournament across many simulations. The next things I would improve are clearer sleeper definitions, deeper round summaries beyond the title race, and continued polish on the app layout.

## Short Version

I used the Kaggle March Madness dataset plus my own current bracket file, engineered team-season and matchup-level features, trained a baseline logistic regression model on historical NCAA tournament games, and got about 69% accuracy on held-out data. Then I used that model to build a full deterministic bracket, where the current baseline champion is Michigan. After that, I added Monte Carlo simulation and ran 1000 full tournament simulations, which showed Duke as the most common simulated national champion. The project now includes both reusable simulation code and a Streamlit app that shows the bracket and simulation results.
