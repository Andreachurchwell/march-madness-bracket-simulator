## Project Walkthrough Script

This project is a men's March Madness bracket simulator. The main goal is to use historical NCAA data, feature engineering, logistic regression, and eventually Monte Carlo simulation to predict matchups, identify possible sleeper teams, and simulate bracket outcomes.

The main dataset I used comes from the Kaggle dataset `march-machine-learning-mania-2026`. From that dataset, I focused on the men's files for teams, regular season results, tournament results, tournament seeds, and bracket slots. I also created my own `bracket_2026.csv` file from the released bracket so I could match the current tournament teams into the project.

The first thing I did was make sure the project setup worked, including the virtual environment, notebook kernel, and package structure. After that, I explored the Kaggle files to understand what each dataset was for and what keys I would need to join them together. The main join keys I ended up using were `TeamID` and `Season`.

From there, I built a team-season feature table using regular season results. The first features I created were wins, losses, win percentage, average points scored, average points allowed, and average scoring margin. Then I merged in team names so the feature table was easier to read and use.

After that, I cleaned the 2026 bracket teams and matched them to Kaggle team IDs. Some team names did not match exactly, so I had to normalize a few names manually. I also separated out the play-in rows until those games were resolved.

Once the teams were matched, I started turning games into matchup rows. Instead of just looking at one team by itself, I built feature differences between two teams, like win percentage difference, points for difference, points against difference, and scoring margin difference. That gave me one row per matchup, which is the structure I needed for modeling.

For the model, I used logistic regression as a baseline. I built historical matchup rows from past NCAA tournament games, using a neutral ordering for team A and team B so the model would not be biased by always putting the winner first. I trained the model on those historical matchup differences and used `team_a_won` as the target label.

The baseline logistic regression currently uses four matchup-difference features:

- win percentage difference
- average points scored difference
- average points allowed difference
- scoring margin difference

The baseline model got about 69% accuracy on held-out historical tournament games, with a log loss of about 0.58. I think that is a pretty solid starting point because it shows the current features already contain useful predictive signal.

After training the model, I applied it to the 2026 first-round matchups and generated round 1 win probabilities and predicted winners. The model also surfaced a few upset-style picks, which matters because one of the main goals of the project is to identify possible sleepers and underdogs instead of just following the bracket seeds.

After that, I started working on bracket advancement logic. One thing I learned there is that later rounds need to carry `TeamID` forward, not just team names. Round 1 worked more easily because it was still directly connected to the cleaned bracket table, but once I started advancing winners, I had to keep winner `TeamID`s attached so I could rebuild matchup features correctly in later rounds.

Using that process, I advanced all four regions through their regional finals, then built the Final Four and national championship game. The current baseline model's regional champions are `Duke`, `Gonzaga`, `Florida`, and `Michigan`. In the Final Four, the model has `Gonzaga` beating `Florida` and `Michigan` beating `Duke`, and the current baseline national champion is `Michigan`.

At this point, the project has a full deterministic bracket path from the opening round through the championship game. The next major step is Monte Carlo simulation so the project can move beyond one single bracket path and estimate things like championship odds, Final Four odds, upset frequency, and sleeper-team runs across many simulated tournaments.

## Short Version

I used the Kaggle March Madness dataset plus my own current bracket file, engineered team-season and matchup-level features, trained a baseline logistic regression model on historical NCAA tournament games, and got about 69% accuracy on held-out data. Then I used that model to generate 2026 first-round predictions, carried winners forward through the bracket using `TeamID`, and completed a full baseline bracket run through the national championship game, where the current predicted champion is Michigan. The next step is Monte Carlo simulation so I can estimate probabilities across many possible tournament outcomes instead of just one bracket path.

