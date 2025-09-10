import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1. Parser for Retrosheet Game Logs
# -----------------------------
def load_retro_gl(filename: str) -> pd.DataFrame:
    usecols = [
        0, 3, 6, 9, 10, 19, 20, 23, 24, 29, 30, 31, 32
    ]
    colnames = [
        "date", "home_team", "away_team",
        "away_runs", "home_runs",
        "away_hits", "home_hits",
        "away_hr", "home_hr",
        "away_bb", "home_bb",
        "away_so", "home_so"
    ]
    
    df = pd.read_csv(
        filename,
        header=None,
        names=colnames,
        usecols=usecols
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    df["home_hits"] = df["home_hits"].astype(str).str.replace("x", "", case=False, regex=False)
    df["home_hits"] = pd.to_numeric(df["home_hits"], errors="coerce")

    #Convert the other numeric columns
    numeric_cols = [
        "away_runs", "home_runs", "away_hits",
        "away_hr", "home_hr",
        "away_bb", "home_bb",
        "away_so", "home_so"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    home_df = pd.DataFrame({
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "home_away": 1,
        "runs_scored": df["home_runs"],
        "runs_allowed": df["away_runs"],
        "hits": df["home_hits"],
        "hr": df["home_hr"],
        "bb": df["home_bb"],
        "so": df["home_so"],
    })

    away_df = pd.DataFrame({
        "date": df["date"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "home_away": 0,
        "runs_scored": df["away_runs"],
        "runs_allowed": df["home_runs"],
        "hits": df["away_hits"],
        "hr": df["away_hr"],
        "bb": df["away_bb"],
        "so": df["away_so"],
    })

    return pd.concat([home_df, away_df], ignore_index=True)


# -----------------------------
# 2. Load all seasons
# -----------------------------
all_files = glob.glob("gl202*.txt")
dfs = [load_retro_gl(f) for f in all_files]
mlb_data = pd.concat(dfs, ignore_index=True)

# print("Total rows:", len(mlb_data))

# -----------------------------
# 3. Rolling averages (form)
# -----------------------------
cols = ["runs_scored", "runs_allowed", "hits", "hr", "bb", "so"]
new_cols = [f"{c}_rolling" for c in cols]


def rolling_averages(group, cols, new_cols, window=10):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(window, closed='left').mean()
    group[new_cols] = rolling_stats
    return group.dropna(subset=new_cols)


mlb_data_rolling = (
    mlb_data.groupby("team", group_keys=False)
    [["team", "date", "opponent", "home_away"] + cols]
    .apply(lambda x: rolling_averages(x, cols, new_cols, window=10))
    .reset_index(drop=True)
)


mlb_data_rolling["opp_code"] = mlb_data_rolling["opponent"].astype("category").cat.codes
mlb_data_rolling["day"] = mlb_data_rolling["date"].dt.dayofweek

# -----------------------------
# 4. Train/Test Split
# -----------------------------
train = mlb_data_rolling[mlb_data_rolling["date"] < "2024-01-01"]
test = mlb_data_rolling[mlb_data_rolling["date"] >= "2024-01-01"]

predictors = ["home_away", "opp_code", "day"] + new_cols

rf_reg = RandomForestRegressor(n_estimators=200, random_state=1)
rf_reg.fit(train[predictors], train["runs_scored"])


# -----------------------------
# 5. Prediction Function
# -----------------------------
def predict_game(home_team: str, away_team: str, date: str = None):
    """
    Predict score for a matchup between home_team and away_team.
    Input teams as 3-letter codes (e.g. 'lan', 'sdn').
    """
    date = pd.to_datetime(date) if date else mlb_data_rolling["date"].max() + pd.Timedelta(days=1)

    features = []
    teams = [(home_team.upper(), away_team.upper(), 1), 
             (away_team.upper(), home_team.upper(), 0)]

    for team, opp, home_away in teams:
        team_data = mlb_data_rolling[mlb_data_rolling["team"] == team].sort_values("date")
        if team_data.empty:
            raise ValueError(f"No data for team {team}")

        last_row = team_data.iloc[-1]

        row = {
            "home_away": home_away,
            "opp_code": mlb_data_rolling["opponent"].astype("category").cat.categories.get_loc(opp),
            "day": date.dayofweek,
        }
        for c in new_cols:
            row[c] = last_row[c]
        features.append(row)

    X = pd.DataFrame(features, index=[home_team.upper(), away_team.upper()])
    preds = rf_reg.predict(X)

    home_score, away_score = preds[0], preds[1]
    winner = home_team.upper() if home_score > away_score else away_team.upper()

    return {
        "home": home_team.upper(),
        "away": away_team.upper(),
        "home_score": round(home_score, 1),
        "away_score": round(away_score, 1),
        "winner": winner
    }


# -----------------------------
# 6. Prediction
# -----------------------------
home_team = input('Home team (3-letter code, lowercase):')
away_team = input('Away team (3-letter code, lowercase):')
result = predict_game(home_team, away_team)
print(result)


# -----------------------------
# 7. Print MAE
# -----------------------------
preds = rf_reg.predict(test[predictors])
mae = mean_absolute_error(test["runs_scored"], preds)
print("Mean Absolute Error (runs):", round(mae, 2))