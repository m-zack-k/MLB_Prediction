import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1. Parser for Retrosheet Game Logs
# -----------------------------
def load_retro_gl(filename: str) -> pd.DataFrame:
    usecols = [0, 3, 6, 9, 10, 19, 20, 23, 24, 29, 30, 31, 32]
    colnames = [
        "date", "home_team", "away_team",
        "away_runs", "home_runs",
        "away_hits", "home_hits",
        "away_hr", "home_hr",
        "away_bb", "home_bb",
        "away_so", "home_so"
    ]
    
    df = pd.read_csv(filename, header=None, names=colnames, usecols=usecols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    df["home_hits"] = df["home_hits"].astype(str).str.replace("x", "", case=False, regex=False)
    df["home_hits"] = pd.to_numeric(df["home_hits"], errors="coerce")

    # Convert numeric columns
    numeric_cols = [
        "away_runs", "home_runs", "away_hits",
        "away_hr", "home_hr", "away_bb", "home_bb", "away_so", "home_so"
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
# 2. Load Data & Create Rolling Stats
# -----------------------------
all_files = glob.glob("gl202*.txt")
dfs = [load_retro_gl(f) for f in all_files]
mlb_data = pd.concat(dfs, ignore_index=True)

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

# -----------------------------
# 3. Feature Engineering (The Major Upgrade)
# -----------------------------
# A. Join Opponent Stats (Defense)
opp_stats = mlb_data_rolling[["date", "team"] + new_cols].copy()
opp_stats.columns = ["date", "opponent"] + [f"opp_{c}" for c in new_cols]

mlb_data_final = pd.merge(
    mlb_data_rolling,
    opp_stats,
    on=["date", "opponent"],
    how="left"
)
mlb_data_final = mlb_data_final.dropna()

# B. Add Time Features
mlb_data_final["day"] = mlb_data_final["date"].dt.dayofweek

# C. Add Team IDs (Class) - CRITICAL STEP
# We map "LAN" to a number so the model learns that LAN is a strong team.
unique_teams = mlb_data_final["team"].unique()
team_map = {team: i for i, team in enumerate(unique_teams)}

mlb_data_final["team_code"] = mlb_data_final["team"].map(team_map)
mlb_data_final["opp_code"] = mlb_data_final["opponent"].map(team_map)

# -----------------------------
# 4. Training
# -----------------------------
# Updated Predictors list including Team IDs
predictors = ["home_away", "day", "team_code", "opp_code"] + new_cols + [f"opp_{c}" for c in new_cols]

train = mlb_data_final[mlb_data_final["date"] < "2024-01-01"]
test = mlb_data_final[mlb_data_final["date"] >= "2024-01-01"]

# Initialize Model
# min_samples_leaf=5 prevents overfitting (memorizing specific games)
rf_reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=1)
rf_reg.fit(train[predictors], train["runs_scored"])

# -----------------------------
# 5. Evaluation
# -----------------------------
test_eval = test.copy()
test_preds = rf_reg.predict(test[predictors])
test_eval['pred_runs_scored'] = test_preds

# Merge predictions to compare Head-to-Head
test_eval = test_eval.merge(
    test_eval[['date', 'team', 'pred_runs_scored']], 
    left_on=['date', 'opponent'], 
    right_on=['date', 'team'], 
    suffixes=('', '_opp')
)

test_eval['actual_win'] = (test_eval['runs_scored'] > test_eval['runs_allowed']).astype(int)
test_eval['pred_win'] = (test_eval['pred_runs_scored'] > test_eval['pred_runs_scored_opp']).astype(int)

accuracy = (test_eval['actual_win'] == test_eval['pred_win']).mean()
mae = mean_absolute_error(test["runs_scored"], test_preds)

print(f"Win Prediction Accuracy: {accuracy:.2%}")
print(f"Mean Absolute Error (runs): {mae:.2f}")

# -----------------------------
# 6. Prediction Function
# -----------------------------
def predict_game(home_team: str, away_team: str, date: str = None):
    home_team = home_team.upper()
    away_team = away_team.upper()
    
    if date:
        date_obj = pd.to_datetime(date)
    else:
        date_obj = pd.Timestamp.now()

    # Get data
    home_data = mlb_data_rolling[mlb_data_rolling["team"] == home_team].sort_values("date")
    away_data = mlb_data_rolling[mlb_data_rolling["team"] == away_team].sort_values("date")

    if home_data.empty or away_data.empty:
        raise ValueError("Team not found or no data available.")

    home_latest = home_data.iloc[-1]
    away_latest = away_data.iloc[-1]

    # Helper to build row
    def build_row(team_code_str, opp_code_str, is_home, stats_own, stats_opp):
        row = {
            "home_away": 1 if is_home else 0,
            "day": date_obj.dayofweek,
            "team_code": team_map[team_code_str],
            "opp_code": team_map[opp_code_str]
        }
        for c in new_cols:
            row[c] = stats_own[c]         # Offense stats
            row[f"opp_{c}"] = stats_opp[c] # Defense stats (opponent's rolling)
        return row

    # Build rows
    home_features = build_row(home_team, away_team, True, home_latest, away_latest)
    away_features = build_row(away_team, home_team, False, away_latest, home_latest)

    # DataFrame and Predict
    X_new = pd.DataFrame([home_features, away_features], index=[home_team, away_team])
    X_new = X_new[predictors] # Ensure correct column order

    preds = rf_reg.predict(X_new)
    
    home_score = preds[0]
    away_score = preds[1]
    winner = home_team if home_score > away_score else away_team

    return {
        "home": home_team,
        "away": away_team,
        "home_score": round(home_score, 2),
        "away_score": round(away_score, 2),
        "winner": winner
    }

# -----------------------------
# 7. User Interface
# -----------------------------
home_input = input('Home team (3-letter code, lowercase): ')
away_input = input('Away team (3-letter code, lowercase): ')

if home_input and away_input:
    try:
        result = predict_game(home_input, away_input)
        print("\nPrediction Result:")
        print(f"{result['home']} ({result['home_score']}) vs {result['away']} ({result['away_score']})")
        print(f"Predicted Winner: {result['winner']}")
    except Exception as e:
        print(f"Error: {e}")