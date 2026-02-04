# MLB Predictor

![MLB Logo](mlb_logo.png)

Predict MLB game scores using historical game logs and a machine learning model.

## Description

This project predicts the number of runs scored by each team in a Major League Baseball game using historical Retrosheet game logs. The model calculates rolling averages of team statistics such as runs, hits, home runs, walks, and strikeouts over the last 10 games and combines this with home/away information and opponent codes to predict scores.

Users can run the program to input two teams (home and away) and receive a predicted score along with the likely winner. The historical data used in this project comes from Retrosheet
, a comprehensive archive of MLB game logs

## Getting Started

### Dependencies

* Python 3.12+
* Libraries: 
```python
pip install pandas scikit-learn
```
* Operating System: Tested on Windows 11

### Installing

1. Clone or download this repository:
```bash
git clone https://github.com/m-zack-k/MLB_Prediction.git

```
2. Ensure that the Retrosheet game log .txt files (gl202*.txt) are placed in the project folder.
3. No further modifications are required; the script reads all game log files automatically!!

### Executing program

1. Open a terminal or command prompt in the project folder.
2. Run the predictor: 
```bash
python MLB_Predictor.py
```
3. Enter the home and away teams as 3-letter codes in lowercase (e.g., lan for Los Angeles Dodgers, sfn for San Francisco Giants).
4. For a complete list of team codes, refer to **team_codes.txt**

## Help

If you encounter issues:
*Make sure all required Python packages are installed:
```bash
pip install pandas scikit-learn
```
* Ensure Retrosheet .txt files are in the project directory.
* Team codes ***must be 3-letter lowercase!***
* For debugging, check for missing values in rolling stats or corrupted input files.

## Author

Kosuke Miyazaki

## License

This project is licensed under the MIT License.

