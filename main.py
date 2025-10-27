
"""
EndID               x
ShotID
ShooterScore
OpponentScore
ShooterHasHammer
Stones
Powerplay

? TeamID
? PlayerID
? Timeout


"0": "Draw"
"1": "Front"
"2": "Guard"
"3": "Raise / Tap-back"
"4": "Wick / Soft Peeling"
"5": "Freeze"
"6": "Take-out"
"7": "Hit and Roll"
"8": "Clearing"
"9": "Double Take-out"
"10": "Promotion Take-out"
"11": "through"
"13": "no statistics"

> Type
> Handle
> Points
"""

from load_data import load_data
import tensorflow as tf
import numpy as np
import pandas as pd

# def make_assignment():
#     f = random.random()
#     if f < 0.8:
#         return "TRAIN"
#     elif f < 0.9:
#         return "DEV"
#     else:
#         return "TEST"

# all_games = dataset[["CompetitionID","SessionID","GameID"]].drop_duplicates()
# all_games["Group"] = [make_assignment() for _ in range(len(all_games)) ]
# print(all_games)

# all_games.to_csv("training_groups.csv")

dataset = load_data()
training_groups = pd.read_csv("training_groups.csv")
dataset = pd.merge(dataset, training_groups, how="left", on=["CompetitionID","SessionID","GameID"])

print(dataset)