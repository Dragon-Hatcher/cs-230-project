import tensorflow as tf
import numpy as np
import pandas as pd

SHOT_ID_DICT = {
    7: 1,  8: 2,  9: 3, 16: 4, 17: 5, 18: 6, 19: 7, 20: 8, 21: 9, 22: 10,
}
STONE_BEING_SHOT_DICT = {
    7: 2,  8: 8,  9: 3, 16: 9, 17: 4, 18: 10, 19: 5, 20: 11, 21: 6, 22: 12,
}


stones = pd.read_csv("dataset/Stones.csv")
ends = pd.read_csv("dataset/Ends.csv")

ends["PowerPlay"] = ends["PowerPlay"].fillna(0)
ends = ends.sort_values(by=["CompetitionID","SessionID","GameID","TeamID","EndID"])

ends["RunningScoreInclusive"] = ends.groupby(["CompetitionID","SessionID","GameID","TeamID"])["Result"].cumsum()
ends["RunningScoreExclusive"] = ends["RunningScoreInclusive"] - ends["Result"]

training_data = pd.DataFrame()

# [ 7  8  9 16 17 18 19 20 21 22]

stones["ShotID"] = stones["ShotID"].replace(SHOT_ID_DICT)
stones["StoneBeingShot"] = stones["ShotID"].replace(STONE_BEING_SHOT_DICT)
stones["IsHammer"] = stones["ShotID"] % 2 == 0

for i in range(1, 13):
    stones[f"stone_{i}_thrown"] = stones[f"stone_{i}_x"] > 0
    stones[f"ps_stone_{i}_thrown"] = stones[f"stone_{i}_thrown"] & (stones["StoneBeingShot"] != i)

for i in range(1, 7):
    for var in ["x", "y", "thrown"]:
        stones[f"ps_stone_shooter_{i}_{var}"] = \
            stones[f"stone_{i}_{var}"] * (1 - stones["IsHammer"]) + \
            stones[f"stone_{i+6}_{var}"] * stones["IsHammer"]
    for var in ["x", "y", "thrown"]:
        stones[f"ps_stone_opponent_{i}_{var}"] = \
            stones[f"stone_{i}_{var}"] * stones["IsHammer"] + \
            stones[f"stone_{i+6}_{var}"] * (1 - stones["IsHammer"])

    # stones[f"stone_{i}_on_ice"] = stones[f"stone_{i}_x"] != 4095

# stones = stones.join(ends, on=["CompetitionID","SessionID","GameID","TeamID","EndID"])
stones = pd.merge(stones, ends, how="left", on=["CompetitionID","SessionID","GameID","TeamID","EndID"])

training_data["EndID"] = stones["EndID"]
training_data["ShotID"] = stones["ShotID"]
training_data["IsHammer"] = stones["IsHammer"]
for i in range(1, 7):
    for var in ["x", "y", "thrown"]:
        training_data[f"ps_stone_shooter_{i}_{var}"] = stones[f"ps_stone_shooter_{i}_{var}"]
        training_data[f"ps_stone_opponent_{i}_{var}"] = stones[f"ps_stone_opponent_{i}_{var}"]
training_data["RunningScoreExclusive"] = stones["RunningScoreExclusive"]
training_data["PowerPlay"] = stones["PowerPlay"]

print(training_data)

# dataset = tf.data.TextLineDataset("dataset/Stones.csv")
# for element in dataset:
    # print(element)

print("Hello")

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

> Type
> Handle
> Points
"""