import numpy as np
import pandas as pd

# The shots in the dataset are ordered in a weird way.
SHOT_ID_DICT = {
    7: 1,  8: 2,  9: 3, 16: 4, 17: 5, 18: 6, 19: 7, 20: 8, 21: 9, 22: 10,
}
STONE_BEING_SHOT_DICT = {
    7: 2,  8: 8,  9: 3, 16: 9, 17: 4, 18: 10, 19: 5, 20: 11, 21: 6, 22: 12,
}

# Convert the shot type names to numerical codes for one-hot encoding.
SHOT_TYPES_DICT = {
    "Draw": 0,
    "Front": 1,
    "Guard": 2,
    "Raise / Tap-back": 3,
    "Wick / Soft Peeling": 4,
    "Freeze": 5,
    "Take-out": 6,
    "Hit and Roll": 7,
    "Clearing": 8,
    "Double Take-out": 9,
    "Promotion Take-out": 10,
    "through": 11,
}

def load_data():
    stones = pd.read_csv("dataset/Stones.csv")
    ends = pd.read_csv("dataset/Ends.csv")

    # Remove NaNs
    stones = stones.drop("TimeOut", axis=1)
    stones = stones.drop(stones[stones["Task"] == -1].index)

    assert(np.isnan(stones.to_numpy()).sum() == 0)
    assert((stones["Task"] == 13).sum() == 0)

    ends["PowerPlay"] = ends["PowerPlay"].fillna(0)

    # Compute the total score in the match at the start of each end.
    ends = ends.sort_values(by=["CompetitionID","SessionID","GameID","TeamID","EndID"])
    ends["RunningScoreInclusive"] = ends.groupby(["CompetitionID","SessionID","GameID","TeamID"])["Result"].cumsum()
    ends["RunningScoreExclusive"] = ends["RunningScoreInclusive"] - ends["Result"]

    # Fix some of the weirdness in how the data is provided.
    stones["ShotID"] = stones["ShotID"].replace(SHOT_ID_DICT)
    stones["StoneBeingShot"] = stones["ShotID"].replace(STONE_BEING_SHOT_DICT)
    stones["IsHammer"] = stones["ShotID"] % 2 == 0

    # The data download provides the stones from the perspective of the team
    # that shoots first and the team that shoots second. However we want the
    # data from the perspective of the team that is currently shooting and the
    # team that is not currently shooting. This code reformats the data in this
    # way
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

    stones = pd.merge(stones, ends, how="left", on=["CompetitionID","SessionID","GameID","TeamID","EndID"])

    # Now we begin to assemble the training data.
    training_data = pd.DataFrame()

    # ------------ Ids -------------------------
    training_data["CompetitionID"] = stones["CompetitionID"]
    training_data["SessionID"] = stones["SessionID"]
    training_data["GameID"] = stones["GameID"]

    # ------------ X Values ---------------------

    training_data["EndID"] = stones["EndID"]
    training_data["ShotID"] = stones["ShotID"]
    training_data["IsHammer"] = stones["IsHammer"]
    for i in range(1, 7):
        for var in ["x", "y", "thrown"]:
            training_data[f"ps_stone_shooter_{i}_{var}"] = stones[f"ps_stone_shooter_{i}_{var}"]
            training_data[f"ps_stone_opponent_{i}_{var}"] = stones[f"ps_stone_opponent_{i}_{var}"]
    training_data["ShooterScore"] = stones["RunningScoreExclusive"]
    training_data["ShooterPowerPlay"] = stones["PowerPlay"]

    up = stones["RunningScoreExclusive"].shift(1)
    down = stones["RunningScoreExclusive"].shift(-1)
    training_data["OpponentScore"] = np.where((stones.index + 1) % 10 == 0, up, down)

    up = stones["PowerPlay"].shift(1)
    down = stones["PowerPlay"].shift(-1)
    training_data["OpponentPowerPlay"] = np.where((stones.index + 1) % 10 == 0, up, down)

    # ------------ Y Values ---------------------
    for name, val in SHOT_TYPES_DICT.items():
        training_data[f"ShotType_{name}"] = stones["Task"] == val
    for i in range(5):
        training_data[f"Quality_{i}"] = stones["Points"] == i
    training_data["Quality"] = stones['Points'] / 4

    return training_data.dropna()
