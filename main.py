from load_data import load_data
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

INPUT_COLS = [
    'EndID', 'ShotID', 'IsHammer',
    'ps_stone_shooter_1_x', 
    'ps_stone_opponent_1_x', 
    'ps_stone_shooter_1_y',
    'ps_stone_opponent_1_y', 
    'ps_stone_shooter_1_thrown',
    'ps_stone_opponent_1_thrown', 
    'ps_stone_shooter_2_x',
    'ps_stone_opponent_2_x', 
    'ps_stone_shooter_2_y',
    'ps_stone_opponent_2_y', 
    'ps_stone_shooter_2_thrown',
    'ps_stone_opponent_2_thrown', 
    'ps_stone_shooter_3_x',
    'ps_stone_opponent_3_x', 
    'ps_stone_shooter_3_y',
    'ps_stone_opponent_3_y', 
    'ps_stone_shooter_3_thrown',
    'ps_stone_opponent_3_thrown', 
    'ps_stone_shooter_4_x',
    'ps_stone_opponent_4_x',
    'ps_stone_shooter_4_y',
    'ps_stone_opponent_4_y',
    'ps_stone_shooter_4_thrown',
    'ps_stone_opponent_4_thrown', 
    'ps_stone_shooter_5_x',
    'ps_stone_opponent_5_x', 
    'ps_stone_shooter_5_y',
    'ps_stone_opponent_5_y', 
    'ps_stone_shooter_5_thrown',
    'ps_stone_opponent_5_thrown', 
    'ps_stone_shooter_6_x',
    'ps_stone_opponent_6_x', 
    'ps_stone_shooter_6_y',
    'ps_stone_opponent_6_y', 
    'ps_stone_shooter_6_thrown',
    'ps_stone_opponent_6_thrown', 
    'ShooterScore', 
    'ShooterPowerPlay',
    'OpponentScore', 
    'OpponentPowerPlay',
]
OUTPUT_COLS = [
    'ShotType_Draw', 'ShotType_Front',
    'ShotType_Guard', 'ShotType_Raise / Tap-back',
    'ShotType_Wick / Soft Peeling', 'ShotType_Freeze', 'ShotType_Take-out',
    'ShotType_Hit and Roll', 'ShotType_Clearing',
    'ShotType_Double Take-out', 'ShotType_Promotion Take-out',
    'ShotType_through'
]
# OUTPUT_COLS = [
#     'Quality'
# ]

NORMALIZE_COLS = [
    'ps_stone_shooter_1_x', 
    'ps_stone_opponent_1_x', 'ps_stone_shooter_1_y',
    'ps_stone_opponent_1_y', 'ps_stone_shooter_2_x',
    'ps_stone_opponent_2_x', 'ps_stone_shooter_2_y',
    'ps_stone_opponent_2_y', 'ps_stone_shooter_3_x',
    'ps_stone_opponent_3_x', 'ps_stone_shooter_3_y',
    'ps_stone_opponent_3_y', 'ps_stone_shooter_4_x',
    'ps_stone_opponent_4_x', 'ps_stone_shooter_4_y',
    'ps_stone_opponent_4_y', 'ps_stone_shooter_5_x',
    'ps_stone_opponent_5_x', 'ps_stone_shooter_5_y',
    'ps_stone_opponent_5_y', 'ps_stone_shooter_6_x',
    'ps_stone_opponent_6_x', 'ps_stone_shooter_6_y',
    'ps_stone_opponent_6_y',
]

scalar = StandardScaler()

for col in NORMALIZE_COLS:
    dataset[col] = scalar.fit_transform(dataset[[col]])
    # dataset[col] = dataset[[col]] / 4095

train_dataset = dataset[dataset["Group"] == "TRAIN"]
dev_dataset = dataset[dataset["Group"] == "DEV"]

# print()
# print(len(dev_dataset))
# for col in OUTPUT_COLS:
#     print(col, ":", dev_dataset[col].sum())

train_data_X = train_dataset[INPUT_COLS].to_numpy().astype('float32')
train_data_Y = train_dataset[OUTPUT_COLS].to_numpy().astype('float32')

dev_data_X = dev_dataset[INPUT_COLS].to_numpy().astype('float32')
dev_data_Y = dev_dataset[OUTPUT_COLS].to_numpy().astype('float32')

tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_data_X, train_data_Y)).batch(1024)
tf_dev_dataset = tf.data.Dataset.from_tensor_slices((dev_data_X, dev_data_Y)).batch(1024)

class_counts = np.array([train_dataset[col].sum() for col in OUTPUT_COLS])
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(train_data_X.shape[1],)),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(len(OUTPUT_COLS), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    tf_train_dataset,
    epochs=1000,
    validation_data=tf_dev_dataset,
    # class_weight=class_weight_dict
)

from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(pd.DataFrame(dev_data_Y[0:10,:]))
p = model.predict(dev_data_X[0:10,:])
print(pd.DataFrame(p))