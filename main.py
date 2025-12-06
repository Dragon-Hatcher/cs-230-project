from load_data import load_data
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

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

print()
print(len(train_dataset))
for col in OUTPUT_COLS:
    print(col, ":", train_dataset[col].sum())

train_data_X = train_dataset[INPUT_COLS].to_numpy().astype('float32')
train_data_Y = train_dataset[OUTPUT_COLS].to_numpy().astype('float32')

dev_data_X = dev_dataset[INPUT_COLS].to_numpy().astype('float32')
dev_data_Y = dev_dataset[OUTPUT_COLS].to_numpy().astype('float32')


class_counts = np.array([train_dataset[col].sum() for col in OUTPUT_COLS])
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

def make_hyperparams():
    layers = [int(10 ** (1 + random.random())) for _ in range(random.randint(1,2))]

    return {
        "layers": layers,
        "learning_rate": 10 ** (random.random() * -8),
        "minibatch_size": 2 ** random.randint(0, 14),
    }

def make_model(hyperparams, percent):
    # First hidden layer takes from the input
    layers = [tf.keras.layers.Dense(hyperparams["layers"][0], activation='relu', input_shape=(train_data_X.shape[1],))]
    
    # Rest of the hidden layers
    for layer in hyperparams["layers"][1:]:
        layers.append(tf.keras.layers.Dense(layer, activation='relu'))

    # Softmax output layer
    layers.append(tf.keras.layers.Dense(len(OUTPUT_COLS), activation='softmax'))

    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"]),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    subset_size = int(len(train_data_X) * percent)
    indices = np.random.choice(len(train_data_X), subset_size, replace=False)
    train_X = train_data_X[indices]
    train_Y = train_data_Y[indices]

    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(1024)
    tf_dev_dataset = tf.data.Dataset.from_tensor_slices((dev_data_X, dev_data_Y)).batch(1024)

    print(f"Trying {hyperparams}")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=50,
        mode='max',
        restore_best_weights = True,
    )

    history = model.fit(
        tf_train_dataset,
        epochs=1000,
        validation_data=tf_dev_dataset,
        callbacks=[early_stop],
        verbose=0,
        # class_weight=class_weight_dict
    )

    best_dev_accuracy = max(history.history['val_accuracy'])
    print(f"Trained => {best_dev_accuracy}")

    return history, best_dev_accuracy, hyperparams

from matplotlib import pyplot as plt

BEST_LR = 0.03

x = []
y = []
for mb in range(0, 14):
    for _ in range(2):
        new_history, new_accuracy, new_hypers = make_model({
            "layers": [100, 80, 50, 30],
            "learning_rate": BEST_LR,
            "minibatch_size": 2 ** mb,
        })
        x.append( mb)
        y.append(new_accuracy)

# plt.plot(x)
# plt.plot(y)
print(x, y)
plt.scatter(x, y)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('log minibatch size')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# history = accuracy = hypers = None
# while True:
#     new_history, new_accuracy, new_hypers = make_model(make_hyperparams())
#     if accuracy is None or new_accuracy > accuracy:
#         history = new_history
#         accuracy = new_accuracy
#         hypers = new_hypers

#         print(f"!! New best model: {hypers} Has accuracy: {accuracy}")

        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        # plt.show(block=False)

# print(pd.DataFrame(dev_data_Y[0:10,:]))
# p = model.predict(dev_data_X[0:10,:])
# print(pd.DataFrame(p))