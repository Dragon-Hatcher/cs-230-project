
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

# 57 ---ReLu-> 50 ---ReLu-> 30 ---softmax-> 12

def initialize_params():
    IN = 43
    L1 = 100
    L2 = 50
    OUT = 12

    initializer = tf.keras.initializers.GlorotNormal()
    W1 = tf.Variable(initializer(shape=(L1,IN)))
    b1 = tf.Variable(initializer(shape=(L1,1)))
    W2 = tf.Variable(initializer(shape=(L2,L1)))
    b2 = tf.Variable(initializer(shape=(L2,1)))
    W3 = tf.Variable(initializer(shape=(OUT,L2)))
    b3 = tf.Variable(initializer(shape=(OUT,1)))

    parameters = {"W1":W1, "W2":W2, "W3":W3, "b1":b1, "b2":b2, "b3":b3}

    return parameters

def forward_prop(X, params):
    W1 = params["W1"]
    W2 = params["W2"]
    W3 = params["W3"]
    b1 = params["b1"]
    b2 = params["b2"]
    b3 = params["b3"]

    Z1  = tf.math.add(tf.linalg.matmul(W1,X), b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2  = tf.math.add(tf.linalg.matmul(W2,A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3  = tf.math.add(tf.linalg.matmul(W3,A2), b3)

    return Z3

def compute_loss(preds, truth):
    return tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(truth), tf.transpose(preds), from_logits=True))


def model(train_data_X, train_data_Y, dev_data_X, dev_data_Y, learning_rate = 0.0001, num_steps = 10000):
    # print(train_data_X)
    # train_data_X = tf.convert_to_tensor(train_data_X)
    # train_data_Y = tf.convert_to_tensor(train_data_Y)
    # dev_data_X = tf.convert_to_tensor(dev_data_X)
    # dev_data_Y = tf.convert_to_tensor(dev_data_Y)

    costs = []
    train_acc = []
    dev_acc = []

    params = initialize_params()

    W1 = params["W1"]
    W2 = params["W2"]
    W3 = params["W3"]
    b1 = params["b1"]
    b2 = params["b2"]
    b3 = params["b3"]

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    dev_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(num_steps):
        with tf.GradientTape() as tape:
            Z3 = forward_prop(tf.transpose(train_data_X), params)
            loss = compute_loss(Z3, tf.transpose(train_data_Y))

        trainable_variables = [W1, b1, W2, b2, W3, b3]
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))

        train_accuracy.reset_state()
        train_accuracy.update_state(train_data_Y, tf.transpose(Z3))

        if epoch % 10 == 0:
            print()
            print("loss = ", loss.numpy())
            print("train_acc: ", train_accuracy.result().numpy())

            Z3 = forward_prop(tf.transpose(dev_data_X), params)
            dev_accuracy.update_state(dev_data_Y, tf.transpose(Z3))
            print("dev_acc: ", dev_accuracy.result().numpy())
            dev_accuracy.reset_state()
    
    return params, costs, train_acc, dev_acc

INPUT_COLS = [
    'EndID', 'ShotID', 'IsHammer',
    'ps_stone_shooter_1_x', 'ps_stone_opponent_1_x', 'ps_stone_shooter_1_y',
    'ps_stone_opponent_1_y', 'ps_stone_shooter_1_thrown',
    'ps_stone_opponent_1_thrown', 'ps_stone_shooter_2_x',
    'ps_stone_opponent_2_x', 'ps_stone_shooter_2_y',
    'ps_stone_opponent_2_y', 'ps_stone_shooter_2_thrown',
    'ps_stone_opponent_2_thrown', 'ps_stone_shooter_3_x',
    'ps_stone_opponent_3_x', 'ps_stone_shooter_3_y',
    'ps_stone_opponent_3_y', 'ps_stone_shooter_3_thrown',
    'ps_stone_opponent_3_thrown', 'ps_stone_shooter_4_x',
    'ps_stone_opponent_4_x', 'ps_stone_shooter_4_y',
    'ps_stone_opponent_4_y', 'ps_stone_shooter_4_thrown',
    'ps_stone_opponent_4_thrown', 'ps_stone_shooter_5_x',
    'ps_stone_opponent_5_x', 'ps_stone_shooter_5_y',
    'ps_stone_opponent_5_y', 'ps_stone_shooter_5_thrown',
    'ps_stone_opponent_5_thrown', 'ps_stone_shooter_6_x',
    'ps_stone_opponent_6_x', 'ps_stone_shooter_6_y',
    'ps_stone_opponent_6_y', 'ps_stone_shooter_6_thrown',
    'ps_stone_opponent_6_thrown', 'ShooterScore', 'ShooterPowerPlay',
    'OpponentScore', 'OpponentPowerPlay',
]
OUTPUT_COLS = [
     'ShotType_Draw', 'ShotType_Front',
    'ShotType_Guard', 'ShotType_Raise / Tap-back',
    'ShotType_Wick / Soft Peeling', 'ShotType_Freeze', 'ShotType_Take-out',
    'ShotType_Hit and Roll', 'ShotType_Clearing',
    'ShotType_Double Take-out', 'ShotType_Promotion Take-out',
    'ShotType_through'
]

train_dataset = dataset[dataset["Group"] == "TRAIN"]
dev_dataset = dataset[dataset["Group"] == "DEV"]

print()
print(len(dev_dataset))
for col in OUTPUT_COLS:
    print(col, ":", dev_dataset[col].sum())

train_data_X = train_dataset[INPUT_COLS].to_numpy().astype('float32')
train_data_Y = train_dataset[OUTPUT_COLS].to_numpy().astype('float32')

dev_data_X = dev_dataset[INPUT_COLS].to_numpy().astype('float32')
dev_data_Y = dev_dataset[OUTPUT_COLS].to_numpy().astype('float32')

model(train_data_X, train_data_Y, dev_data_X, dev_data_Y)

