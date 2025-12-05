from load_data import load_data
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

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
training_groups = pd.read_csv(r"C:\Users\gmjam\Downloads\training_groups.csv")
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

train_dataset = dataset[dataset["Group"] == "TRAIN"]
dev_dataset = dataset[dataset["Group"] == "DEV"]

# ---------- AUGMENTATION: JITTERING ----------

def flip_x_coordinates(df, x_cols, max_x=4095):

    df_flipped = df.copy()
    for col in x_cols:
        # Only flip non-special values if needed (e.g., not 0 or max_x)
        mask = (df_flipped[col] != 0) & (df_flipped[col] != max_x)
        df_flipped.loc[mask, col] = 1500 - df_flipped.loc[mask, col]
    return df_flipped

def permute_stones_vectorized(df, stone_count=6):
    """
    Vectorized permutation of stones 1..stone_count for both shooter and opponent.
    Returns a new DataFrame with permuted stone positions.
    """
    df_aug = df.copy()
    n_rows = len(df_aug)

    # Shooter stones
    shooter_cols_x = [f'ps_stone_shooter_{i}_x' for i in range(1, stone_count+1)]
    shooter_cols_y = [f'ps_stone_shooter_{i}_y' for i in range(1, stone_count+1)]
    shooter_cols_thrown = [f'ps_stone_shooter_{i}_thrown' for i in range(1, stone_count+1)]

    # Opponent stones
    opponent_cols_x = [f'ps_stone_opponent_{i}_x' for i in range(1, stone_count+1)]
    opponent_cols_y = [f'ps_stone_opponent_{i}_y' for i in range(1, stone_count+1)]
    opponent_cols_thrown = [f'ps_stone_opponent_{i}_thrown' for i in range(1, stone_count+1)]

    # Convert to numpy arrays
    shooter_x = df_aug[shooter_cols_x].to_numpy()
    shooter_y = df_aug[shooter_cols_y].to_numpy()
    shooter_thrown = df_aug[shooter_cols_thrown].to_numpy()

    opponent_x = df_aug[opponent_cols_x].to_numpy()
    opponent_y = df_aug[opponent_cols_y].to_numpy()
    opponent_thrown = df_aug[opponent_cols_thrown].to_numpy()

    # Generate random permutations for each row
    shooter_perms = np.array([np.random.permutation(stone_count) for _ in range(n_rows)])
    opponent_perms = np.array([np.random.permutation(stone_count) for _ in range(n_rows)])

    # Apply permutations
    for i, col in enumerate(shooter_cols_x):
        df_aug[col] = shooter_x[np.arange(n_rows), shooter_perms[:, i]]
    for i, col in enumerate(shooter_cols_y):
        df_aug[col] = shooter_y[np.arange(n_rows), shooter_perms[:, i]]
    for i, col in enumerate(shooter_cols_thrown):
        df_aug[col] = shooter_thrown[np.arange(n_rows), shooter_perms[:, i]]

    for i, col in enumerate(opponent_cols_x):
        df_aug[col] = opponent_x[np.arange(n_rows), opponent_perms[:, i]]
    for i, col in enumerate(opponent_cols_y):
        df_aug[col] = opponent_y[np.arange(n_rows), opponent_perms[:, i]]
    for i, col in enumerate(opponent_cols_thrown):
        df_aug[col] = opponent_thrown[np.arange(n_rows), opponent_perms[:, i]]

    return df_aug


JITTER_COLS = [c for c in INPUT_COLS if c.endswith("_x") or c.endswith("_y")]

jittered = train_dataset.copy()


JITTER_AMOUNT = 5

for col in JITTER_COLS:
    # Only add jitter where the value is not a special case (0 or 4095)
    mask = (jittered[col] != 0) & (jittered[col] != 4095)
    jittered.loc[mask, col] += np.random.uniform(-JITTER_AMOUNT, JITTER_AMOUNT, size=mask.sum())

augmented_train = permute_stones_vectorized(train_dataset)
flipped = flip_x_coordinates(train_dataset, [c for c in INPUT_COLS if c.endswith("_x")])

# Concatenate with original training data
#train_dataset = pd.concat([train_dataset, flipped], ignore_index=True)
#train_dataset = pd.concat([train_dataset, jittered], ignore_index=True)
#train_dataset = pd.concat([train_dataset, augmented_train], ignore_index=True)
print(train_dataset.shape)

scalar = StandardScaler()
for col in NORMALIZE_COLS:
    train_dataset[col] = scalar.fit_transform(train_dataset[[col]])
    dev_dataset[col] = scalar.transform(dev_dataset[[col]])




train_data_X = train_dataset[INPUT_COLS].to_numpy().astype('float32')
train_data_Y = train_dataset[OUTPUT_COLS].to_numpy().astype('float32')

dev_data_X = dev_dataset[INPUT_COLS].to_numpy().astype('float32')
dev_data_Y = dev_dataset[OUTPUT_COLS].to_numpy().astype('float32')

tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_data_X, train_data_Y)).batch(1024)
tf_dev_dataset = tf.data.Dataset.from_tensor_slices((dev_data_X, dev_data_Y)).batch(1024)

class_counts = np.array([train_dataset[col].sum() for col in OUTPUT_COLS])
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}


# Amount of L2 regularization
L2_DECAY = 0.00
DROPOUT_RATE = 0.2
model = tf.keras.Sequential([
    tf.keras.Input(shape=(train_data_X.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu',
                          kernel_regularizer=regularizers.l2(L2_DECAY)),
    BatchNormalization(),  # <-- Normalize before activation
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(80, activation='relu',
                          kernel_regularizer=regularizers.l2(L2_DECAY)),
    BatchNormalization(),  # <-- Normalize before activation
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(50, activation='relu',
                          kernel_regularizer=regularizers.l2(L2_DECAY)),
    BatchNormalization(),  # <-- Normalize before activation
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(30, activation='relu',
                          kernel_regularizer=regularizers.l2(L2_DECAY)),
    BatchNormalization(),  # <-- Normalize before activation
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(len(OUTPUT_COLS), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    tf_train_dataset,
    epochs=2000,
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