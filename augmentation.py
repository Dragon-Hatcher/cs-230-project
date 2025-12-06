from load_data import load_data
from augmentations import augment_data

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random

def make_assignment():
    f = random.random()
    if f < 0.8:
        return "TRAIN"
    elif f < 0.9:
        return "DEV"
    else:
        return "TEST"

def assign_groups(dataset):
    """
    Assign each game in the dataset to the training, dev, or test set. Save
    the results to a file. (This should only be run once).
    """
    all_games = dataset[["CompetitionID","SessionID","GameID"]].drop_duplicates()
    all_games["Group"] = [make_assignment() for _ in range(len(all_games)) ]
    all_games.to_csv("training_groups.csv")

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


# Load the dataset
dataset = load_data()

# Split our dataset into the different groups. We load the groups from a file
# so they are consistent run to run.
training_groups = pd.read_csv("training_groups.csv")
dataset = pd.merge(dataset, training_groups, how="left", on=["CompetitionID","SessionID","GameID"])

train_dataset = dataset[dataset["Group"] == "TRAIN"]
dev_dataset = dataset[dataset["Group"] == "DEV"]
test_dataset = dataset[dataset["Group"] == "TEST"]

# Perform data augmentation.
train_dataset = augment_data(
    train_dataset,
    input_cols=INPUT_COLS,
    augmentations=[
        # No augmentations right now.
        # "flip",
        # "jitter",
        # "permute"
    ]
)

# Perform data normalization
scalar = StandardScaler()
NORMALIZE_COLS = [c for c in INPUT_COLS if c.endswith("_x") or c.endswith("_y")]
for col in NORMALIZE_COLS:
    train_dataset[col] = scalar.fit_transform(train_dataset[[col]])
    dev_dataset[col] = scalar.transform(dev_dataset[[col]])
    test_dataset[col] = scalar.transform(test_dataset[[col]])

# Prepare the datasets for use by the model.
train_data_X = train_dataset[INPUT_COLS].to_numpy().astype('float32')
train_data_Y = train_dataset[OUTPUT_COLS].to_numpy().astype('float32')

dev_data_X = dev_dataset[INPUT_COLS].to_numpy().astype('float32')
dev_data_Y = dev_dataset[OUTPUT_COLS].to_numpy().astype('float32')

test_data_X = test_dataset[INPUT_COLS].to_numpy().astype('float32')
test_data_Y = test_dataset[OUTPUT_COLS].to_numpy().astype('float32')

tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_data_X, train_data_Y)).batch(1024)
tf_dev_dataset = tf.data.Dataset.from_tensor_slices((dev_data_X, dev_data_Y)).batch(1024)

def get_random_hyperparameters():
    layers = [int(10 ** (1 + random.random())) for _ in range(random.randint(1,2))]

    return {
        "layers": layers,
        "learning_rate": 10 ** (random.random() * -8),
        "minibatch_size": 2 ** random.randint(0, 14),
        "l2_decay": 0.0,
        "dropout_rate": 0.2,
        "use_class_counts": False,
    }

def get_standard_hyperparameters():
    return {
        "layers": [100, 90, 50, 30],
        "learning_rate": 0.01,
        "minibatch_size": 1024,
        "l2_decay": 0.0,
        "dropout_rate": 0.2,
        "use_class_counts": False,
    }

class_counts = np.array([train_dataset[col].sum() for col in OUTPUT_COLS])
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

def train_model(hyperparams):
    # First hidden layer takes from the input
    layers = [tf.keras.Input(shape=(train_data_X.shape[1],))]
    
    # Rest of the hidden layers
    for layer in hyperparams["layers"]:
        layers.append(tf.keras.layers.Dense(
            layer, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(hyperparams["l2_decay"])
        )),
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation('relu'))
        layers.append(tf.keras.layers.Dropout(hyperparams["dropout_rate"]))

    # Softmax output layer
    layers.append(tf.keras.layers.Dense(len(OUTPUT_COLS), activation='softmax'))

    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"]),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    mb = hyperparams["minibatch_size"]
    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_data_X, train_data_Y)).batch(mb)
    tf_dev_dataset = tf.data.Dataset.from_tensor_slices((dev_data_X, dev_data_Y)).batch(mb)

    print(f"Training model: {hyperparams}")
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
        class_weight=class_weight_dict if hyperparams["use_class_counts"] else None,
    )

    best_dev_accuracy = max(history.history['val_accuracy'])
    print(f"Finished training. Accuracy {best_dev_accuracy}")

    return model, history, best_dev_accuracy, hyperparams

def plot_model_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def evaluate_model(model):
    p = model.predict(test_data_X)
    
    # ---------------------------------------
    # Convert predictions → class indices
    # ---------------------------------------
    y_pred = np.argmax(p, axis=1)
    y_true = np.argmax(test_data_Y, axis=1)

    # ---------------------------------------
    # Raw confusion matrix
    # ---------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    print("Raw Confusion Matrix:")
    print(cm)

    # ---------------------------------------
    # Normalized confusion matrix (row-wise)
    # ---------------------------------------
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    print("\nNormalized Confusion Matrix (rows sum to 1):")
    print(cm_normalized)

    # ---------------------------------------
    # Classification report (precision, recall, f1)
    # ---------------------------------------
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=OUTPUT_COLS, digits=4))


    # ---------------------------------------
    # Plot Raw Confusion Matrix
    # ---------------------------------------
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=OUTPUT_COLS,
                yticklabels=OUTPUT_COLS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Raw Counts)")
    plt.tight_layout()
    plt.show()


    # ---------------------------------------
    # Plot Normalized Confusion Matrix
    # ---------------------------------------
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=OUTPUT_COLS,
                yticklabels=OUTPUT_COLS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.show()

hyperparams = get_standard_hyperparameters()
model, history, _, _ = train_model(hyperparams)
plot_model_history(history)
evaluate_model(model)