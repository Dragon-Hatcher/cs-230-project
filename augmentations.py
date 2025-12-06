import numpy as np
import pandas as pd

def flip_x_coordinates(df, x_cols, max_x=4095):
    """
    Flip the x coordinates of each stone in the dataset.    
    """
    df_flipped = df.copy()
    for col in x_cols:
        # Only flip non-special values if needed (e.g., not 0 or max_x)
        mask = (df_flipped[col] != 0) & (df_flipped[col] != max_x)
        df_flipped.loc[mask, col] = 1500 - df_flipped.loc[mask, col]
    return df_flipped

def permute_stones(df, stone_count=6):
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

def jitter(df, cols):
    """
    Add a random offset to each stone in the dataset.
    """
    jittered = df.copy()

    JITTER_AMOUNT = 5
    for col in cols:
        # Only add jitter where the value is not a special case (0 or 4095)
        mask = (jittered[col] != 0) & (jittered[col] != 4095)
        jittered.loc[mask, col] += np.random.uniform(-JITTER_AMOUNT, JITTER_AMOUNT, size=mask.sum())
    
    return jittered

def augment_data(df, augmentations, input_cols):
    if "flip" in augmentations:
        flipped = flip_x_coordinates(df, [c for c in input_cols if c.endswith("_x")])
        df = pd.concat([df, flipped], ignore_index=True)

    if "jitter" in augmentations:
        jittered = jitter(df, [c for c in input_cols if c.endswith("_x") or c.endswith("_y")])
        df = pd.concat([df, jittered], ignore_index=True)

    if "permute" in augmentations:
        permuted = permute_stones(df)
        df = pd.concat([df, permuted], ignore_index=True)

    return df