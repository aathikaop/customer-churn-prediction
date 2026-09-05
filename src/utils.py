import pandas as pd


def load_data(file_path):
    """
    Load the dataset from the given file path.

    Parameters:
        file_path (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)