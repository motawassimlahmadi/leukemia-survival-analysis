import polars as pl

def load_dataset(dataset_path):
    """
    Loads data from the specified path
    """
    return pl.read_csv(dataset_path)

def load_additional_data(path):
    """
    Load additional data
    """
    return pl.read_csv(path)