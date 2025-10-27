from datasets import load_dataset
import pandas as pd

def load_nba_data():
    """
    Loads the NBA tracking dataset from Hugging Face and returns it as a Pandas DataFrame.
    """
    print("Loading dataset from Hugging Face...")
    # This command downloads the data (if not already cached) and loads it.
    dataset = load_dataset("dcayton/nba_tracking_data_15_16", 'tiny')
    print("Dataset loaded successfully.")

    # The dataset is stored in the 'train' split. We convert it to a Pandas DataFrame.
    df = dataset['train'].to_pandas()
    
    return df

if __name__ == "__main__":
    # Run the function to load the data
    nba_df = load_nba_data()
    # --- Now you can work with your data ---
    
    # Print the first 5 rows of the DataFrame
    print("\nFirst 5 rows of the dataset:")
    print(nba_df.head())

    # Print some info about the DataFrame
    print("\nDataset Info:")
    nba_df.info()
