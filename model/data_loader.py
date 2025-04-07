import pandas as pd

def load_and_prepare_data(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Drop rows where 'title', 'text', or 'label' is missing
    df = df.dropna(subset=['title', 'text', 'label'])

    # Combine 'title' and 'text' columns into a single 'content' column
    df['content'] = df['title'] + ' ' + df['text']

    # Return a DataFrame with only the 'content' and 'label' columns
    return df[['content', 'label']]
