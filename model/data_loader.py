import pandas as pd


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['title', 'text', 'label'])
    df['content'] = df['title'] + ' ' + df['text']
    return df[['content', 'label']]
