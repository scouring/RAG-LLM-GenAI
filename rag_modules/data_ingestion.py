# -*- coding: utf-8 -*-
import pandas as pd

def load_and_prepare_data(csv_path: str, sample_size: int = 700):
    """Load and clean the wind dataset."""
    df = pd.read_csv(csv_path)
    df = df[df['variety'].notna()]
    data = df.sample(sample_size).to_dict('records')
    return data