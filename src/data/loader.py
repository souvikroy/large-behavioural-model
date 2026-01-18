"""Data loading utilities."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of dictionaries containing question-response data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load JSON data as pandas DataFrame.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        DataFrame with question-response data
    """
    data = load_json_data(file_path)
    df = pd.DataFrame(data)
    return df
