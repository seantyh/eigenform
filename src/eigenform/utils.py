from pathlib import Path

def get_data_dir():
    cache_dir = Path(__file__).parent / f"../../data/"
    return cache_dir.resolve()
