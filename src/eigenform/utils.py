from pathlib import Path

def ensure_dir(dirpath):
    if not dirpath.exists():
        dirpath.mkdir(parents=True)
    
def get_data_dir():
    cache_dir = Path(__file__).parent / f"../../data/"
    return cache_dir.resolve()
