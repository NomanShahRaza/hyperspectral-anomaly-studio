import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import UPLOADS, _purge_duplicate_uploads

_purge_duplicate_uploads()
print("remaining:")
for path in sorted(UPLOADS.iterdir()):
    if path.is_file():
        print(path.name)
