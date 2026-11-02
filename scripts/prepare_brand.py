from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "frontend" / "icon"
ASSETS = ROOT / "frontend" / "assets"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

icon = Image.open(SRC / "HAD hyperspectral lettermark icon.png").convert("RGBA")
cover = Image.open(SRC / "HAD Studio Spectral Cube Cover.png").convert("RGBA")
print("icon", icon.size, "cover", cover.size)

icon.resize((512, 512), Image.Resampling.LANCZOS).save(ASSETS / "logo.png", optimize=True)
icon.resize((64, 64), Image.Resampling.LANCZOS).save(ASSETS / "favicon-64.png", optimize=True)
icon.resize((32, 32), Image.Resampling.LANCZOS).save(ASSETS / "favicon-32.png", optimize=True)

width = 1600
height = int(cover.size[1] * (width / cover.size[0]))
cover.resize((width, height), Image.Resampling.LANCZOS).convert("RGB").save(
    DOCS / "had-studio-cover.png", quality=90, optimize=True
)
print("wrote logo and cover")
