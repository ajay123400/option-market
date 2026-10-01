"""paths.py -- resolves the app's base directory consistently whether
running as a plain `python x.py` script or as a PyInstaller-frozen .exe.

A frozen --onefile .exe unpacks its bundled code into a temp directory
(sys._MEIPASS) that gets deleted when the process exits -- fine for
read-only resources like templates/, but wrong for anything that must
persist or that the user should be able to edit: .env, the data/ and
results/ folders, instrument lists, the cached Fyers token, the paper-trade
state file. Those all need to live next to the .exe itself instead.
"""
import json
import os
import sys

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = getattr(sys, "_MEIPASS", BASE_DIR)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = BASE_DIR


def atomic_write_json(path, data, indent=1):
    """Writes JSON to `path` without ever leaving a truncated/corrupt file
    behind if the process dies mid-write (killed exe, crash, power loss).
    A plain `open(path, "w")` + json.dump() is NOT safe against this --
    if the write is interrupted, the file is left half-written and every
    later json.load() of it raises, which is exactly the kind of crash
    a background writer (the historical recorder, manual positions) must
    not be able to cause. Writes to a sibling temp file first, then
    os.replace()s it over the target -- that swap is atomic on both
    POSIX and Windows, so readers only ever see the fully-old or fully-
    new file, never a partial one."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=indent)
    os.replace(tmp_path, path)
