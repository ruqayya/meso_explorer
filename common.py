"""Small shared helpers; generated paths are relative to the run directory."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def save_settings(run, stage, settings, overwrite=False):
    path = Path(run) / f"{stage}_settings.json"
    if path.exists() and json.loads(path.read_text()) != settings and not overwrite:
        raise ValueError(f"{stage} settings changed. Use a new --run directory or --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def metadata(rel):
    parts = Path(rel).parts
    source = parts[0] if len(parts) > 1 else "unknown"
    label = parts[1] if len(parts) > 2 else "unlabelled"
    text = source.lower()
    control = any(x in text for x in ("a549", "nsclc", "lung cancer"))
    subtype = "unknown"
    if control:
        subtype = "other/control"
    elif "biphas" in text or "msto" in text:
        subtype = "biphasic"
    elif "epithel" in text:
        subtype = "epithelioid"
    elif "sarcom" in text:
        subtype = "sarcomatoid"
    markers = [m for m in ("PCK", "WT1", "CAL") if re.search(rf"(?<![A-Z]){m}(?![A-Z0-9])", label.upper())]
    return {
        "source_folder": source,
        "cell_line": "MSTO-211H" if "msto" in text else "A549" if "a549" in text else source,
        "marker_label": label,
        "marker_group": "+".join(markers) if markers else label,
        "subtype": subtype,
        "disease_group": "nsclc_control"
        if control
        else "mesothelioma"
        if subtype != "unknown"
        else "unknown",
    }
