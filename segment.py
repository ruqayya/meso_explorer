"""Segment preprocessed cells with standalone SAM 3.1. Successful masks resume automatically."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from common import fingerprint, read_csv, save_settings, settings_hash, write_csv
from postprocess import ExternalMaskCleanupConfig, clean_external_mask

DEFAULT_PROMPT = "Segment the human cell centered in this brightfield image"


def can_resume(record, image_sha, signature, mask_path, raw_mask_path):
    return (
        record.get("status") == "ok"
        and record.get("settings_hash") == signature
        and record.get("image_sha256") == image_sha
        and mask_path.exists()
        and record.get("mask_sha256") == fingerprint(mask_path)
        and raw_mask_path.exists()
        and record.get("raw_mask_sha256") == fingerprint(raw_mask_path)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint", type=Path, help="SAM 3.1 multiplex .pt or original-key .safetensors")
    parser.add_argument("--hf-token-file", type=Path, help="Optional dotenv file; never copied or logged")
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--refine-iterations", type=int, default=2)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--no-postproc",
        "--no_postproc",
        action="store_true",
        help="Disable the default guarded light morphological cleanup of SAM masks",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate masks, allowing changed settings"
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.refine_iterations <= 5:
        parser.error("threshold must be in [0,1] and refine-iterations in [0,5]")
    run = args.run.resolve()
    images = read_csv(run / "images.csv")
    images = images[images.preprocess_status == "ok"]
    if images.empty:
        parser.error("No successfully preprocessed images")
    checkpoint = args.checkpoint
    if checkpoint is None:
        from dotenv import dotenv_values
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN")
        if args.hf_token_file:
            secrets = dotenv_values(args.hf_token_file)
            token = next(
                (
                    secrets[k]
                    for k in (
                        "HF_TOKEN",
                        "HUGGINGFACE_TOKEN",
                        "HUGGING_FACE_HUB_TOKEN",
                        "HUGGINGFACEHUB_API_TOKEN",
                    )
                    if secrets.get(k)
                ),
                token,
            )
            if not token:
                parser.error("No recognized Hugging Face token key found in token file")
        checkpoint = Path(hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt", token=token))
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    from sam31 import SAM_REVISION, Sam31

    settings = {
        "model": "sam3.1",
        "sam_revision": SAM_REVISION,
        "checkpoint_sha256": fingerprint(checkpoint),
        "prompt": args.prompt,
        "threshold": args.threshold,
        "refine_iterations": args.refine_iterations,
        "corner_inset": 4,
        "input_range": "0..1",
        "device": args.device,
        "precision": "bfloat16 autocast",
        "postproc": "none" if args.no_postproc else "light",
    }
    cleanup_config = ExternalMaskCleanupConfig(mode=settings["postproc"])
    save_settings(run, "segment", settings, args.overwrite)
    signature = settings_hash(settings)
    manifest = run / "masks.csv"
    previous = (
        {}
        if args.overwrite or not manifest.exists()
        else read_csv(manifest).set_index("image_id").to_dict("index")
    )
    rows, model = [], None
    for row in tqdm(images.to_dict("records"), desc="SAM 3.1"):
        image_path = run / row["processed_path"]
        sha = fingerprint(image_path)
        dest_rel = Path("masks") / Path(row["processed_path"]).relative_to("images")
        dest = run / dest_rel
        raw_rel = Path("masks_raw") / Path(row["processed_path"]).relative_to("images")
        raw_dest = run / raw_rel
        old = previous.get(row["image_id"], {})
        record = {
            "image_id": row["image_id"],
            "image_sha256": sha,
            "mask_path": dest_rel.as_posix(),
            "settings_hash": signature,
            "raw_mask_path": raw_rel.as_posix(),
        }
        if can_resume(old, sha, signature, dest, raw_dest):
            rows.append({"image_id": row["image_id"], **old})
            continue
        if model is None:
            print(f"Loading SAM 3.1 on {args.device}...", flush=True)
            if args.device == "cpu":
                torch.set_num_threads(min(8, os.cpu_count() or 1))
            model = Sam31(checkpoint, device=args.device, prompt=args.prompt)
        start = time.perf_counter()
        try:
            with Image.open(image_path) as image:
                mask, info = model.predict(image.convert("RGB"), args.threshold, args.refine_iterations)
            raw_dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(raw_dest)
            mask, cleanup_info = clean_external_mask(mask, cleanup_config)
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(dest)
            record.update(
                info,
                **cleanup_info,
                status="ok" if mask.any() else "empty",
                mask_sha256=fingerprint(dest),
                raw_mask_sha256=fingerprint(raw_dest),
                error="",
                elapsed_seconds=round(time.perf_counter() - start, 3),
            )
        except (OSError, ValueError) as exc:
            record.update(
                status="error", error=str(exc), elapsed_seconds=round(time.perf_counter() - start, 3)
            )
        rows.append(record)
        # Persist after each image. Keep unvisited rows so interrupted resumes remain useful.
        completed = {r["image_id"] for r in rows}
        remaining = [{"image_id": key, **value} for key, value in previous.items() if key not in completed]
        write_csv(pd.DataFrame(rows + remaining), manifest)
    frame = pd.DataFrame(rows)
    write_csv(frame, manifest)
    errors = (frame.status != "ok").sum()
    print(f"Masks ready: {len(frame) - errors}/{len(frame)} -> {manifest}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
