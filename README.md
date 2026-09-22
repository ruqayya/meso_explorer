# Single-cell morphology exploration

Raw brightfield images → cropped images → SAM 3.1 masks → morphology CSV → UMAPs, boxplots and heatmaps.

## Setup (Windows PowerShell, Python 3.12)

```powershell
git clone https://github.com/measty/meso_explorer.git
cd meso_explorer
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

An NVIDIA GPU is recommended. `segment.py --device cpu` also works, but is slower. Install a suitable PyTorch build for other hardware. Linux uses the same scripts (activate with `source .venv/bin/activate`).

## Run these four steps to generate umap and boxplots

```powershell
python preprocess.py --input "path\to\cell_images" --run results
python segment.py --run results --checkpoint "path\to\models\checkpoints\sam3.1_multiplex_fp16.safetensors"
python features.py --run results
python plot.py --run results --meso-only
```

The checkpoint path above can be used to provide a weights file. Alternatively, omit `--checkpoint` to download the official gated `facebook/sam3.1` checkpoint. After accepting its model terms, optionally supply `--hf-token-file "path\to\.env.hf"` (a dotenv file containing `HF_TOKEN=...`, or a supported Hugging Face token alias).

preprocess.py: removes the top band and crops around the cell
segment.py: runs sam segmentation & morphological cleanup (use --no-postproc to disable this)
features.py: calculates the morphological features
plot.py: creates the plots  

## Outputs and interpretation

Once the above steps have been run, outputs will be in these folders:

| Output in the run directory | Contents |
|---|---|
| `images/`, `images.csv` | Native-size crops, original identity, crop coordinates and labels |
| `masks_raw/` | Untouched binary SAM masks |
| `masks/`, `masks.csv` | Masks after optional cleanup |
| `cell_masks/`, `features.csv` | The measured dominant component; shape, intensity and texture features; QC flags |
| `qc/` | Sample sheets showing crop, segmentation mask and measured boundary |
| `plots_meso/` (or `plots/`) | UMAP PNGs/coordinates, boxplots, heatmaps, counts, summaries and analysis settings |

Features use the **largest connected component**, consistently for shape and intensity. Area is in pixels², lengths in pixels, intensities in [0,1]; there is no micrometre calibration. Texture includes entropy, Sobel, LBP and GLCM, measured inside the cell. Masks that are empty, tiny, very large, or contain substantial disconnected components are flagged. The `touches_border` flag requires **15 or more consecutive mask pixels along any single image edge**; separate short contacts are not added together. Plotting excludes flagged/failed cells by default. Inspect `qc/` and `features.csv`; use `--include-flagged` deliberately if appropriate. To apply updated QC rules to an existing run, rerun `features.py` and `plot.py`; segmentation does not need to be repeated.

Folder names supply `marker_group` (CAL/PCK/WT1/WT1+CAL), `cell_line`, `subtype` and `disease_group`. To correct labels, pass `--metadata labels.csv` to preprocessing; use `image_id` (raw-root-relative path with `/`) plus any label columns to override. Example:

```csv
image_id,subtype,marker_group
Epithelioid Cells 23/23-CAL/1.png,epithelioid,CAL
```

```powershell
# Include controls; one shared UMAP colored three ways
python plot.py --run results --group-by marker_group subtype cell_line
# Mesothelioma only, with a separate output folder
python plot.py --run results --meso-only --group-by marker_group subtype
```

UMAP uses only the listed biological feature columns, median imputation and z-score scaling, with seed 42. subtype/marker differences can be confounded by cell line or acquisition.