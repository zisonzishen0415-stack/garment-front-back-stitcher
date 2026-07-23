# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (first time)
pip install -r requirements.txt

# Launch the review/edit GUI (main entry point)
python reviewer.py

# Launch the old batch-processing GUI
python main.py

# Run the liquify tool standalone on a single image
python liquify.py

# Run the manual annotation tool
python annotator.py

# Build icon assets from logo.svg (requires svgpathtools)
python build_icon.py

# Build Windows installer
# 1. Generate logo assets: python build_icon.py (requires svgpathtools)
# 2. Prepare model:     mkdir models && cp ~/.u2net/u2net.onnx models/
# 3. Build exe:         python -m PyInstaller garment-stitcher.spec
#    → dist/GarmentStitcher/ (onedir, model bundled in _internal/models/)
# 4. Package installer: makensis installer.nsi (requires NSIS)
#    → dist/GarmentStitcher_Setup.exe
```

There is no test suite, linter config, or build step. The project runs directly as Python scripts.

## Architecture

**Garment Front-Back Stitcher** — a fully offline desktop app that uses AI (rembg) + joint contour matching to detect garment regions in paired front/back photos, then crops and stitches them into 1:1 square output images, with a GUI for review and manual adjustment.

### Entry points (three GUIs + standalone tools)

| Script | Role |
|---|---|
| `reviewer.py` | **Primary GUI** — streaming AI processing, bbox drag-editing, angle adjustment, liquify integration, per-pair export |
| `gui.py` + `main.py` | **Old GUI** — batch folder-in/folder-out with progress bar (uses `processor.py`) |
| `annotator.py` | **Manual annotation tool** — draw bboxes on individual images, saves to `annotations.json`. **Self-contained** — has its own rembg import, does not depend on any processor module. |
| `liquify.py` | **Standalone** — can be run directly as `python liquify.py` for single-image PS-style warp editing |
| `batch_manual.py` | **Offline batch export** using existing `annotations.json`. Uses `processor.py` (old v22+ pipeline), not `processor_v11.py`. |

### Core processing pipeline

The reviewer (`reviewer.py`) imports `processor_v11.py` (`ImageProcessorV11`). The old GUI (`gui.py`) imports `processor.py` (`ImageProcessor`). These are **distinct, incompatible processor versions**:

#### `processor_v11.py` (used by reviewer, production)

Single pipeline: `contrast-enhanced rembg → bbox → joint contour consensus → rod/human-form bottom trimming → crop → stitch`.

1. `_single_pipe_bbox()` — runs rembg on contrast-enhanced image (factor 1.4), returns bbox
2. `_joint_detect()` — the key algorithm:
   - Gets raw (un-enhanced) masks for both images via `_get_mask_arr()`
   - Computes vertical width profiles (`_vertical_profile`)
   - Interpolates both onto a common y-axis, computes width ratio row-by-row
   - Rows where ratio < `CONSENSUS_RATIO_THRESHOLD` (1.35) are "consensus" (symmetric width = real garment)
   - Takes the largest contiguous consensus interval → re-derives bboxes within that interval
   - Applies `_trim_rod_bottom()` to detect narrow regions (mask width < 12% of bbox width) and crop them off
3. The reviewer then allows manual bbox/angle override before cropping

**Important**: The reviewer's streaming worker calls `processor._joint_detect()` directly (not `process_pair()`) because it only needs bboxes. The full `process_pair()` which does crop+stitch is used by the old GUI and `process_all()`.

#### `processor.py` (experimental v22+, used by old GUI)

Multi-pipeline variant with confidence voting, trim-only bbox constraints, and pair height ratio checks (`PAIR_HEIGHT_RATIO_MAX = 1.18`). More complex but not used by the main reviewer.

### Data flow

```
Input folder (sorted by filename)
  → odd-index = front, even-index = back (paired: [0,1], [2,3], ...)
  → reviewer: streaming AI per pair in background thread
  → each pair loads immediately when AI completes
  → user adjusts bbox/angle via BBoxEditor (drag corners/edges/body, scroll to zoom)
  → annotations auto-saved with 800ms debounce to annotations.json
  → export: crop front (anchor right) + crop back (anchor left) → resize to equal height → stitch into 1:1 square
```

### Key files and their relationships

- `reviewer.py` — depends on `processor_v11.py` and `liquify.py`. Contains `BBoxEditor` (tk.Canvas with handle-drag interaction), `ReviewerApp` (ctk.CTk main window with toolbar, dual editors, preview canvas), `AboutWindow` (modal popup with logo), and `DebugWindow` (6-step pipeline visualization showing rembg mask → bbox → width profiles → consensus interval → rod trim → result).
- `processor_v11.py` — self-contained, only imports rembg/numpy/Pillow. No internal state beyond a lazily-created rembg session.
- `liquify.py` — self-contained liquify engine (`LiquifyEngine` + `LiquifyCanvas` + `LiquifyTool`). The engine maintains a 2D deformation grid (spacing=8px) with undo history (50 steps max). Can work with or without scipy (falls back to bilinear interpolation in pure numpy).
- `batch_manual.py` — batch export using annotations.json (not used in the main GUI flow).
- `garment-stitcher.spec` — PyInstaller **onedir** build config. Uses the pattern: `EXE(..., exclude_binaries=True)` + `COLLECT(exe, a.binaries, a.zipfiles, a.datas)`. The EXE is a thin bootloader (~18MB) that loads everything from `_internal/`. Do NOT pass `a.binaries`/`a.zipfiles` to EXE — that creates a 129MB hybrid exe that misbehaves (extracts to %TEMP% instead of using `_internal/`). Excludes heavy packages (matplotlib, torch, tensorflow). Hidden imports needed: `rembg`, `onnxruntime`, `skimage`, `pymatting`, `pooch`, `tkinter.ttk`.
- `installer.nsi` — NSIS installer script, packages the PyInstaller output into `GarmentStitcher_Setup.exe`.
- `build_icon.py` — renders `logo.svg` into PNG/ICO assets at various sizes (toolbar, about, placeholder, favicon).

### Annotation persistence

The reviewer reads/writes `annotations.json` in the input directory:
```json
{"source_dir": "...", "annotations": [{"file": "name.jpg", "bbox": [x1,y1,x2,y2], "angle": 0.5}]}
```
On load: manual annotations take priority over AI bboxes. On all-export: uses annotations for unseen pairs, current editor state for the active pair.

### Liquify integration

The "液化" button in the reviewer generates a full-resolution stitched preview and opens `LiquifyTool` as a modal (`Toplevel`). On apply, the result saves to `审核输出/` subdirectory and replaces the preview until the user navigates away.

### Model handling

**Installed (onedir):** `sys._MEIPASS` points to `_internal/` under the install dir. The module-level code sets `U2NET_HOME` to point there so rembg reads the bundled model directly. No file copying needed.

**Develop mode (`python reviewer.py`):** `_BUNDLED_MODEL` won't exist, so rembg uses its default `~/.u2net/` path. The model must be downloaded on first run (rembg auto-downloads via pooch).

**Threading** (see `processor_v11.py`):
- `_get_session()` uses **double-checked locking** (`threading.Lock`) — prevents race conditions where prewarm and worker threads both create separate sessions.
- `prewarm()` only loads model weights (`new_session()`), no inference.
- The **warmup inference** happens in the worker thread right before the first pair: a 32×32 dummy image is passed through rembg. This is critical because ONNX Runtime's first `run()` triggers lazy init (memory planning, kernel JIT, thread pool binding) — it must happen in the same thread that will process real images, otherwise the first pair gets wrong results.

### Reviewer keyboard shortcuts

| Key | Action |
|-----|--------|
| ← → | Previous/next pair |
| E / S | Export current pair |
| X | Swap front/back |
| R | Reset rotation to 0° |
| F | Fit editors to window |
| , / . | Rotate 0.5° CCW/CW |

### Important constants

- `MARGIN = 0.12` (reviewer) — extra space around bbox when cropping
- `CONSENSUS_RATIO_THRESHOLD = 1.35` — max front/back width ratio for consensus rows
- `CATEGORIES` — maps garment height ratio to margin/fill parameters for crop sizing
- Liquify grid spacing = 8px, max undo steps = 50

### File pairing convention

Images are sorted lexicographically by filename. Pairs are formed as `(files[i], files[i+1])` for `i = 0, 2, 4, ...`. The first in each pair is treated as the front (正面), the second as the back (反面). This can be swapped in the reviewer UI.
