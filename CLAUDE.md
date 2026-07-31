# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install core dependencies (first time)
pip install -r requirements.txt

# For liquify with scipy acceleration (optional — works without it via pure numpy)
pip install scipy

# For building the Windows installer
pip install pyinstaller svgpathtools

# Launch the review/edit GUI (main entry point)
python reviewer.py

# Launch the old batch-processing GUI
python main.py

# Run the liquify tool standalone on a single image
python liquify.py

# Run the manual annotation tool
python annotator.py <素材目录> [输出JSON]

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
| `reviewer.py` | **Primary GUI** — streaming AI processing, bbox drag-editing, angle adjustment (buttons + drag-to-rotate), liquify integration, per-pair export |
| `gui.py` + `main.py` | **Old GUI** — batch folder-in/folder-out with progress bar (uses `processor.py`) |
| `annotator.py` | **Manual annotation tool** — draw bboxes on individual images, saves to `annotations.json`. **Self-contained** — has its own rembg import, does not depend on any processor module. |
| `liquify.py` | **Standalone** — can be run directly as `python liquify.py` for single-image PS-style warp editing |
| `mask_annotator.py` | **Mask brush tool** — paint to erase/restore rembg mask regions (mannequin neck/torso). Saves `<stem>_mask.png`. Auto-detected by reviewer. |
| `train_u2net.py` | **Fine-tuning pipeline** — trains U-2-Net on human-annotated masks, exports ONNX. Multi-directory data support, checkpoint resume. |
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
   - **Consensus refines only horizontal (x1, x2) and bottom (y2) — top (y1) is preserved** (fine-tuned model handles mannequin neck)
   - Applies `_trim_rod_bottom()` to detect narrow regions (mask width < 12% of bbox width) and crop them off only from the bottom
3. The reviewer then allows manual bbox/angle override before cropping

**Golden ratio crop anchoring**: `_simple_crop()` positions the bbox offset within the crop window using φ ≈ 1.618 — outer margin (near frame edge) : total inner gap (front+back combined) = φ : 1. This pulls front and back toward the center for narrow garments while barely affecting wide garments. See reviewer.py:972-996.

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
- `liquify.py` — self-contained liquify engine (`LiquifyEngine` + `LiquifyCanvas` + `LiquifyTool`). The engine maintains a 2D deformation grid (spacing=8px) with undo history (50 steps max). Uses scipy for acceleration if available, otherwise falls back to bilinear interpolation in pure numpy.
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

**Installed (onedir):** `sys._MEIPASS` points to `_internal/` under the install dir. The top of `reviewer.py` detects `_internal/models/u2net.onnx` and sets `U2NET_HOME` env var before importing processor_v11, so rembg reads the bundled model directly. No file copying needed.

**Develop mode (`python reviewer.py`):** No `_internal/models/` directory, so `U2NET_HOME` is not set; rembg uses its default `~/.u2net/` path. The model must be downloaded on first run (rembg auto-downloads via pooch).

**Model priority** (see `processor_v11._single_pipe()`):
1. **Manual mask (`<stem>_mask.png`)** — if exists in source dir, loaded directly, no inference
2. **Fine-tuned ONNX (`models/u2net_finetuned.onnx`)** — if exists, runs onnxruntime directly (skips rembg)
3. **Original rembg u2net** — fallback, uses rembg's `new_session()` + `remove()`

The fine-tuned model is auto-detected; no config changes needed. Same priority applies to `mask_annotator.py`.

**Deferred AI** (see `reviewer.py`):
- If user clicks "AI 处理" before model finishes loading, sets `_pending_ai = True`
- `_check_prewarm()` detects model ready → auto-calls `_start_process()`
- Status bar shows "模型就绪后将自动开始 AI 处理..." while pending

**Threading** (see `processor_v11.py`):
- `_get_session()` uses **double-checked locking** (`threading.Lock`) — prevents race conditions where prewarm and worker threads both create separate sessions.
- `prewarm()` only loads model weights (`new_session()`), no inference.
- The **warmup inference** happens in the worker thread right before the first pair: the first real image is passed through `_single_pipe()`. This is critical because ONNX Runtime's first `run()` triggers lazy init (memory planning, kernel JIT, thread pool binding) — it must happen in the same thread that will process real images. A 32×32 dummy image is explicitly NOT used because ONNX allocates different memory/kernel paths for small vs. large images, so a small-image warmup would be useless for real photos.

**Streaming worker** (see `reviewer.py` `_start_process()`):
- AI processing runs in a `daemon=True` background thread that iterates over all pairs.
- After each pair's `_joint_detect()` completes, it calls `self.after(0, lambda: self._on_one_done(idx))` to signal the main (GUI) thread — tkinter is NOT thread-safe, so all UI updates must go through `after()`.
- `_on_one_done()` updates the status bar and, for the very first pair, auto-loads it immediately so the user can start reviewing while remaining pairs process in the background.
- On completion, `_on_all_done()` writes annotations.json via `_save_ai_results()`.
- The `_processing` flag prevents double-starting; `daemon=True` ensures the thread doesn't block app exit.

### Reviewer keyboard shortcuts

| Key | Action |
|-----|--------|
| ← → | Previous/next pair |
| E / S | Export current pair |
| X | Swap front/back |
| R | Reset rotation to 0° |
| F | Fit editors to window |
| F1 | About dialog |
| , / . | Rotate 0.5° CCW/CW (bound to `<comma>`/`<period>`) |

**Mouse interactions (BBoxEditor)**:

| Action | Effect |
|--------|--------|
| Drag corner | Resize bbox |
| Drag edge midpoint | Resize one side |
| Drag inside bbox | Move bbox |
| **Drag outside bbox near corner** | **Rotate bbox around center** ↻ |
| Shift + rotate drag | Snap to 15° increments |
| Scroll | Zoom |
| Middle-drag / Right-drag | Pan |

### Annotator keyboard shortcuts

The annotator (`annotator.py`) is a standalone tool with its own rembg import and keyboard shortcuts, independent of any processor module:

| Key | Action |
|-----|--------|
| S / Enter | Save and jump to next image |
| R | Reset to rembg bbox |
| F | Fit to window |
| Space / → | Next image |
| ← | Previous image |
| 1 / 2 / 3 | Zoom 100% / 200% / 300% |
| Escape | Exit |

### Supported input formats

JPG, JPEG, PNG, BMP, TIFF (defined as `IMAGE_EXTS` in reviewer.py). Recommended resolution ≤ 4080px for responsive UI.

### Important constants

- `MARGIN = 0.12` (reviewer and batch_manual) — 12% extra padding around bbox when cropping
- `CONSENSUS_RATIO_THRESHOLD = 1.35` — max front/back width ratio for consensus rows
- `CATEGORIES` — list of `(height_ratio_threshold, margin, fill_ratio)` tuples for crop sizing; e.g. `(0.60, 0.07, 0.60)` means garments with height/bbox-height > 0.60 get 7% margin and 60% fill
- Liquify grid spacing = 8px, max undo steps = 50

### Exploratory scripts (not part of main architecture)

The repo contains ~30 `batch_*.py`, `diagnose_*.py`, `explore_*.py`, `test_*.py`, `validate_*.py`, `compare_*.py` files from iterative development, plus `processor_v11_backup.py`. These are experimental artifacts from evolving the pipeline through many versions (v9→v23+) and are NOT used in production. They can be safely ignored.

### File pairing convention

Images are sorted lexicographically by filename. Pairs are formed as `(files[i], files[i+1])` for `i = 0, 2, 4, ...`. The first in each pair is treated as the front (正面), the second as the back (反面). This can be swapped in the reviewer UI.
