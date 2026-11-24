# Cross-Era Fantasy Cricket: Full Project Context

> Handover document for teammates. It covers what was built, why, with which tools, how to run it, what results we got, and what is still left.
> Status as of **Tue 24 Nov 2026, ~19:00 IST**. Written from Krishav's (Student 3) side of the project.

---

## 0. TL;DR

- **Goal:** take a **modern cricket clip**, extract the player's **motion**, and make a **historical cricketer** (from one old photo) perform that motion. Then give it an **era look** (1930s film / 1970s TV / 1990s broadcast).
- **What works now:** a full pipeline that turns a modern cover drive into **Sir Don Bradman batting in a 1930s newsreel style** (5.6 s, real speed). The same pipeline is running for **Harold Larwood bowling**, so the final video is *"Larwood bowls → Bradman drives"* (Bodyline, 1932–33).
- **Main technical finding:** a **depth control video fixes the bat**. With only a skeleton, the model loses or mangles the bat; with depth, it holds one proper bat in every run.
- **Honest limitation:** face likeness is still weak, a limit of the small 1.3B model.
- **Deadline:** **Wed 25 Nov 2026, 11:00 PM IST** (Google Form: report + ≤3 min video + public GitHub link + declaration; one member submits).

---

## 1. Submission requirements (from the course announcement)

| Item | Requirement |
|---|---|
| Report | Course LaTeX template **unmodified**, **max 4 double-column pages INCLUDING references**. Must contain **Individual Contributions**, a **Generative AI Disclosure** and the **GitHub link** (repo must be **public**). |
| Video | **≤ 3 minutes** |
| Declaration | Confirm the submission is final |
| Who submits | **One member** submits for the whole team |

---

## 2. Hardware and accounts

| Thing | Details |
|---|---|
| Laptop | Windows 11, **RTX 4050 Laptop 6 GB VRAM**, 16 GB RAM, NVIDIA driver 561 (not updated), Python 3.12 |
| What runs on the laptop | Pose extraction, control videos, depth maps, era filters, metrics, final editing. **All on CPU.** |
| What runs in the cloud | Video generation on **Kaggle free T4** (16 GB nominal, ~15 GB usable, **no bf16**, so fp16) |
| Kaggle quota | 30 GPU-hours/week. Used so far: **~3 h 25 min** |
| Why not Colab | Colab's free-tier terms ban web UIs and "creating deepfakes", so Kaggle was used instead (ComfyUI runs **headless**, no UI/tunnel) |
| Repo | `https://github.com/krishavgoyal/Cross_Era_Fantasy_Cricket`, branch **`Krishav-main`** (branched from `main`) |
| Local repo path | `C:\Users\ASUS V16\OneDrive\Desktop\Cross_Era_Fantasy_Cricket` |
| Python venv | `C:\cefc\venv` (activate: `C:\cefc\venv\Scripts\Activate.ps1`) |
| FFmpeg | 9.1 installed via `winget install Gyan.FFmpeg` |

---

## 3. Why the models in the proposal were replaced

The proposal named **LTX-2** and **MagicAnimate**. Research (Phase 0 / 0.1, saved in `docs/research/`) found:

| Model | Problem for our task |
|---|---|
| **LTX-2** (Lightricks, arXiv 2601.03233) | Uses the photo only as the **first frame**, so nothing keeps the player's identity through the clip; officially needs **32 GB+** VRAM |
| **MagicAnimate** | DensePose-based, last updated in 2023. Kept only as the **team baseline** (our teammates' AnimateDiff/MagicAnimate engine) |
| Papers cited in the proposal (2510.03909, 2506.09411, 2304.01186) | None has usable code for "one photo + pose → video" |
| **Chosen: Wan2.1-VACE-1.3B** (Wan, arXiv 2503.20314) | Takes a **reference photo + a control video** (pose or depth), Apache-2.0, small enough for a free T4 in fp16 |

---

## 4. The pipeline, step by step

```
modern clip ──► DWPose (laptop CPU) ──► pose skeleton video ─┐
            └─► Depth Anything V2 (laptop) ──► depth video ──┤
historical photo ──► background removed, tight crop ─────────┤
                                                             ▼
                         Wan2.1-VACE-1.3B in headless ComfyUI on Kaggle T4
                                                             ▼
             raw 33-frame video ──► retime to real speed ──► era filters
                                                             ▼
                 metrics (PCK / DINOv2 identity / flicker) + final videos
```

### 4.1 Scripts (all in `scripts/` on `Krishav-main`)

| Script | What it does |
|---|---|
| `pose_utils.py` | Shared helpers: DWPose loader, OpenPose-style drawing, video read/write, `find_ffmpeg()`, overlay blurring (`blur_bands`), ffmpeg called with `-nostdin` |
| `extract_pose.py` | **rtmlib DWPose** (Wholebody, DWPose-l 384×288, CPU onnxruntime). `--list-boxes` shows the people in frame 0; `--init-box x,y,w,h` picks the player; best-IoU tracking. Output: `raw.npz` (keypoints T×133×2, COCO-WholeBody layout, scores, fps, boxes) + `overlay.mp4` |
| `make_control.py` | Cleans keypoints (score < 0.3 masked, small gaps filled, Savitzky-Golay smoothing), resamples to **33 frames @ 16 fps, 480×832**, draws an **OpenPose skeleton on black**. Options: `--start`, `--duration`, `--top-margin 0.08`, `--name`. Output `control_33*.mp4/.npz` |
| `make_depth_control.py` | **Depth Anything V2 Small** (Apache-2.0) depth video with the **same timing and crop** as the pose control. One global min/max (no flicker), near = bright. `--soften-head` flattens the modern helmet; `--target-only` keeps **only the tracked player** (rembg mask + keypoint blobs + a "skeleton capsule"), so fielders don't become extra people |
| `prepare_ref.py` | Reference photo: **rembg** background removal, on white, tight crop (~90% of height), plus a before/after contact sheet |
| `build_kaggle_inputs.py` | Builds the Kaggle input zip (`refs/`, `controls/`, `pairs.csv`) with forward-slash paths. Presets: `batter-v2`, `bowler` |
| `kaggle/vace_generate.ipynb` | The generation notebook (see 4.2) |
| `era_filters.py` | `1930s_film` (grayscale, grain, blur, gate jitter, flicker, vignette, 18 fps), `1970s_tv` (desaturated, colour fringing, scanlines, tape noise, 25 fps), `1990s_broadcast` (mild colour, chroma bleed, softness, 25 fps). Seeded, keeps duration, burns the **"AI-generated - academic project"** label |
| `retime.py` | Stretches the 33 frames back to **real time** at 24 fps. Methods: `blend` (**default**), `hold`, `mci` (rejected: it tore the thin bat) |
| `eval_metrics.py` | Metrics: **PCK@0.2** (17 body keypoints, torso-normalised, vs the control), **DINOv2-small identity** cosine (whole-body crops, grayscale), **flicker** (brightness std, mean frame difference) |
| `score_ablation.py` | Scores every run in `results/ablation_runs.csv` → `results/ablation.csv` |
| `make_sequence.py` | Joins clips with title cards (ffmpeg) |
| `make_final.py` | One command: retime → era styles → `final<tag>.mp4`, `final_eras<tag>.mp4`, `compare<tag>.mp4` (source | control | result) |
| `make_match.py` | Builds `final_match.mp4`: intro card → "Larwood bowls" → bowler → "Bradman drives" → batter |
| `make_figures.py` | Report figures in `docs/report/figs/` (ablation strip, final strip, pipeline diagram, bowler strip) |

### 4.2 The Kaggle notebook (`kaggle/vace_generate.ipynb`)

- Clones **ComfyUI** (tag v0.46.0, falls back to latest) and runs it **headless** (localhost API only).
- Downloads only 3 model files (~11.3 GB) from `Comfy-Org/Wan_2.1_ComfyUI_repackaged`:
  `wan2.1_vace_1.3B_fp16.safetensors`, `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (text encoder), `wan_2.1_vae.safetensors`.
- Settings: **480×832, 16 fps, 33 frames, 20 steps, CFG 5, uni_pc / simple, shift 8**, fp16, SDPA attention, fp32 VAE.
- Workflow nodes: UNETLoader, CLIPLoader (wan), VAELoader, CLIPTextEncode ×2, LoadImage, LoadVideo, GetVideoComponents, **WanVaceToVideo**, ModelSamplingSD3, KSampler, TrimVideoLatent, VAEDecode, CreateVideo, SaveVideo.
- Reads `pairs.csv` (`name,ref,control,prompt,negative,seed,frames`), runs a **17-frame smoke test** first, then each row.
- Saves `outputs/` (labelled), `outputs_raw/` (unlabelled, used for metrics and editing) and `kaggle_runs.csv` (time and peak VRAM).
- Cell 1 switches: `RUN_MODE = "pairs"` (run `pairs.csv`) or `"finals"`, plus `BEST_SEED`.

### 4.3 Bugs we hit on Kaggle and fixed

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `No module named 'torchsde'` | The requirements filter skipped every line starting with "torch" | Skip only torch/torchvision/torchaudio; install torchsde explicitly |
| 2 | "No input dataset found" | Kaggle mounts datasets 5 levels deep (`/kaggle/input/datasets/<user>/<slug>/<folder>`) | Search with `find -L` |
| 3 | "No such file …_ctrl.mp4" after a successful run | ComfyUI history also lists the LoadVideo input preview | Keep only entries with `type == "output"` |
| – | Kaggle upload error "forbidden character '\'" | Zip made with PowerShell `Compress-Archive` | Zip with Python `shutil` |
| – | "Save & Run All" stayed **Queued** for 1 hour | Kaggle had no free background GPU | Cancelled it and ran the cells **live** in the editor |

---

## 5. Inputs used

| Input | Details | Source / licence |
|---|---|---|
| Batter clip `data/clips/C2_coverdrive.mp4` | Modern cover drive, 576×1296, 30 fps, 174 frames. Player box `213,153,336,766`, 0 lost frames, median keypoint score 0.75 | Social-media clip, **used only for motion**; overlays blurred in figures. **[TODO: source]** |
| Bradman photo `bradman_raw.tif` → `data/refs/bradman.png` (padded) and `bradman_crop.png` (tight) | 1932 Sam Hood photo, 2138×2500 grayscale | **[TODO: source link + licence]** |
| Bowler clip `data/clips/B1_bowler.mp4` | Modern fast bowler running at the camera, 488×944, 30 fps, **60 frames = 2.0 s**. Bowler box `136,151,295,782`, 0 lost frames, median score 0.77 | Social-media clip, used only for motion. **[TODO: source]** |
| Larwood photo `larwood.jpg` → `data/refs/larwood_crop.png` | 1929 glass-plate photo, 1989×2785 (half body, fedora, suit, cigarette) | **Public domain:** National Library of Australia, Fairfax glass plate negatives, nla.obj-161865715 (Wikimedia Commons "Harold Larwood 1929.jpg") |

Rejected photos: an Alamy stock photo (watermarked or with the watermark removed = copyrighted) and two WhatsApp photos with unknown licences.

---

## 6. Experiments and results

### 6.1 Run A: first attempt (skeleton control + padded photo)

- 17-frame smoke test: **265 s** on the T4. 33-frame run: **546.4 s**, **peak VRAM 14,885 MB** (the T4 is almost full, so **33 frames is the maximum**).
- Problems: generic face, **bat wrong or missing** (the skeleton has no bat information), extra balls, head cut off at the top, video 2.7× too fast (5.6 s of motion squeezed into 2 s).

### 6.2 Ablation: 2×2 + seeds (all 17 frames, ~280 s each on the T4)

Two changes tested: **control** (skeleton vs **depth**) × **reference** (padded vs **tight crop**), plus 4 seeds.

| Run | Control | Ref | Seed | PCK@0.2 ↑ | DINO vs crop ↑ | DINO vs orig ↑ | Brightness std ↓ | Frame diff | Bat (by eye) |
|---|---|---|---|---|---|---|---|---|---|
| A | skeleton | padded | 42 | 0.934 | 0.205 | 0.158 | 2.27 | 10.10 | lost |
| B | depth | padded | 42 | **0.997** | 0.223 | 0.291 | 1.63 | 11.43 | one bat |
| C | skeleton | crop | 42 | 0.969 | 0.172 | 0.130 | 2.49 | 10.62 | **lost** |
| D1 | depth | crop | 42 | 0.983 | 0.206 | 0.207 | 1.98 | 10.26 | one bat |
| D2 | depth | crop | 7 | 0.900 | 0.230 | 0.227 | 1.49 | 10.97 | one bat |
| D3 | depth | crop | 123 | 0.979 | 0.165 | 0.156 | 2.06 | 11.90 | one bat (black pads) |
| D4 | depth | crop | 2024 | 0.990 | **0.302** | **0.305** | **1.25** | 10.40 | one bat (black trousers) |

**Findings:**
1. **Depth control fixed the bat.** All 5 depth runs keep one proper bat; both skeleton runs lose it. Motion accuracy stays high (PCK 0.90–1.00).
2. **Depth lowered brightness flicker** (pose 2.38 vs depth 1.68 on average). Frame-to-frame difference did **not** drop (10.36 vs 10.99), probably because the bat swings further.
3. **The tight crop did NOT clearly raise identity** (B → D1: −0.017 vs crop, −0.084 vs the original).
4. **The seed matters more than the reference** (D1–D4 identity spread 0.137–0.149).
5. Identity is low everywhere (0.13–0.31). Whole-body DINO also measures clothing and pose, so it **ranks runs**; it is not a true face-likeness score.
6. A large dark **cap hid the face** (the prompt said "cap", and the depth map keeps the helmet outline).

Also fixed in this round: an **8% top margin** (head no longer cut off), an extended negative prompt (extra balls gone).

### 6.3 Final batter runs (33 frames, depth + tight crop, seed 42)

| Run | Prompt | Time | PCK | DINO crop | DINO orig | Brightness std | Frame diff |
|---|---|---|---|---|---|---|---|
| **F1 `final_cap`** | default (with cap) | 530.5 s | 0.904 | 0.217 | 0.175 | 1.65 | 11.93 |
| F2 `final_nocap` | bare-headed | 536.5 s | 0.882 | 0.203 | 0.195 | 3.78 | 9.17 |

**Chosen: F1 `final_cap`**, by visual inspection (better face and bat); the numbers do not separate the two.

### 6.4 Real speed

The 33 frames cover 5.6 s of the source, so they play 2.7× too fast at 16 fps. `retime.py` stretches them back to **5.67 s @ 24 fps** (method `blend`; `mci` tore the bat; `hold` steps visibly). The era filters keep that duration.

### 6.5 Bowler: Harold Larwood (running on Kaggle now)

- Controls at **real speed** (2.0 s → 33 frames @ 16 fps, no retime). **Target-only depth** removes the fielders' legs.
- Runs **L1 / L2 / L3** = seeds 42 / 7 / 2024, 33 frames, depth + `larwood_crop.png`.
- Prompt: *"archival 1930s photograph, fast bowler, bare-headed with short neatly combed hair, face clearly visible, white long-sleeved flannel shirt and long white trousers, white boots, delivering a cricket ball with a high overarm bowling action, natural motion"*.
- Negative: default (without "missing bat") + *cricket bat, cap, hat, fedora, suit, blazer, jacket, dark coat, cigarette, helmet, logo, purple shirt, coloured kit, shorts, bare legs, sleeveless, extra people*.
- Risk: the reference shows a fedora, suit and cigarette, so some seeds may copy them.
- **[TODO]** scores, visual verdict, chosen seed.

---

## 7. Output files (laptop, in `data/`, not in git)

| File | What it is |
|---|---|
| `data/final/final_cap.mp4` | **Final batter video**: 2.5 s title card + 5.67 s Bradman cover drive (1930s) = 8.2 s |
| `data/final/final_eras_cap.mp4` | Same shot in 1930s → 1970s → 1990s styles, with cards (21.6 s) |
| `data/final/compare_cap.mp4` | Source (overlays blurred) \| depth control \| generated 1930s, same timeline (5.6 s) |
| `data/final/final_nocap.mp4` (+ eras/compare) | The no-cap alternative |
| `data/final/final_match.mp4` | **[after the bowler run]** Larwood bowls → Bradman drives |
| `data/gen/v1_raw/`, `v2_raw/`, `v3_raw/`, `bowler_raw/` | Raw Kaggle outputs (run A, ablation, finals, bowler) |
| `data/pose/batter/`, `data/pose/bowler/` | `raw.npz` (DWPose keypoints), `overlay.mp4`, `control_33*.mp4/.npz` (skeleton), `depth_33.mp4/.npz`, previews |
| `results/ablation.csv`, `results/ablation_runs.csv` | All scores (**committed to git**) |
| `docs/report/figs/*.png` | Report figures (committed) |

Pose-data format note: the `.npz` files use the **DWPose / COCO-WholeBody 133-keypoint** layout (0–16 body, 17–22 feet, 23–90 face, 91–132 hands; missing = NaN), **not** OpenPose JSON.

---

## 8. Documents in the repo (`Krishav-main`)

| File | Contents |
|---|---|
| `CLAUDE.md` | Final decisions and rules for the coding assistant |
| `README_PIPELINE.md` | How to run every stage (commands) |
| `docs/research/PHASE0_RESEARCH.md`, `PHASE0_1_FREE_ALTERNATIVES.md` | Model, hardware and free-GPU research |
| `docs/REPORT_NOTES.md` | Everything for the report: method, settings, bugs, ablation, results, limitations, ethics, licences |
| `docs/TEAM_INTEGRATION.md` | How to score the teammates' AnimateDiff output for comparison |
| `docs/report/main.tex`, `refs.bib`, `figs/` | **Report draft** (plain two-column for now; to be moved onto the course template). Estimated ~3.5 pages |
| `docs/VIDEO_SCRIPT.md` | 3:00 video script with timings and a shot list |
| `docs/PR_KRISHAV_MAIN.md` | Pull-request text for merging `Krishav-main` → `main` (merges cleanly, no conflicts) |

Commit history on `Krishav-main` (oldest → newest): `2c59fd6` pose scripts → `167498c` notebook → `f30ecac` era filters → `7989a58` metrics + sequence → `cf2d320` / `f79978c` Kaggle fixes → `f0da750` report notes → `85e68d2` ablation (depth, crop, margin, seeds) → `bed420b` cap/no-cap finals → `b1a3f3b` real-speed retime + final videos → `48ddd29` bowler pipeline + match video → `fea6314` / `9a44b6c` report draft, figures, video script, PR notes.

---

## 9. How to reproduce (PowerShell, repo folder, venv active)

```powershell
# setup (once)
py -3.12 -m venv C:\cefc\venv
C:\cefc\venv\Scripts\Activate.ps1
pip install -r requirements-pipeline.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pillow "rembg[cpu]"

# batter: pose -> controls -> reference -> Kaggle zip
python scripts\extract_pose.py --video data\clips\C2_coverdrive.mp4 --out data\pose\batter --init-box 213,153,336,766
python scripts\make_control.py --pose data\pose\batter\raw.npz --out data\pose\batter --start 0 --duration 5.6 --top-margin 0.08 --name control_33_margin
python scripts\make_depth_control.py --video data\clips\C2_coverdrive.mp4 --pose data\pose\batter\raw.npz --out data\pose\batter --start 0 --duration 5.6 --top-margin 0.08 --soften-head
python scripts\prepare_ref.py --input "<bradman_raw.tif>" --out data\refs\bradman_crop.png
python scripts\build_kaggle_inputs.py --preset batter-v2

# Kaggle: New Dataset (zip, Private) -> notebook: Import kaggle\vace_generate.ipynb -> Add Input (only that dataset)
# -> GPU T4 x2, Internet On -> run cells (Cell 1: RUN_MODE) -> download outputs_raw\*.mp4 + kaggle_runs.csv

# after downloading the raw videos
python scripts\score_ablation.py
python scripts\make_final.py --input data\gen\v3_raw\final_cap_s42_33f.mp4 --tag _cap

# bowler
python scripts\extract_pose.py --video data\clips\B1_bowler.mp4 --out data\pose\bowler --init-box 136,151,295,782
python scripts\make_control.py --pose data\pose\bowler\raw.npz --out data\pose\bowler --start 0 --duration 2.0 --top-margin 0.08 --name control_33_margin
python scripts\make_depth_control.py --video data\clips\B1_bowler.mp4 --pose data\pose\bowler\raw.npz --out data\pose\bowler --start 0 --duration 2.0 --top-margin 0.08 --soften-head --target-only
python scripts\prepare_ref.py --input "<larwood.jpg>" --out data\refs\larwood_crop.png
python scripts\build_kaggle_inputs.py --preset bowler
# ...Kaggle as above, then:
python scripts\make_final.py --input data\gen\bowler_raw\<best>_33f.mp4 --tag _bowler --source-duration 2.0 --source-clip data\clips\B1_bowler.mp4 --control data\pose\bowler\depth_33.mp4 --title "Harold Larwood - bowling, reimagined"
python scripts\make_match.py --bowler data\gen\bowler_raw\<best>_33f.mp4
python scripts\make_figures.py
```

Tip: don't type in the terminal while a script runs (older versions let ffmpeg grab key presses; now fixed with `-nostdin`).

---

## 10. Licences, ethics and tools

- **Models:** Wan2.1-VACE-1.3B (Apache-2.0), Depth Anything V2 Small (Apache-2.0), DINOv2-small (Apache-2.0), DWPose via rtmlib, rembg / U²-Net (MIT / Apache-2.0), ComfyUI (GPL-3.0, used as a tool).
- **Ethics:** every generated output carries a burned-in **"AI-generated - academic project"** label. Subjects are **deceased public figures** (Bradman d. 2001, Larwood d. 1995). Photos are public domain (Larwood confirmed; Bradman **[TODO confirm]**). Modern clips are used **only to extract motion**, with their overlays blurred in published figures. Academic use only.
- **Generative AI disclosure (for the report):** Claude (chat) was used for research and step-by-step guidance; Claude Code (Opus 5.5) wrote and tested most of the pipeline code and drafted the documents. All outputs were checked and run by us.

---

## 11. What is left (as of 24 Nov, 19:00)

**Tonight**
1. Kaggle bowler run (L1–L3) finishing ~19:30 → download from `outputs_raw` → `score_ablation.py` → pick the best seed → `make_final` (bowler) + `make_match` → `final_match.mp4`.
2. Put the bowler results, verdict and figure into the report and notes.

**Need from the team**
3. Names, BITS IDs, course code/name; who is in the team (GitHub has 4 branches: Ahaan, Ankit, Pranahv, Krishav); **who did what** (Individual Contributions).
4. Source links: the Bradman photo, the cover-drive clip, the bowler clip.
5. Who speaks which part of the video.
6. Decision: remove old broadcast-derived videos, two large `pose_data.json` files and `student3.zip` from `main` before the repo goes public?
7. Optional: AnimateDiff (team baseline) vs VACE identity/flicker numbers (`docs/TEAM_INTEGRATION.md`).

**Tomorrow (Wed 25 Nov), before 11 PM**
8. Move `main.tex` onto the **course template** (`docs/report/template/`), compile on Overleaf, check **≤ 4 pages**.
9. Record the **≤ 3 min video** (`docs/VIDEO_SCRIPT.md`; take the Kaggle screenshots).
10. Merge `Krishav-main` → `main`, make the repo **public**.
11. One member submits the Google Form (aim for ~8 PM).

**Verified references:** Wan arXiv 2503.20314; MimicMotion arXiv 2406.19680; LTX-2 arXiv 2601.03233 (HaCohen et al., Lightricks).
