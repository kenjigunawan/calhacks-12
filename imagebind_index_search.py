#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ImageBind
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

try:
    from tqdm import tqdm
except Exception:
    # fall back if tqdm not installed
    def tqdm(x, **kwargs): return x

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

import subprocess, shlex
from pytorchvideo.data.encoded_video import EncodedVideo

def has_video_stream(path: str) -> bool:
    # Returns True if ffprobe sees a video stream v:0
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 {shlex.quote(path)}'
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
        return len(out) > 0
    except subprocess.CalledProcessError:
        return False

def load_video_via_pyav(path: str, device, clip_frames: int = 2):
    """
    Very small fallback: load a short clip with PyAV backend and format
    as [B=1, C=3, T=clip_frames, H=224, W=224] with CLIP-like norm.
    """
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from pytorchvideo.data.encoded_video import EncodedVideo

    vid = EncodedVideo.from_path(path, decoder="pyav")
    # Take the middle of the clip; fall back to [0, duration)
    dur = float(vid.duration) if vid.duration is not None else 0.0
    t0, t1 = (0.0, dur) if dur <= 0.0 else (max(0.0, dur*0.25), dur*0.75)
    # Get a temporal slice; returns dict with 'video' tensor [T, H, W, C]
    clip = vid.get_clip(t0, t1)
    frames = clip["video"]  # float32 [T, H, W, C] in [0, 255] or [0,1] depending on backend
    if frames.dtype != torch.float32:
        frames = frames.float()
    # ensure [0,1]
    if frames.max() > 1.5:
        frames = frames / 255.0

    # Uniformly sample clip_frames from available frames
    T = frames.shape[0]
    if T == 0:
        raise RuntimeError("PyAV fallback: no frames decoded")
    idx = torch.linspace(0, T-1, steps=min(clip_frames, T)).round().long()
    frames = frames.index_select(0, idx)  # [t, H, W, C]

    # Resize + center-crop to 224, per-frame
    proc = []
    for f in frames:  # f: [H, W, C]
        f = f.permute(2,0,1)  # [C,H,W]
        # keep aspect: resize short side to 256 then center-crop 224
        h, w = f.shape[1], f.shape[2]
        short = min(h, w)
        scale = 256.0 / short
        nh, nw = int(round(h*scale)), int(round(w*scale))
        f = TF.resize(f, [nh, nw], antialias=True)
        f = TF.center_crop(f, [224, 224])
        proc.append(f)
    x = torch.stack(proc, dim=1)  # [C, T, 224, 224]

    # CLIP/ImageBind normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:,None,None,None]
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:,None,None,None]
    x = (x - mean) / std
    x = x.unsqueeze(0).to(device)  # [1, C, T, H, W]
    return x


def find_files(root: Path, exts: set) -> List[Path]:
    if root is None:
        return []
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p.resolve())
    return sorted(out)

def sha1_for_file(p: Path) -> str:
    """Stable id based on path + size + mtime (fast; robust to renames in same tree)."""
    st = p.stat()
    h = hashlib.sha1()
    h.update(str(p).encode("utf-8"))
    h.update(str(st.st_size).encode("utf-8"))
    h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def save_vector(vec: torch.Tensor, dest_dir: Path, item_id: str) -> str:
    """Save 1D torch vector (already on CPU) to .npy. Return relative path."""
    ensure_dir(dest_dir)
    arr = vec.detach().cpu().numpy().astype(np.float32)
    out_path = dest_dir / f"{item_id}.npy"
    np.save(out_path, arr)
    return str(out_path)

def load_manifest(meta_path: Path) -> Dict[str, dict]:
    """Return id->record dict."""
    if not meta_path.exists():
        return {}
    idx = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx[rec["id"]] = rec
    return idx

def append_manifest(meta_path: Path, rec: dict):
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_vectors(meta_path: Path, root: Path) -> Tuple[np.ndarray, List[dict]]:
    """Load all .npy vectors referenced by meta.jsonl; returns (X, records)."""
    records = []
    vecs = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            p = (root / rec["vector_path"]).resolve()
            if not p.exists():
                continue
            v = np.load(p)
            # ensure unit length
            n = np.linalg.norm(v) + 1e-8
            vecs.append((v / n).astype(np.float32))
            records.append(rec)
    if not vecs:
        return np.empty((0, 0), dtype=np.float32), []
    X = np.stack(vecs, axis=0)
    return X, records

def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

# -----------------------------
# INDEX (compute & save vectors)
# -----------------------------

def index_main(args):
    out_dir = Path(args.out_dir).resolve()
    vec_dir = out_dir / "embeds"
    meta_path = out_dir / "meta.jsonl"
    ensure_dir(vec_dir)

    # Discover files
    img_paths = []
    vid_paths = []
    if args.images:
        for d in args.images:
            img_paths += find_files(Path(d), IMG_EXTS)
    if args.videos:
        for d in args.videos:
            vid_paths += find_files(Path(d), VID_EXTS)

    if args.image_list:
        with open(args.image_list, "r") as f:
            img_paths += [Path(x.strip()).resolve() for x in f if x.strip()]
    if args.video_list:
        with open(args.video_list, "r") as f:
            vid_paths += [Path(x.strip()).resolve() for x in f if x.strip()]

    img_paths = sorted(set(img_paths))
    vid_paths = sorted(set(vid_paths))

    print(f"Found {len(img_paths)} images, {len(vid_paths)} videos")

    # Model & device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()

    # Resume support
    existing = load_manifest(meta_path) if args.resume else {}

    # ---- Images (batched) ----
    for batch in tqdm(list(batched(img_paths, args.image_batch)), desc="Images"):
        # Skip items already indexed
        todo = [p for p in batch if sha1_for_file(p) not in existing]
        if not todo:
            continue
        # Loader builds CPU tensors then .to(device)
        xb = data.load_and_transform_vision_data([str(p) for p in todo], device)
        with torch.inference_mode():
            out = model({ModalityType.VISION: xb})[ModalityType.VISION]  # [B, D]
            out = F.normalize(out, dim=-1).cpu()
        # Save each vector
        for i, p in enumerate(todo):
            item_id = sha1_for_file(p)
            vec_rel = os.path.relpath(save_vector(out[i], vec_dir, item_id), out_dir)
            rec = {
                "id": item_id,
                "path": str(p),
                "modality": "image",
                "vector_path": vec_rel,
                "size": p.stat().st_size,
            }
            append_manifest(meta_path, rec)
            existing[item_id] = rec
        # free
        del xb, out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Videos (process one by one to keep RAM flat) ----
# ---- Videos (robust: skip bad + fallback to PyAV) ----
for p in tqdm(vid_paths, desc="Videos"):
    item_id = sha1_for_file(p)
    if args.resume and item_id in existing:
        continue

    # Quick precheck: skip files without a video stream
    if not has_video_stream(str(p)):
        print(f"[skip] no video stream: {p}")
        continue

    xb = None
    try:
        # Try Decord path first (ImageBind helper)
        xb = data.load_and_transform_video_data(
            [str(p)], device,
            clips_per_video=args.clips_per_video,
            clip_duration=args.clip_duration_frames,
        )
    except Exception as e:
        print(f"[warn] Decord failed on {p}: {e}\nTrying PyAV fallback…")
        try:
            xb = load_video_via_pyav(str(p), device, clip_frames=args.clip_duration_frames)
        except Exception as e2:
            print(f"[skip] PyAV fallback also failed on {p}: {e2}")
            continue

    with torch.inference_mode():
        out = model({ModalityType.VISION: xb})[ModalityType.VISION]  # [1, D]
        out = F.normalize(out, dim=-1).squeeze(0).cpu()

    vec_rel = os.path.relpath(save_vector(out, vec_dir, item_id), out_dir)
    rec = {
        "id": item_id,
        "path": str(p),
        "modality": "video",
        "vector_path": vec_rel,
        "size": p.stat().st_size,
        "clips_per_video": args.clips_per_video,
        "clip_duration_frames": args.clip_duration_frames,
    }
    append_manifest(meta_path, rec)
    existing[item_id] = rec

    del xb, out
    if device.type == "cuda":
        torch.cuda.empty_cache()

# -----------------------------
# SEARCH (load & compare)
# -----------------------------

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # inputs assumed L2-normalized
    return A @ B.T

def search_main(args):
    out_dir = Path(args.out_dir).resolve()
    meta_path = out_dir / "meta.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"Manifest not found: {meta_path}. Run 'index' first.")

    X, records = load_vectors(meta_path, out_dir)
    if X.shape[0] == 0:
        raise SystemExit("No vectors found.")
    print(f"Loaded {X.shape[0]} vectors of dim {X.shape[1]}")

    # Build lookup by path
    path_to_idx = {rec["path"]: i for i, rec in enumerate(records)}

    if args.query_path:
        q_path = str(Path(args.query_path).resolve())
        if q_path not in path_to_idx:
            raise SystemExit(f"Query path not in dataset: {q_path}")
        qi = path_to_idx[q_path]
        qv = X[qi : qi + 1]                        # [1, D]
        sims = cosine_sim_matrix(qv, X).ravel()    # [N]
        order = np.argsort(-sims)                  # descending
        print(f"\nTop-{args.topk} nearest to:\n  {q_path}\n")
        k = min(args.topk, len(order))
        for rank in range(k):
            j = order[rank]
            print(f"{rank+1:>2}.  sim={sims[j]:.4f}  {records[j]['modality']:>5}  {records[j]['path']}")
    else:
        # Pairwise matrix (careful for large N)
        N = X.shape[0]
        if N > 2000 and not args.force:
            print(f"Dataset has {N} items; pairwise matrix is {N}×{N} (~{(N*N*4)/1e9:.2f} GB float32). "
                  f"Use --force if you really want it, or use --query-path.")
            return
        S = cosine_sim_matrix(X, X)  # [N, N]
        np.save(out_dir / "similarity_matrix.npy", S.astype(np.float32))
        print(f"Saved pairwise cosine similarity matrix to {out_dir/'similarity_matrix.npy'}")

# -----------------------------
# CLI
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Batch ImageBind embeddings (images & videos) and search.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # index
    p_idx = sub.add_parser("index", help="Compute & store vectors.")
    p_idx.add_argument("--images", nargs="*", type=str, default=[], help="Image directories (recursive).")
    p_idx.add_argument("--videos", nargs="*", type=str, default=[], help="Video directories (recursive).")
    p_idx.add_argument("--image-list", type=str, help="Text file with one image path per line.")
    p_idx.add_argument("--video-list", type=str, help="Text file with one video path per line.")
    p_idx.add_argument("--out-dir", type=str, required=True, help="Output directory for embeds/ and meta.jsonl.")
    p_idx.add_argument("--image-batch", type=int, default=32, help="Batch size for images.")
    p_idx.add_argument("--clips-per-video", type=int, default=1, help="Temporal clips per video for loader.")
    p_idx.add_argument("--clip-duration-frames", type=int, default=2, help="Frames per clip (not seconds).")
    p_idx.add_argument("--resume", action="store_true", help="Skip files already in manifest if unchanged.")

    # search
    p_s = sub.add_parser("search", help="Load vectors from files and compare.")
    p_s.add_argument("--out-dir", type=str, required=True, help="Directory with embeds/ and meta.jsonl.")
    p_s.add_argument("--query-path", type=str, help="Find nearest neighbors to this asset path.")
    p_s.add_argument("--topk", type=int, default=10, help="Top-K neighbors to print for --query-path.")
    p_s.add_argument("--force", action="store_true", help="Allow saving full NxN similarity matrix even if large.")

    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "index":
        index_main(args)
    elif args.cmd == "search":
        search_main(args)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()

