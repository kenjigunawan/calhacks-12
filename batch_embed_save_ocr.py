#!/usr/bin/env python3
# batch_embed_save_ocr.py  (Option B: auto-transcode AV1 → H.264 temp file)
import os, gc, argparse, subprocess, shlex, tempfile, pathlib
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

# ImageBind
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# OCR & media
import easyocr                    # pip install easyocr
from PIL import Image             # pip install pillow
from decord import VideoReader    # pip install decord
from pytorchvideo.data.encoded_video import EncodedVideo

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
VID_EXTS = {".mp4",".mov",".avi",".mkv",".webm",".m4v"}

# ---------- basic utils ----------
def find_files(root, exts):
    if not root: return []
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def save_vec(vec: torch.Tensor, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}.npy", vec.detach().cpu().numpy().astype(np.float32))

def load_any_state_dict(path, device):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=str(device))
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for k in ("state_dict","model","ema","module","net"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]; break
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in obj.items())

def clear_cuda(device, do_ipc=False):
    """Aggressively clear Python & CUDA allocators."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if do_ipc:
            try: torch.cuda.ipc_collect()
            except Exception: pass
    gc.collect()

# ---------- codec helpers (Option B) ----------
def video_codec(path: str) -> str:
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 {shlex.quote(path)}'
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip().lower()
    except subprocess.CalledProcessError:
        return ""

def ensure_h264(path: str) -> tuple[str, str | None]:
    codec = video_codec(path)
    if codec != "av1":
        return path, None
    stem = pathlib.Path(path).stem
    tmp_out = os.path.join(tempfile.gettempdir(), f"{stem}_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", path, "-map", "0:v:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "main",
        "-movflags", "+faststart", "-an", tmp_out
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp_out, tmp_out
    except Exception as e:
        print(f"[skip] Failed to transcode AV1 to H.264 for {path}: {e}")
        return path, None

def has_video_stream(path: str) -> bool:
    return video_codec(path) != ""

# ---------- OCR ----------
def ocr_reader(lang="en", gpu=False):
    if not hasattr(ocr_reader, "_reader"):
        ocr_reader._reader = easyocr.Reader([lang], gpu=gpu)
    return ocr_reader._reader

def ocr_image_np(np_img, use_gpu=False, max_side=1280) -> str:
    h, w = np_img.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / float(s)
        new_w, new_h = int(w * scale), int(h * scale)
        pil = Image.fromarray(np_img)
        pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
        np_img = np.asarray(pil)
    res = ocr_reader(gpu=use_gpu).readtext(np_img, detail=0)
    res = [s.strip() for s in res if isinstance(s, str) and s.strip()]
    return " ".join(res)[:500] if res else ""

# ---------- frame normalization ----------
def _to_hwc_uint8(frames):
    import numpy as np, torch as _t
    if isinstance(frames, _t.Tensor):
        frames = frames.detach().cpu().numpy()
    if not hasattr(frames, "shape") or frames.ndim != 4:
        raise RuntimeError(f"Bad frames shape: {getattr(frames, 'shape', type(frames))}")

    shape = list(frames.shape)
    axes = [0,1,2,3]
    cand = [i for i,s in enumerate(shape) if s in (1,3,4)]
    if not cand:
        raise RuntimeError(f"No channel-like axis in {shape}")
    c_ax = cand[0]
    non_c = [i for i in axes if i != c_ax]
    t_ax = max(non_c, key=lambda i: shape[i])
    hw_axes = [i for i in axes if i not in (c_ax, t_ax)]
    h_ax, w_ax = sorted(hw_axes, key=lambda i: shape[i])
    arr = np.transpose(frames, (t_ax, h_ax, w_ax, c_ax))

    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    out = []
    for i in range(arr.shape[0]):
        f = arr[i]
        c = f.shape[-1]
        if c == 1:
            f = np.repeat(f, 3, axis=-1)
        elif c == 4:
            f = f[..., :3]
        out.append(f)
    return out

# ---------- OCR frame sampling ----------
def sample_video_frames_for_ocr(safe_path: str, num_frames: int = 6):
    vr = VideoReader(safe_path)
    n = len(vr)
    if n == 0: return []
    idxs = np.linspace(0, n - 1, num=min(num_frames, n), dtype=np.int64)
    batch = vr.get_batch(idxs)
    return _to_hwc_uint8(batch)

# ---------- fusion ----------
def fuse(vision_vec: torch.Tensor,
         ocr_text_vec: torch.Tensor | None,
         w_vis: float = 1.0,
         w_txt: float = 1.0) -> torch.Tensor:
    zs = []
    if vision_vec is not None:
        zs.append(w_vis * F.normalize(vision_vec, dim=0))
    if ocr_text_vec is not None:
        zs.append(w_txt * F.normalize(ocr_text_vec, dim=0))
    z = torch.stack(zs).sum(0) if len(zs) > 1 else zs[0]
    return F.normalize(z, dim=0)

# ---------- video embedding backends ----------
def embed_video_with_decord(model, device, safe_path: str, clips_per_video=1, clip_duration=2, half=False):
    with torch.cuda.amp.autocast(enabled=half and device.type == "cuda", dtype=torch.float16):
        xb = data.load_and_transform_video_data([safe_path], device,
                                                clips_per_video=clips_per_video,
                                                clip_duration=clip_duration)
        if half and device.type == "cuda" and torch.is_floating_point(xb):
            xb = xb.to(dtype=torch.float16)
        with torch.inference_mode():
            z = model({ModalityType.VISION: xb})[ModalityType.VISION].squeeze(0)
    del xb
    return z

def embed_frames_as_images(model, device, frames_uint8_list, half=False):
    tmp_paths = []
    try:
        for i, fr in enumerate(frames_uint8_list):
            im = Image.fromarray(fr)
            fd, pth = tempfile.mkstemp(suffix=f"_frm{i}.jpg")
            os.close(fd); im.save(pth, quality=90)
            tmp_paths.append(pth)
        with torch.cuda.amp.autocast(enabled=half and device.type == "cuda", dtype=torch.float16):
            xb = data.load_and_transform_vision_data(tmp_paths, device)
            if half and device.type == "cuda" and torch.is_floating_point(xb):
                xb = xb.to(dtype=torch.float16)
            with torch.inference_mode():
                Z = model({ModalityType.VISION: xb})[ModalityType.VISION]
                Z = F.normalize(Z, dim=-1).mean(0)
        del xb
        return Z
    finally:
        for p in tmp_paths:
            try: os.remove(p)
            except: pass

def embed_video_with_pyav(model, device, safe_path: str, half=False):
    vid = EncodedVideo.from_path(safe_path, decoder="pyav")
    dur = float(vid.duration) if vid.duration is not None else 0.0
    t0, t1 = (0.0, max(0.1, dur)) if dur > 0 else (0.0, 1.0)
    clip = vid.get_clip(t0, t1)
    frames = clip.get("video", None)
    if frames is None:
        raise RuntimeError("PyAV produced no frames")
    hwc = _to_hwc_uint8(frames)
    if len(hwc) > 2:
        idx = np.linspace(0, len(hwc)-1, 2, dtype=int)
        hwc = [hwc[i] for i in idx]
    return embed_frames_as_images(model, device, hwc, half=half)

def embed_video_with_opencv(model, device, safe_path: str, half=False):
    import cv2
    cap = cv2.VideoCapture(safe_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = np.linspace(0, max(0, total-1), num=min(2, max(1, total)), dtype=int)
    frames = []
    for t in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("OpenCV extracted no frames")
    return embed_frames_as_images(model, device, frames, half=half)

# ---------- main ----------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = imagebind_model.imagebind_huge(pretrained=(args.weights is None)).to(device).eval()
    if args.weights:
        sd = load_any_state_dict(args.weights, device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded custom weights: {args.weights} (missing={len(missing)}, unexpected={len(unexpected)})")

    if args.half and device.type == "cuda":
        model.half()

    out_dir = Path(args.out_dir); emb_dir = out_dir / "embeds"
    emb_dir.mkdir(parents=True, exist_ok=True)

    img_paths = find_files(args.images, IMG_EXTS) if args.images else []
    vid_paths = find_files(args.videos, VID_EXTS) if args.videos else []
    print(f"Found {len(img_paths)} images, {len(vid_paths)} videos")

    # ---------- IMAGES (batched) ----------
    for bi in range(0, len(img_paths), args.batch):
        batch = img_paths[bi:bi+args.batch]

        # Vision
        with torch.cuda.amp.autocast(enabled=args.half and device.type == "cuda", dtype=torch.float16):
            xb = data.load_and_transform_vision_data([str(p) for p in batch], device)
            if args.half and device.type == "cuda" and torch.is_floating_point(xb):
                xb = xb.to(dtype=torch.float16)
            with torch.inference_mode():
                z_vis = model({ModalityType.VISION: xb})[ModalityType.VISION]  # [B,D]
        del xb  # free now

        # OCR text
        ocr_texts = []
        for p in batch:
            with Image.open(p) as im:
                img_np = np.array(im.convert("RGB"))
            ocr_texts.append(ocr_image_np(img_np, use_gpu=args.ocr_gpu))
        z_txt = None
        if any(s.strip() for s in ocr_texts):
            with torch.cuda.amp.autocast(enabled=args.half and device.type == "cuda", dtype=torch.float16):
                T = data.load_and_transform_text(ocr_texts, device)
                with torch.inference_mode():
                    z_txt = model({ModalityType.TEXT: T})[ModalityType.TEXT]  # [B,D]
            del T  # free now

        # Fuse & save
        with torch.no_grad():
            for j, p in enumerate(batch):
                v = z_vis[j]
                t = z_txt[j] if z_txt is not None else None
                fused = fuse(v, t, args.w_vision, args.w_ocr)
                save_vec(fused, emb_dir, p.stem)
                print(f"[done] image: {p} → {emb_dir / (p.stem + '.npy')}")
                del fused, v, t
        del z_vis, z_txt

        # Clear caches every N batches
        if ((bi // args.batch) + 1) % args.clear_cache_every == 0:
            clear_cuda(device, do_ipc=args.cuda_ipc_collect)

    # ---------- VIDEOS (robust, one-by-one, with Option B) ----------
    for p in vid_paths:
        orig_path = str(p)
        safe_path, tmp_path = ensure_h264(orig_path)

        if not has_video_stream(safe_path):
            print(f"[skip] no usable video stream: {orig_path}")
            if tmp_path and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
            continue

        z_vis = None
        try:
            z_vis = embed_video_with_decord(model, device, safe_path, clips_per_video=1, clip_duration=2, half=args.half)
        except Exception as e1:
            print(f"[warn] Decord failed on {orig_path}: {e1}\nTrying PyAV…")
            try:
                z_vis = embed_video_with_pyav(model, device, safe_path, half=args.half)
            except Exception as e2:
                print(f"[warn] PyAV failed on {orig_path}: {e2}\nTrying OpenCV…")
                try:
                    z_vis = embed_video_with_opencv(model, device, safe_path, half=args.half)
                except Exception as e3:
                    print(f"[skip] OpenCV fallback also failed on {orig_path}: {e3}")
                    if tmp_path and os.path.exists(tmp_path):
                        try: os.remove(tmp_path)
                        except: pass
                    continue

        # OCR on several frames
        try:
            frames = sample_video_frames_for_ocr(safe_path, num_frames=args.ocr_video_frames)
            ocr_text = " ".join([ocr_image_np(f, use_gpu=args.ocr_gpu) for f in frames if f is not None]) or " "
            with torch.cuda.amp.autocast(enabled=args.half and device.type == "cuda", dtype=torch.float16):
                T = data.load_and_transform_text([ocr_text], device)
                with torch.inference_mode():
                    z_txt = model({ModalityType.TEXT: T})[ModalityType.TEXT].squeeze(0)
            del T
        except Exception as e:
            print(f"[warn] OCR failed on {orig_path}: {e}")
            z_txt = None

        fused = fuse(z_vis, z_txt, args.w_vision, args.w_ocr)
        save_vec(fused, emb_dir, p.stem)
        print(f"[done] video: {p} → {emb_dir / (p.stem + '.npy')}")
        del fused, z_vis, z_txt

        # Clear right after each video to avoid buildup
        clear_cuda(device, do_ipc=args.cuda_ipc_collect)

        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

    print(f"Done. Saved fused vectors (VISION + OCR-TEXT) to {emb_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Batch embed with ImageBind + OCR fusion (auto-transcode AV1 → H.264)")
    ap.add_argument("--images", type=str, help="Folder with images (recursive)")
    ap.add_argument("--videos", type=str, help="Folder with videos (recursive)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--weights", type=str, help="Path to .pth/.pt/.safetensors weights (optional)")
    ap.add_argument("--batch", type=int, default=32, help="Image batch size")
    ap.add_argument("--half", action="store_true", help="Use FP16 on GPU to cut VRAM")
    ap.add_argument("--w-vision", type=float, default=1.0, help="Fusion weight for VISION")
    ap.add_argument("--w-ocr", type=float, default=1.0, help="Fusion weight for OCR TEXT")
    ap.add_argument("--ocr-video-frames", type=int, default=6, help="Frames to OCR per video")
    ap.add_argument("--ocr-gpu", action="store_true", help="Run EasyOCR on GPU")
    ap.add_argument("--clear-cache-every", type=int, default=1, help="How often (in image batches) to clear CUDA cache")
    ap.add_argument("--cuda-ipc-collect", action="store_true", help="Also call torch.cuda.ipc_collect() when clearing cache")
    args = ap.parse_args()
    main(args)
