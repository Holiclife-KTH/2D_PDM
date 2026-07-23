"""
SimilarityStream Validation 스크립트

Usage:
  # 단일 이미지 검증
  python validate_similarity_stream.py --scene path/to/scene.png

  # 여러 이미지 일괄 검증 (디렉토리 지정, --n 개수 샘플링)
  python validate_similarity_stream.py --scene_dir path/to/rgb_dir --n 12

  # GT가 있는 경우 자동으로 비교 시각화
  python validate_similarity_stream.py --scene_dir path/to/rgb_dir --gt_dir path/to/gt_dir --n 6
"""

import sys
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from similarity_stream import SimilarityStream, load_image_tensor, make_fg_mask


# ── 기본 경로 ──────────────────────────────────────────────────────────────────

_ROOT = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws"

DEFAULT_TARGET    = f"{_ROOT}/data/target/packaged_food_2/000000_right.png"
DEFAULT_LOCAL_PTH = f"{_ROOT}/model/dinov3_vit7b16_hf_converted.pth"
DEFAULT_CKPT      = f"{_ROOT}/checkpoints_multi/best.pt"

DEFAULT_SCENE_DIR = f"{_ROOT}/data/scene/packaged_food_2/scene/rgb"
DEFAULT_GT_DIR    = f"{_ROOT}/data/similarity/distribution_map(packaged_food_2_beta0.0)/distribution_map"


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def build_model(ckpt_path: str, dino_pth: str, device: str) -> SimilarityStream:
    """SimilarityStream 초기화 후 checkpoint (비-backbone 부분) 로드."""
    stream = SimilarityStream(
        dino_model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        dino_model_path=dino_pth,
        num_classes=1,
        freeze_dino=True,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = stream.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  [WARN] missing keys : {len(missing)}")
    if unexpected:
        print(f"  [WARN] unexpected   : {len(unexpected)}")
    print(f"  [INFO] checkpoint: epoch={ckpt['epoch']}  val_mse={ckpt['val_loss']:.6f}")

    stream.eval()
    return stream


# ── Target feature 사전 계산 ──────────────────────────────────────────────────

def cache_target(stream: SimilarityStream, target_path: str, device: str) -> tuple:
    """Target 이미지를 DINOv3로 한 번만 통과 → patch/CLS features + FG mask 캐싱."""
    patch_size = stream.matching.extractor.patch_size
    target_t, _ = load_image_tensor(target_path, patch_size, device)
    tgt_mask = make_fg_mask(target_t, patch_size)           # [1, 1, H_p, W_p]
    fg_ratio = tgt_mask.mean().item()
    print(f"  [INFO] target FG mask: {fg_ratio*100:.1f}% foreground 패치")
    with torch.no_grad():
        patch_feats, cls_feats = stream.matching.extractor(target_t)
    cached_patch = {b: f.squeeze(0) for b, f in patch_feats.items()}  # {b: [D,H_p,W_p]}
    cached_cls   = {b: c.squeeze(0) for b, c in cls_feats.items()}    # {b: [D]}
    cached_mask  = tgt_mask.squeeze(0)                                 # [1, H_p, W_p]
    return cached_patch, cached_cls, cached_mask


def expand_target(cached: dict, B: int) -> dict:
    return {b: f.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            for b, f in cached.items()}


def expand_cls(cached_cls: dict, B: int) -> dict:
    return {b: c.unsqueeze(0).expand(B, -1).contiguous()
            for b, c in cached_cls.items()}


def expand_mask(cached_mask: torch.Tensor, B: int) -> torch.Tensor:
    return cached_mask.unsqueeze(0).expand(B, -1, -1, -1).contiguous()


# ── 추론 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(
    stream: SimilarityStream,
    scene_path: str,
    cached_patch: dict,
    cached_cls: dict,
    cached_mask: torch.Tensor,
    device: str,
) -> tuple:
    """
    Returns:
        prob_np:    [H, W]  float [0,1]  — sigmoid(S_pred)
        scene_np:   [H, W, 3] float [0,1] — scene 이미지
    """
    patch_size = stream.matching.extractor.patch_size
    scene_t, (H, W) = load_image_tensor(scene_path, patch_size, device)

    S_pred, _ = stream(
        scene_t,
        target_feats=expand_target(cached_patch, 1),
        target_cls=expand_cls(cached_cls, 1),
        target_mask=expand_mask(cached_mask, 1),
    )
    prob_np  = torch.sigmoid(S_pred[0, 0]).cpu().numpy()   # [H, W]
    scene_np = scene_t[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    return prob_np, scene_np


# ── 시각화 ────────────────────────────────────────────────────────────────────

def load_gt(gt_path: str, H: int, W: int) -> np.ndarray:
    """GT grayscale PNG → float [0,1] numpy [H, W]."""
    gt = Image.open(gt_path).convert("L")
    if (gt.width, gt.height) != (W, H):
        gt = gt.resize((W, H), Image.BILINEAR)
    return np.asarray(gt, dtype=np.float32) / 255.0


def make_overlay(scene_np: np.ndarray, prob_np: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """scene 위에 heatmap을 alpha blend."""
    cmap    = plt.get_cmap("hot")
    heat    = cmap(prob_np)[:, :, :3]   # [H, W, 3]
    return np.clip(scene_np * (1 - alpha) + heat * alpha, 0, 1)


def visualize_single(
    scene_path: str,
    prob_np: np.ndarray,
    scene_np: np.ndarray,
    target_np: np.ndarray,
    gt_np: np.ndarray | None = None,
    save_path: str | None = None,
):
    """단일 scene 결과 시각화."""
    has_gt  = gt_np is not None
    n_cols  = 5 if has_gt else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    fig.suptitle(Path(scene_path).name, fontsize=11)

    col = 0
    axes[col].imshow(target_np)
    axes[col].set_title("Target")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(scene_np)
    axes[col].set_title("Scene")
    axes[col].axis("off")
    col += 1

    if has_gt:
        im = axes[col].imshow(gt_np, cmap="hot", vmin=0, vmax=1)
        axes[col].set_title("GT distribution")
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
        col += 1

    im2 = axes[col].imshow(prob_np, cmap="hot", vmin=0, vmax=1)
    axes[col].set_title("S_pred (sigmoid)")
    axes[col].axis("off")
    plt.colorbar(im2, ax=axes[col], fraction=0.046, pad=0.04)
    col += 1

    axes[col].imshow(make_overlay(scene_np, prob_np))
    axes[col].set_title("Overlay (α=0.5)")
    axes[col].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.show()


def visualize_grid(
    results: list,
    target_np: np.ndarray,
    save_path: str | None = None,
):
    """
    results: list of (scene_name, prob_np, scene_np, gt_np or None)
    그리드: 각 row = [scene | GT | S_pred | overlay]  (GT 없으면 3열)
    """
    has_gt  = any(r[3] is not None for r in results)
    n_cols  = 4 if has_gt else 3
    n_rows  = len(results)

    fig = plt.figure(figsize=(5 * n_cols, 3.8 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.05, wspace=0.05)

    col_titles = ["Scene", "GT distribution", "S_pred (sigmoid)", "Overlay"]
    if not has_gt:
        col_titles = ["Scene", "S_pred (sigmoid)", "Overlay"]

    for ci, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_title(title, fontsize=10, pad=4)
        ax.axis("off")

    for ri, (name, prob_np, scene_np, gt_np) in enumerate(results):
        col = 0

        ax = fig.add_subplot(gs[ri, col]); col += 1
        ax.imshow(scene_np)
        ax.set_ylabel(name, fontsize=7, rotation=0, labelpad=60, va="center")
        ax.axis("off")

        if has_gt:
            ax = fig.add_subplot(gs[ri, col]); col += 1
            ax.imshow(gt_np if gt_np is not None else np.zeros_like(prob_np),
                      cmap="hot", vmin=0, vmax=1)
            ax.axis("off")

        ax = fig.add_subplot(gs[ri, col]); col += 1
        ax.imshow(prob_np, cmap="hot", vmin=0, vmax=1)
        ax.axis("off")

        ax = fig.add_subplot(gs[ri, col]); col += 1
        ax.imshow(make_overlay(scene_np, prob_np))
        ax.axis("off")

    # colorbar 대신 범례 텍스트
    fig.text(0.5, 0.01, "heatmap: hot colormap  [0=low, 1=high similarity]",
             ha="center", fontsize=9, color="gray")

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  saved → {save_path}")
    plt.show()


# ── 메트릭 ────────────────────────────────────────────────────────────────────

def compute_metrics(prob_np: np.ndarray, gt_np: np.ndarray) -> dict:
    """
    예측값과 GT 간의 수치 지표 계산.

    반환 항목:
      mse, mae, psnr_dB          — 오차 지표
      pred_mean/std/max/min      — 예측값 분포
      gt_mean/std/max/min        — GT 분포
      gt_nonzero_ratio           — GT에서 0 초과인 픽셀 비율
      pearson_r                  — 선형 상관계수 (공간 분포 일치도)
      iou_05                     — threshold=0.5 기준 binary IoU
    """
    diff = prob_np - gt_np
    mse  = float(np.mean(diff ** 2))
    mae  = float(np.mean(np.abs(diff)))
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf")

    # 분포 통계
    pred_flat = prob_np.ravel()
    gt_flat   = gt_np.ravel()

    # Pearson 상관계수
    if pred_flat.std() > 1e-8 and gt_flat.std() > 1e-8:
        pearson_r = float(np.corrcoef(pred_flat, gt_flat)[0, 1])
    else:
        pearson_r = 0.0

    # Binary IoU @ threshold 0.5
    pred_bin = (prob_np >= 0.5)
    gt_bin   = (gt_np   >= 0.5)
    inter    = (pred_bin & gt_bin).sum()
    union    = (pred_bin | gt_bin).sum()
    iou_05   = float(inter / union) if union > 0 else float("nan")

    return {
        "mse":              mse,
        "mae":              mae,
        "psnr_dB":          psnr,
        "pearson_r":        pearson_r,
        "iou_05":           iou_05,
        "pred_mean":        float(pred_flat.mean()),
        "pred_std":         float(pred_flat.std()),
        "pred_max":         float(pred_flat.max()),
        "pred_min":         float(pred_flat.min()),
        "gt_mean":          float(gt_flat.mean()),
        "gt_std":           float(gt_flat.std()),
        "gt_max":           float(gt_flat.max()),
        "gt_nonzero_ratio": float((gt_flat > 0).mean()),
    }


# ── 수치 데이터 저장 ──────────────────────────────────────────────────────────

def save_numerical_data(
    save_dir: Path,
    results: list,
    metrics_rows: list,
):
    """
    결과를 numerical 포맷으로 저장.

    save_dir/
      data/
        <scene_name>.npz   — pred [H,W], gt [H,W] float32 배열
      metrics_summary.csv  — 이미지별 통계 요약
      all_preds.npz        — 전체 pred/gt 묶음 (샘플 수 적을 때)

    npz 로드 예시:
      d = np.load('data/scene00001_env0000_bottom.npz')
      pred = d['pred']   # [H, W] float32  sigmoid 출력
      gt   = d['gt']     # [H, W] float32  GT / 255
    """
    import csv

    data_dir = save_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_pred_list, all_gt_list, all_names = [], [], []

    for name, prob_np, _, gt_np in results:
        stem = Path(name).stem
        npz_path = data_dir / f"{stem}.npz"

        save_kwargs = {"pred": prob_np.astype(np.float32)}
        if gt_np is not None:
            save_kwargs["gt"] = gt_np.astype(np.float32)
        np.savez_compressed(npz_path, **save_kwargs)

        all_pred_list.append(prob_np.astype(np.float32))
        if gt_np is not None:
            all_gt_list.append(gt_np.astype(np.float32))
        all_names.append(name)

    print(f"  [SAVE] {len(results)}개 .npz  →  {data_dir}")

    # metrics_summary.csv
    if metrics_rows:
        csv_path = save_dir / "metrics_summary.csv"
        fieldnames = ["scene_name"] + list(metrics_rows[0][1].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for name, m in metrics_rows:
                row = {"scene_name": name}
                row.update({k: f"{v:.8f}" if isinstance(v, float) else v
                            for k, v in m.items()})
                w.writerow(row)

        # 요약 통계 (마지막 행)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            vals = {k: [r[1][k] for r in metrics_rows if not np.isnan(r[1][k])]
                    for k in metrics_rows[0][1].keys()}
            w.writerow([])
            w.writerow(["# mean"] + [f"{np.mean(v):.8f}" if v else "nan"
                                      for v in vals.values()])
            w.writerow(["# std"]  + [f"{np.std(v):.8f}"  if v else "nan"
                                      for v in vals.values()])
        print(f"  [SAVE] metrics_summary.csv  →  {csv_path}")

    # all_preds.npz — 전체 묶음 (분석 편의용)
    if all_pred_list:
        bundle_kwargs = {
            "preds": np.stack(all_pred_list),          # [N, H, W]
            "names": np.array(all_names),              # [N]
        }
        if all_gt_list and len(all_gt_list) == len(all_pred_list):
            bundle_kwargs["gts"] = np.stack(all_gt_list)  # [N, H, W]
        np.savez_compressed(save_dir / "all_preds.npz", **bundle_kwargs)
        print(f"  [SAVE] all_preds.npz  shape={np.stack(all_pred_list).shape}"
              f"  →  {save_dir / 'all_preds.npz'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SimilarityStream validation")
    p.add_argument("--scene",      type=str, default=None,
                   help="단일 scene 이미지 경로")
    p.add_argument("--scene_dir",  type=str, default=DEFAULT_SCENE_DIR,
                   help="scene 이미지 디렉토리 (--scene 미지정 시 사용)")
    p.add_argument("--gt_dir",     type=str, default=DEFAULT_GT_DIR,
                   help="GT distribution map 디렉토리 (없으면 비교 생략)")
    p.add_argument("--target",     type=str, default=DEFAULT_TARGET)
    p.add_argument("--ckpt",       type=str, default=DEFAULT_CKPT)
    p.add_argument("--dino_pth",   type=str, default=DEFAULT_LOCAL_PTH)
    p.add_argument("--n",          type=int, default=6,
                   help="디렉토리 모드 시 랜덤 샘플 개수")
    p.add_argument("--save_dir",   type=str, default=None,
                   help="결과 이미지 저장 폴더 (None이면 저장 안 함)")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    random.seed(args.seed)

    # ── 모델 초기화 ──────────────────────────────────────────────────────────
    print("[INFO] loading model ...")
    stream                                    = build_model(args.ckpt, args.dino_pth, device)
    cached_target, cached_cls, cached_mask   = cache_target(stream, args.target, device)
    patch_size                               = stream.matching.extractor.patch_size

    # target 이미지 numpy (시각화용)
    tgt_img   = Image.open(args.target).convert("RGB")
    target_np = np.asarray(tgt_img.resize(
        ((tgt_img.width  // patch_size) * patch_size,
         (tgt_img.height // patch_size) * patch_size),
        Image.BILINEAR
    ), dtype=np.float32) / 255.0

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── 검증 대상 이미지 목록 결정 ────────────────────────────────────────────
    if args.scene:
        scene_paths = [Path(args.scene)]
    else:
        scene_paths = sorted(Path(args.scene_dir).glob("*.png"))
        if len(scene_paths) > args.n:
            scene_paths = random.sample(scene_paths, args.n)
        scene_paths = sorted(scene_paths)

    print(f"[INFO] validating {len(scene_paths)} image(s) ...")

    gt_dir        = Path(args.gt_dir)
    results       = []
    metrics_rows  = []   # [(scene_name, metrics_dict), ...]

    for sp in scene_paths:
        prob_np, scene_np = infer(stream, str(sp), cached_target, cached_cls, cached_mask, device)
        H, W              = prob_np.shape

        gt_path = gt_dir / sp.name
        gt_np   = load_gt(str(gt_path), H, W) if gt_path.exists() else None

        results.append((sp.name, prob_np, scene_np, gt_np))

        if gt_np is not None:
            m = compute_metrics(prob_np, gt_np)
            metrics_rows.append((sp.name, m))
            print(f"  {sp.name:40s}  "
                  f"MSE={m['mse']:.5f}  MAE={m['mae']:.5f}  "
                  f"PSNR={m['psnr_dB']:.2f}dB  "
                  f"PearsonR={m['pearson_r']:.4f}  IoU@0.5={m['iou_05']:.4f}")
        else:
            print(f"  {sp.name:40s}  (GT 없음)")

    # ── 평균 메트릭 출력 ──────────────────────────────────────────────────────
    if metrics_rows:
        keys = list(metrics_rows[0][1].keys())
        print(f"\n{'─'*65}")
        for k in ["mse", "mae", "psnr_dB", "pearson_r", "iou_05"]:
            vals = [r[1][k] for r in metrics_rows if not np.isnan(r[1][k])]
            print(f"  {k:15s}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")
        print(f"{'─'*65}")

    # ── 수치 데이터 저장 ──────────────────────────────────────────────────────
    if save_dir:
        save_numerical_data(save_dir, results, metrics_rows)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    if len(results) == 1:
        name, prob_np, scene_np, gt_np = results[0]
        save_path = str(save_dir / f"val_{name}") if save_dir else None
        visualize_single(str(scene_paths[0]), prob_np, scene_np, target_np, gt_np, save_path)
    else:
        save_path = str(save_dir / "val_grid.png") if save_dir else None
        visualize_grid(results, target_np, save_path)
