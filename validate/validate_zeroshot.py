"""
ZeroShotPipeline (proj + head) 학습 결과 검증 스크립트.

체크포인트에서 proj_state + head_state를 복원하고,
seg 기반 GT와 비교해 MSE / Pearson r / 이진 정확도를 계산.

사용법:
    cd th_ws
    python validate/validate_zeroshot.py --ckpt checkpoints/zeroshot_<RUN>/zeroshot_best.pt
    python validate/validate_zeroshot.py --ckpt ... --scene fruit_1
    python validate/validate_zeroshot.py --ckpt ... --max_samples 200
"""

import argparse
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---- src 경로 설정 ----
_VAL_DIR = os.path.dirname(os.path.abspath(__file__))
_TH_WS   = os.path.dirname(_VAL_DIR)
sys.path.insert(0, os.path.join(_TH_WS, "src"))

from zeroshot_pipeline import ZeroShotPipeline, DINOV3_PATCH_SZ
from gt_builder import (
    build_color_to_score, compute_gt, load_scene_mapping, render_gt_map,
    load_scene_config,
)

# --------------------------------------------------------------------------- #
# 기본 경로 / 설정
# --------------------------------------------------------------------------- #
DATA_ROOT         = os.path.join(_TH_WS, "data")
TARGET_ROOT       = os.path.join(DATA_ROOT, "target")
CKPT_DIR          = os.path.join(_TH_WS, "checkpoints")
SCENE_CONFIG_PATH = os.path.join(_TH_WS, "config", "scenes.yaml")
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --------------------------------------------------------------------------- #
# 데이터 탐색
# --------------------------------------------------------------------------- #

def discover_samples(scene_name: str) -> list[tuple[str, str, str]]:
    """(rgb_path, seg_path, scene_id) 반환 — seg 파일 있는 것만."""
    rgb_dir = os.path.join(DATA_ROOT, "scene", scene_name, "scene", "rgb")
    seg_dir = os.path.join(DATA_ROOT, "scene", scene_name, "scene", "seg")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(seg_dir):
        return []
    out = []
    for fname in sorted(os.listdir(rgb_dir)):
        if not fname.endswith(".png"):
            continue
        seg_path = os.path.join(seg_dir, fname)
        if os.path.isfile(seg_path):
            scene_id = fname.split("_")[0]
            out.append((os.path.join(rgb_dir, fname), seg_path, scene_id))
    return out


def build_scene_score_cache(scene_name: str,
                             target_category: str,
                             target_usd_name: str) -> dict[tuple, dict]:
    """(scene_id, target_usd) → color_to_score 캐시."""
    seg_dir = os.path.join(DATA_ROOT, "scene", scene_name, "scene", "seg")
    cache = {}
    if not os.path.isdir(seg_dir):
        return cache
    for fname in os.listdir(seg_dir):
        if not fname.endswith("_mapping.json"):
            continue
        scene_id = fname.replace("_mapping.json", "")
        mpath = os.path.join(seg_dir, fname)
        mapping = load_scene_mapping(mpath)
        cache[(scene_id, target_usd_name)] = build_color_to_score(
            mapping, target_category, target_usd_name
        )
    return cache

# --------------------------------------------------------------------------- #
# 메트릭 계산
# --------------------------------------------------------------------------- #

def pearson_r(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pearson 상관계수 (공간 패턴 일치도)."""
    p = pred.flatten().astype(np.float64)
    g = gt.flatten().astype(np.float64)
    if p.std() < 1e-8 or g.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(p, g)[0, 1])


def binary_accuracy(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.5) -> float:
    """GT와 pred 모두 thresh 기준으로 이진화한 픽셀 정확도."""
    return float(((pred >= thresh) == (gt >= thresh)).mean())


def compute_metrics(pred_np: np.ndarray, gt_np: np.ndarray) -> dict[str, float]:
    mse  = float(np.mean((pred_np - gt_np) ** 2))
    pr   = pearson_r(pred_np, gt_np)
    bacc = binary_accuracy(pred_np, gt_np, thresh=0.5)
    return {"mse": mse, "pearson_r": pr, "bin_acc": bacc}

# --------------------------------------------------------------------------- #
# Scene 단위 validation
# --------------------------------------------------------------------------- #

def validate_scene(scene_name: str,
                   pipe: ZeroShotPipeline,
                   target_entry: dict,
                   score_cache: dict,
                   max_samples: int,
                   panel_save_path: str | None = None) -> dict | None:
    """
    한 씬 전체를 검증해서 평균 메트릭 반환.
    target_entry: {"bgr": ..., "label": ..., "category": ..., "usd_name": ...}
    """
    samples = discover_samples(scene_name)
    if not samples:
        print(f"  [SKIP] {scene_name}: rgb/seg 파일 없음")
        return None

    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)

    # 타겟 인코딩 (한 번만)
    target_cached = pipe.encode_target(target_entry["bgr"], label=target_entry["label"])

    agg = {"mse": 0.0, "pearson_r": 0.0, "bin_acc": 0.0}
    n = 0
    panel_sample = None  # (rgb, gt, pred) 첫 번째 유효 샘플

    pipe.head.eval()
    pipe.semantic.proj.eval()

    for rgb_path, seg_path, scene_id in tqdm(samples, desc=scene_name, leave=False):
        bgr = cv2.imread(rgb_path)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]

        # GT 계산
        c2s = score_cache.get((scene_id, target_entry["usd_name"]), {})
        seg = cv2.imread(seg_path)
        if seg is None:
            continue
        gt = render_gt_map(seg, c2s)  # (H, W) float32

        # 추론
        rgb_f = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        scene_t = ((torch.from_numpy(rgb_f).permute(2, 0, 1) - _DINO_MEAN) / _DINO_STD
                   ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            scene_feats = pipe.dino(scene_t)
            query = pipe.expand_query(target_cached, B=1)
            out = pipe.head(scene_feats, query, out_size=(H, W))

        pred_np = out["prob_full"][0, 0].cpu().numpy()  # (H, W) [0,1]

        metrics = compute_metrics(pred_np, gt)
        for k in agg:
            agg[k] += metrics[k]
        n += 1

        if panel_sample is None:
            panel_sample = (bgr, gt, pred_np)

    if n == 0:
        return None

    result = {k: v / n for k, v in agg.items()}
    result["n"] = n
    result["scene"] = scene_name

    # 패널 저장
    if panel_sample is not None and panel_save_path:
        _save_panel(panel_save_path, scene_name, target_entry, *panel_sample)

    return result

# --------------------------------------------------------------------------- #
# 패널 저장
# --------------------------------------------------------------------------- #

def _save_panel(out_path: str, scene_name: str, target_entry: dict,
                bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    H, W = bgr.shape[:2]

    def to_bgr(m):
        return cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR)

    def labeled(img, text, bar_h=32):
        bar = np.zeros((bar_h, img.shape[1], 3), np.uint8)
        cv2.putText(bar, text, (6, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    tgt_bgr = target_entry["bgr"]
    tgt_vis = cv2.resize(tgt_bgr, (H, H))

    mse_str = f"mse={compute_metrics(pred, gt)['mse']:.4f}"
    panel = np.hstack([
        labeled(tgt_vis,     f"TARGET ({target_entry['label']})"),
        labeled(bgr,         f"SCENE  ({scene_name})"),
        labeled(to_bgr(gt),  f"GT     ({target_entry['category']})"),
        labeled(to_bgr(pred), f"PRED   ({mse_str})"),
    ])
    cv2.imwrite(out_path, panel)

# --------------------------------------------------------------------------- #
# --target_img 모드: GT 없이 추론 패널만 저장
# --------------------------------------------------------------------------- #

def infer_custom_target(scene_name: str,
                        pipe: ZeroShotPipeline,
                        target_bgr: np.ndarray,
                        target_label: str | None,
                        max_samples: int,
                        out_dir: str):
    """
    임의 타겟 이미지로 씬 추론 — GT 없이 시각적 패널만 저장.
    """
    samples = discover_samples(scene_name)
    if not samples:
        print(f"  [SKIP] {scene_name}: rgb/seg 파일 없음")
        return

    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)

    target_cached = pipe.encode_target(target_bgr, label=target_label)
    label_str = target_label or "(image only)"

    pipe.head.eval()
    pipe.semantic.proj.eval()

    panels_saved = 0
    for rgb_path, _, _ in tqdm(samples, desc=scene_name, leave=False):
        bgr = cv2.imread(rgb_path)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]

        rgb_f = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        scene_t = ((torch.from_numpy(rgb_f).permute(2, 0, 1) - _DINO_MEAN) / _DINO_STD
                   ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            scene_feats = pipe.dino(scene_t)
            query = pipe.expand_query(target_cached, B=1)
            out = pipe.head(scene_feats, query, out_size=(H, W))

        pred_np = out["prob_full"][0, 0].cpu().numpy()

        # 패널: TARGET | SCENE | PRED  (GT 없음)
        if panels_saved < 5:   # 최대 5장
            fname      = os.path.splitext(os.path.basename(rgb_path))[0]
            scene_tag  = scene_name.replace("/", "-")
            _save_panel_no_gt(
                os.path.join(out_dir, f"panel_{scene_tag}_{fname}.png"),
                scene_name, target_bgr, label_str, bgr, pred_np,
            )
            panels_saved += 1

    print(f"  {scene_name}: 패널 {panels_saved}장 저장 → {out_dir}/")


def _save_panel_no_gt(out_path: str, scene_name: str,
                      target_bgr: np.ndarray, label_str: str,
                      scene_bgr: np.ndarray, pred: np.ndarray):
    """GT 없는 3-패널: TARGET | SCENE | PRED."""
    H, W = scene_bgr.shape[:2]

    def to_bgr(m):
        heat = cv2.applyColorMap(
            (np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        return heat

    def labeled(img, text, bar_h=32):
        bar = np.zeros((bar_h, img.shape[1], 3), np.uint8)
        cv2.putText(bar, text, (6, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    tgt_vis = cv2.resize(target_bgr, (H, H))
    panel = np.hstack([
        labeled(tgt_vis,          f"TARGET  ({label_str})"),
        labeled(scene_bgr,        f"SCENE   ({scene_name})"),
        labeled(to_bgr(pred),     f"PRED    (heatmap)"),
    ])
    cv2.imwrite(out_path, panel)


# --------------------------------------------------------------------------- #
# 메인
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="ZeroShotPipeline 검증")
    p.add_argument("--ckpt", required=True,
                   help="학습된 체크포인트 경로 (zeroshot_best.pt 등)")
    p.add_argument("--scene", default=None,
                   help="특정 씬만 검증 (예: fruit_1). 생략 시 전체")
    p.add_argument("--target", default=None,
                   help="타겟 override (예: Fruit/Apple). "
                        "생략 시 SCENE_TARGET_MAP 기본값 사용")
    p.add_argument("--target_usd", default=None,
                   help="씬 내 타겟 USD 이름 override (예: Apple). "
                        "--target 지정 시 같이 설정 권장")
    p.add_argument("--target_img", default=None,
                   help="임의 이미지를 타겟으로 직접 지정 (예: Test_dataset/Banana.png). "
                        "지정 시 --target/--target_usd 대신 이 이미지를 인코딩.")
    p.add_argument("--target_label", default=None,
                   help="--target_img 사용 시 SigLIP에 넘길 텍스트 (예: banana). "
                        "생략 시 이미지만 사용.")
    p.add_argument("--max_samples", type=int, default=200,
                   help="씬당 최대 샘플 수 (default: 200, 0=전체)")
    p.add_argument("--out", default=None,
                   help="결과 저장 디렉터리 (기본: ckpt 폴더 내 validate_<time>)")
    p.add_argument("--seed", type=int, default=42,
                   help="샘플 랜덤 선택 seed (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # ── 출력 경로 ──────────────────────────────────────────────────────────
    ckpt_dir = os.path.dirname(args.ckpt)
    out_dir  = args.out or os.path.join(
        ckpt_dir, f"validate_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"체크포인트: {args.ckpt}")
    print(f"출력 디렉터리: {out_dir}")
    print(f"device: {DEVICE}")

    # ── 파이프라인 로딩 & 체크포인트 복원 ────────────────────────────────
    print("\n[1/3] 파이프라인 로딩 ...")
    pipe = ZeroShotPipeline(device=DEVICE)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    pipe.semantic.proj.load_state_dict(ckpt["proj_state"])
    pipe.head.load_state_dict(ckpt["head_state"])
    pipe.semantic.proj.to(DEVICE).eval()
    pipe.head.to(DEVICE).eval()
    print(f"  복원 완료: epoch={ckpt.get('epoch', '?')}  "
          f"train_val_mse={ckpt.get('val_mse', '?'):.5f}" if "val_mse" in ckpt
          else f"  복원 완료: epoch={ckpt.get('epoch', '?')}")

    # ── scenes.yaml 로드 ─────────────────────────────────────────────────
    scene_entries = load_scene_config(SCENE_CONFIG_PATH)
    # scene_name → (target, usd) 빠른 조회용
    _scene_map = {e.scene: e for e in scene_entries}

    # ── --target_img 모드: 임의 이미지 타겟으로 씬 추론만 수행 ─────────────
    if args.target_img:
        if not os.path.isfile(args.target_img):
            sys.exit(f"타겟 이미지 파일 없음: {args.target_img}")
        target_bgr = cv2.imread(args.target_img)
        if target_bgr is None:
            sys.exit(f"이미지 로딩 실패: {args.target_img}")

        label_str = args.target_label
        print(f"\n[custom target] {args.target_img}")
        print(f"  label = {label_str!r}")

        if args.scene:
            scene_names = [args.scene]
        else:
            scene_names = [e.scene for e in scene_entries]

        print(f"\n[2/3] 타겟 이미지 인코딩 (label={label_str!r}) ...")
        print(f"\n[3/3] 씬 추론 ({len(scene_names)}개 씬, 씬당 최대 {args.max_samples}샘플) ...")
        for scene_name in scene_names:
            infer_custom_target(scene_name, pipe, target_bgr, label_str,
                                max_samples=args.max_samples or 0,
                                out_dir=out_dir)
        print(f"\n완료. 패널: {out_dir}/")
        return

    # ── 타겟 캐시 빌드 ─────────────────────────────────────────────────────
    print("\n[2/3] 타겟 인코딩 ...")
    target_cache = pipe.precompute_target_cache(TARGET_ROOT)

    # 검증할 씬 목록
    if args.scene:
        scene_names = [args.scene]
    else:
        scene_names = [e.scene for e in scene_entries]

    # ── 씬별 검증 ──────────────────────────────────────────────────────────
    print(f"\n[3/3] 검증 시작 ({len(scene_names)}개 씬, 씬당 최대 {args.max_samples}샘플) ...")
    results = []

    for scene_name in scene_names:
        # 타겟 결정: --target 인자 > scenes.yaml 기본값
        entry = _scene_map.get(scene_name)
        if args.target:
            cache_key = args.target
            usd_name  = args.target_usd or (entry.usd if entry else None)
        elif entry:
            cache_key = entry.target
            usd_name  = args.target_usd or entry.usd
        else:
            print(f"  [SKIP] {scene_name}: scenes.yaml에 없음 (--target 으로 지정 가능)")
            continue

        if cache_key not in target_cache:
            print(f"  [SKIP] {scene_name}: target 캐시 없음 ({cache_key})")
            continue

        meta = target_cache[cache_key]["meta"]
        target_entry = {
            "bgr":      cv2.imread(os.path.join(TARGET_ROOT,
                                                *cache_key.split("/"), "target.png")),
            "label":    target_cache[cache_key].get("prompt", cache_key),
            "category": meta["category"],
            "usd_name": usd_name,
        }

        score_cache = build_scene_score_cache(
            scene_name, meta["category"], usd_name
        )

        # 패널 파일명에 씬+타겟 조합 반영 ("/" → "-" for safe filename)
        scene_tag  = scene_name.replace("/", "-")
        tgt_tag    = cache_key.replace("/", "-")
        panel_path = os.path.join(out_dir, f"panel_{scene_tag}_tgt-{tgt_tag}.png")
        result = validate_scene(
            scene_name, pipe, target_entry, score_cache,
            max_samples=args.max_samples or 0,
            panel_save_path=panel_path,
        )
        if result is None:
            continue

        results.append(result)
        print(f"  {scene_name:20s}  n={result['n']:5d}  "
              f"mse={result['mse']:.5f}  "
              f"pearson_r={result['pearson_r']:+.4f}  "
              f"bin_acc={result['bin_acc']:.4f}")

    # ── 집계 ────────────────────────────────────────────────────────────────
    if not results:
        print("\n검증 가능한 씬이 없습니다.")
        return

    print("\n" + "=" * 72)
    print(f"{'SCENE':<22} {'N':>6}  {'MSE':>8}  {'PEARSON_R':>10}  {'BIN_ACC':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['scene']:<22} {r['n']:>6}  {r['mse']:>8.5f}  "
              f"{r['pearson_r']:>+10.4f}  {r['bin_acc']:>8.4f}")
    print("-" * 72)

    avg = {k: sum(r[k] for r in results) / len(results)
           for k in ("mse", "pearson_r", "bin_acc")}
    print(f"{'MEAN':<22} {'':>6}  {avg['mse']:>8.5f}  "
          f"{avg['pearson_r']:>+10.4f}  {avg['bin_acc']:>8.4f}")

    # ── 요약 파일 저장 ───────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"checkpoint : {args.ckpt}\n")
        f.write(f"epoch      : {ckpt.get('epoch', '?')}\n")
        f.write(f"val_mse    : {ckpt.get('val_mse', '?')}\n\n")
        f.write(f"{'SCENE':<22} {'N':>6}  {'MSE':>8}  {'PEARSON_R':>10}  {'BIN_ACC':>8}\n")
        f.write("-" * 72 + "\n")
        for r in results:
            f.write(f"{r['scene']:<22} {r['n']:>6}  {r['mse']:>8.5f}  "
                    f"{r['pearson_r']:>+10.4f}  {r['bin_acc']:>8.4f}\n")
        f.write("-" * 72 + "\n")
        f.write(f"{'MEAN':<22} {'':>6}  {avg['mse']:>8.5f}  "
                f"{avg['pearson_r']:>+10.4f}  {avg['bin_acc']:>8.4f}\n")

    print(f"\n요약 저장: {summary_path}")
    print(f"패널  저장: {out_dir}/panel_<scene>.png")


if __name__ == "__main__":
    main()
