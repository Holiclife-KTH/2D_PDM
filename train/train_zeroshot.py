"""
Zero-shot similarity map 학습 스크립트.

파이프라인: DINOv3 vits16 (appearance) + SigLIP so400m (semantic, Gemma3 vision encoder)
학습 대상:  pipe.semantic.proj (1.77M) + pipe.head (1.81M)  =  3.58M params
고정:       DINOv3 + SigLIP encoder  (모두 frozen)

Gradient 흐름:
    scene → DINOv3(frozen) → scene_feats ──────────────────────────────┐
    target.png → DINOv3(frozen)+SigLIP(frozen) → appearances, sem_raw  │
                                                      ↓                 │
                               semantic.proj (학습) → sem_proj          │
                                   appearance + sem_proj = query ───────┤
                                                         head (학습) → prob → loss

데이터 구조:
    data/scene/<name>/scene/rgb/<fname>.png
    data/scene/<name>/scene/seg/<fname>.png          (seg, GT 계산용)
    data/scene/<name>/scene/seg/<scene_id>_mapping.json
    data/target/<Category>/<SpecificObject>/target.png

GT 방식 (sky_ws 동일):
    seg 이미지 + mapping.json → USD 이름 → 카테고리 → SIMILARITY_MAP 점수
    예) 타겟=fruit: fruit픽셀=0.8, packaged_food=0.5, book/toy=0.2, bg=0.0

SCENE_TARGET_MAP 에 "scene 폴더명" → "Category/SpecificObject" 매핑을 추가해서 사용.

실행:
    cd th_ws
    python train/train_zeroshot.py
    python train/train_zeroshot.py --epochs 50 --batch 32
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# zeroshot_pipeline.py import 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from zeroshot_pipeline import (
    DINOV3_LAYERS, DINOV3_PATCH_SZ,
    ZeroShotPipeline, discover_target_entries,
)
from gt_builder import (
    build_color_to_score, compute_gt, load_scene_mapping, render_gt_map,
    load_scene_config,
)

# --------------------------------------------------------------------------- #
# 하이퍼파라미터
# --------------------------------------------------------------------------- #
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
LR         = 1e-3
EPOCHS     = 200
TRAIN_RATIO = 0.8
SPLIT_SEED  = None     # None → 매 실행 랜덤, 정수 → 재현 가능

EARLY_STOP_PATIENCE      = 10
EARLY_STOP_MIN_DELTA_PCT = 0.05   # 5% 이상 개선돼야 best로 인정

SAVE_INTERVAL = 5      # N epoch마다 last checkpoint 저장

# --------------------------------------------------------------------------- #
# 경로
# --------------------------------------------------------------------------- #
_TRAIN_DIR  = os.path.dirname(os.path.abspath(__file__))
_TH_WS      = os.path.dirname(_TRAIN_DIR)
DATA_ROOT   = os.path.join(_TH_WS, "data")
TARGET_ROOT = os.path.join(DATA_ROOT, "target")
OUT_DIR     = os.path.join(_TH_WS, "checkpoints")
SCENE_CONFIG_PATH = os.path.join(_TH_WS, "config", "scenes.yaml")

# Scene ↔ Target 매핑은 config/scenes.yaml 에서 로드 (main() 진입 후)
# 새 씬 추가 시 scenes.yaml 만 수정하면 됩니다.

# --------------------------------------------------------------------------- #
# Early Stopping
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """val_loss가 이전 최고 대비 min_delta_pct(%) 이상 줄어들 때만 개선으로 인정."""

    def __init__(self, patience: int = 10, min_delta_pct: float = 0.05):
        self.patience = patience
        self.min_delta_pct = min_delta_pct
        self.best_loss  = None
        self.counter    = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """새 최고 기록이면 True (best checkpoint 저장 트리거)."""
        if self.best_loss is None:
            self.best_loss = val_loss
            return True

        threshold = self.best_loss * (1.0 - self.min_delta_pct)
        if val_loss < threshold:
            pct = (self.best_loss - val_loss) / self.best_loss * 100
            print(f"    ✓ val_loss 개선: {self.best_loss:.6f} → {val_loss:.6f} ({pct:.1f}% 감소)")
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        print(f"    EarlyStopping: {self.counter}/{self.patience}  "
              f"(best={self.best_loss:.6f}, 이번={val_loss:.6f})")
        if self.counter >= self.patience:
            self.early_stop = True
        return False

# --------------------------------------------------------------------------- #
# 데이터셋
# --------------------------------------------------------------------------- #

def _discover_samples_seg(scene_name: str) -> list[tuple[str, str, str]]:
    """
    (rgb_path, seg_path, scene_id) 목록 반환.
    seg 파일이 존재하는 샘플만 포함 (seg → GT 계산에 필요).
    """
    rgb_dir = os.path.join(DATA_ROOT, "scene", scene_name, "scene", "rgb")
    seg_dir = os.path.join(DATA_ROOT, "scene", scene_name, "scene", "seg")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(seg_dir):
        return []
    triples = []
    for fname in sorted(os.listdir(rgb_dir)):
        if not fname.endswith(".png"):
            continue
        seg_path = os.path.join(seg_dir, fname)
        if not os.path.isfile(seg_path):
            continue
        scene_id = fname.split("_")[0]   # "scene00001"
        triples.append((os.path.join(rgb_dir, fname), seg_path, scene_id))
    return triples


def _split_by_scene(triples: list, ratio: float, seed: int | None):
    """scene 번호 단위로 분할해서 leakage 방지. tuple[2] == scene_id."""
    scene_map: dict[str, list] = {}
    for item in triples:
        scene_map.setdefault(item[2], []).append(item)

    scene_ids = sorted(scene_map.keys())
    if seed is None:
        seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
    random.Random(seed).shuffle(scene_ids)

    n_train = round(len(scene_ids) * ratio)
    train_ids = scene_ids[:n_train]
    val_ids   = scene_ids[n_train:]

    train_triples = [p for sid in train_ids for p in scene_map[sid]]
    val_triples   = [p for sid in val_ids   for p in scene_map[sid]]
    return train_triples, val_triples, seed


_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SceneDataset(Dataset):
    """(scene_tensor, gt_tensor, target_key) 반환.

    samples 항목: (rgb_path, seg_path, scene_id, target_key, target_category, target_usd_name)
    GT = seg + mapping.json → 카테고리 유사도 점수 (sky_ws 방식):
        same object = 1.0 / same category = 0.8 / similar = 0.5 / different = 0.2
    """

    def __init__(self, samples: list[tuple]):
        self.samples = samples

        # (scene_id, target_usd_name) → {(B,G,R): score} 미리 빌드 (worker 안전)
        self.score_cache: dict[tuple, dict] = {}
        for _, seg_path, scene_id, _, target_category, target_usd_name in samples:
            key = (scene_id, target_usd_name)
            if key in self.score_cache:
                continue
            seg_dir = os.path.dirname(seg_path)
            mpath = os.path.join(seg_dir, f"{scene_id}_mapping.json")
            if os.path.isfile(mpath):
                mapping = load_scene_mapping(mpath)
                self.score_cache[key] = build_color_to_score(
                    mapping, target_category, target_usd_name
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, seg_path, scene_id, target_key, target_category, target_usd_name = self.samples[idx]

        bgr = cv2.imread(rgb_path)
        seg = cv2.imread(seg_path)

        # GT: seg → 카테고리 유사도 맵 [0, 1]
        c2s = self.score_cache.get((scene_id, target_usd_name), {})
        gt = render_gt_map(seg, c2s) if seg is not None else np.zeros(bgr.shape[:2], np.float32)

        # (3, H, W) float, ImageNet-normalized
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        scene_t = (torch.from_numpy(rgb).permute(2, 0, 1) - _DINO_MEAN) / _DINO_STD
        gt_t = torch.from_numpy(gt)   # (H, W) float32

        return scene_t, gt_t, target_key


def _collate(batch):
    scenes = torch.stack([b[0] for b in batch])   # (B, 3, H, W)
    gts    = torch.stack([b[1] for b in batch])   # (B, H, W)
    keys   = [b[2] for b in batch]                # List[str]
    return scenes, gts, keys

# --------------------------------------------------------------------------- #
# Query 조립 (학습 중 proj 실시간 적용 → gradient 보장)
# --------------------------------------------------------------------------- #

def assemble_query(pipe: ZeroShotPipeline,
                   target_cache: dict,
                   target_keys: list[str]) -> list[torch.Tensor]:
    """
    target_cache의 frozen 임베딩에 proj(학습 대상)를 실시간으로 적용.
    query_vecs(캐싱된 것)를 재사용하면 proj의 gradient가 끊기기 때문에 매 step 재계산.

    반환: List[(B, 384)] × num_layers
    """
    num_layers = len(DINOV3_LAYERS)
    batch_queries = []
    for key in target_keys:
        cached = target_cache[key]
        # appearances: List[(384,)] from frozen DINOv3  — no grad needed
        # semantic_raw: (1152,)    from frozen SigLIP   — no grad needed
        apps    = [a.to(DEVICE) for a in cached["appearances"]]          # [(384,)] × n
        sem_raw = cached["semantic_raw"].unsqueeze(0).to(DEVICE)         # (1, 1152)

        # proj는 학습 대상 → 여기서 적용해야 gradient 흐름
        sem_projs = [pipe.semantic.proj[li](sem_raw) for li in range(num_layers)]  # [(1,384)]

        query = [(a.unsqueeze(0) + sp).contiguous()
                 for a, sp in zip(apps, sem_projs)]   # [(1, 384)] × n
        batch_queries.append(query)

    # List[(B, 384)] × num_layers
    return [
        torch.cat([bq[li] for bq in batch_queries], dim=0)
        for li in range(num_layers)
    ]

# --------------------------------------------------------------------------- #
# 한 epoch 실행
# --------------------------------------------------------------------------- #

def run_epoch(pipe: ZeroShotPipeline,
              loader: DataLoader,
              target_cache: dict,
              optim=None,
              desc: str = "") -> tuple[float, float]:
    """(avg_mse, avg_rmse) 반환. optim=None이면 validation."""
    train_mode = optim is not None
    pipe.head.train(train_mode)
    pipe.semantic.proj.train(train_mode)

    total_loss, total_n = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")

    for scene_batch, gt_batch, target_keys in pbar:
        scene_batch = scene_batch.to(DEVICE)   # (B, 3, H, W)
        gt_batch    = gt_batch.to(DEVICE)      # (B, H, W)

        # Scene features (DINOv3 frozen, no_grad 내부 처리됨)
        scene_feats = pipe.dino(scene_batch)   # [(patch(B,C,Hp,Wp), cls), ...]

        # GT를 patch 해상도로 다운샘플
        gt_patch = F.avg_pool2d(
            gt_batch.unsqueeze(1), kernel_size=DINOV3_PATCH_SZ, stride=DINOV3_PATCH_SZ
        )   # (B, 1, Hp, Wp)

        # Query 조립 (proj 실시간 적용)
        query = assemble_query(pipe, target_cache, target_keys)

        if train_mode:
            out = pipe.head(scene_feats, query)
            loss = F.mse_loss(out["prob_patch"], gt_patch)
            optim.zero_grad()
            loss.backward()
            optim.step()
        else:
            with torch.no_grad():
                out = pipe.head(scene_feats, query)
                loss = F.mse_loss(out["prob_patch"], gt_patch)

        B = scene_batch.shape[0]
        total_loss += loss.item() * B
        total_n    += B
        pbar.set_postfix(mse=f"{loss.item():.5f}")

    avg_mse  = total_loss / total_n
    avg_rmse = avg_mse ** 0.5
    return avg_mse, avg_rmse

# --------------------------------------------------------------------------- #
# Qualitative panel 저장
# --------------------------------------------------------------------------- #

def save_panel(pipe: ZeroShotPipeline,
               val_samples: list,
               target_cache: dict,
               out_path: str,
               epoch: int):
    """무작위 val 샘플로 TARGET | SCENE | GT | PRED 4-패널 저장."""
    rgb_path, seg_path, scene_id, target_key, target_category, target_usd_name = random.choice(val_samples)
    bgr = cv2.imread(rgb_path)
    H, W = bgr.shape[:2]

    # GT — seg 기반 카테고리 유사도 맵 (same-object=1.0 포함)
    seg_dir = os.path.dirname(seg_path)
    mapping_path = os.path.join(seg_dir, f"{scene_id}_mapping.json")
    gt = compute_gt(seg_path, mapping_path, target_category, target_usd_name)
    if gt is None:
        gt = np.zeros((H, W), dtype=np.float32)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    scene_t = ((torch.from_numpy(rgb).permute(2, 0, 1) - _DINO_MEAN) / _DINO_STD
               ).unsqueeze(0).to(DEVICE)

    pipe.head.eval()
    pipe.semantic.proj.eval()
    with torch.no_grad():
        scene_feats = pipe.dino(scene_t)
        query = assemble_query(pipe, target_cache, [target_key])
        out = pipe.head(scene_feats, query, out_size=(H, W))
    pred_np = out["prob_full"][0, 0].cpu().numpy()

    def to_bgr(m):
        return cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def labeled(img, text, bar_h=32):
        bar = np.zeros((bar_h, img.shape[1], 3), np.uint8)
        cv2.putText(bar, text, (6, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    target_bgr = cv2.imread(
        os.path.join(TARGET_ROOT, *target_key.split("/"), "target.png")
    )
    tgt_vis = cv2.resize(target_bgr, (H, H)) if target_bgr is not None else np.zeros((H, H, 3), np.uint8)

    panel = np.hstack([
        labeled(tgt_vis,         f"TARGET ({target_key})"),
        labeled(bgr,             f"SCENE (epoch {epoch})"),
        labeled(to_bgr(gt),      f"GT ({target_category})"),
        labeled(to_bgr(pred_np), "PRED"),
    ])
    cv2.imwrite(out_path, panel)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int,   default=EPOCHS)
    p.add_argument("--batch",    type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",       type=float, default=LR)
    p.add_argument("--patience", type=int,   default=EARLY_STOP_PATIENCE)
    p.add_argument("--seed",     type=int,   default=SPLIT_SEED)
    p.add_argument("--out",      type=str,   default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(OUT_DIR, f"zeroshot_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"출력 디렉터리: {out_dir}")
    print(f"device: {DEVICE}")

    # ── 파이프라인 로딩 ──────────────────────────────────────────────────
    print("\n[1/4] 파이프라인 로딩 ...")
    pipe = ZeroShotPipeline(device=DEVICE)

    # 학습 대상 파라미터만 optimizer에 등록
    trainable = list(pipe.semantic.proj.parameters()) + list(pipe.head.parameters())
    print(f"  학습 파라미터: {sum(p.numel() for p in trainable):,}")

    # ── scenes.yaml 로드 ─────────────────────────────────────────────────
    print(f"\n  config: {SCENE_CONFIG_PATH}")
    scene_entries = load_scene_config(SCENE_CONFIG_PATH)
    print(f"  씬 설정: {len(scene_entries)}개 로드됨")

    # ── 타겟 캐시 ────────────────────────────────────────────────────────
    print("\n[2/4] 타겟 인코딩 (학습 중 재계산 없음) ...")
    target_cache = pipe.precompute_target_cache(TARGET_ROOT)

    # scenes.yaml → target 캐시 검증
    for entry in scene_entries:
        if entry.target not in target_cache:
            print(f"  [WARN] '{entry.scene}' → target '{entry.target}' 가 캐시에 없음")

    # ── 데이터셋 구성 ────────────────────────────────────────────────────
    print("\n[3/4] 데이터셋 구성 ...")
    train_samples, val_samples = [], []
    used_seed = args.seed

    for entry in scene_entries:
        cache_key = entry.target
        if cache_key not in target_cache:
            print(f"  [SKIP] {entry.scene} (target 캐시 없음: {cache_key})")
            continue
        triples = _discover_samples_seg(entry.scene)
        if not triples:
            print(f"  [SKIP] {entry.scene} (rgb/seg 파일 없음)")
            continue
        # target_category: scenes.yaml target_key → target_cache meta
        target_category = target_cache[cache_key]["meta"]["category"]
        target_usd_name = entry.usd
        tr, va, used_seed = _split_by_scene(triples, TRAIN_RATIO, used_seed)
        # (rgb_path, seg_path, scene_id, target_key, target_category, target_usd_name)
        train_samples += [(r, s, sid, cache_key, target_category, target_usd_name) for r, s, sid in tr]
        val_samples   += [(r, s, sid, cache_key, target_category, target_usd_name) for r, s, sid in va]
        print(f"  {entry.scene:30s} → {cache_key:30s} [{target_category}, usd={target_usd_name}]"
              f"  train={len(tr):5d} val={len(va):5d}")

    if not train_samples:
        sys.exit("학습 샘플이 없습니다. SCENE_TARGET_MAP과 데이터 경로를 확인하세요.")

    print(f"  split_seed={used_seed}")
    print(f"  총 train={len(train_samples):,}  val={len(val_samples):,}")

    train_ds = SceneDataset(train_samples)
    val_ds   = SceneDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  collate_fn=_collate,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, collate_fn=_collate,
                              num_workers=4, pin_memory=True)

    # ── 학습 ─────────────────────────────────────────────────────────────
    print("\n[4/4] 학습 시작 ...")
    optim     = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    stopper   = EarlyStopping(patience=args.patience,
                              min_delta_pct=EARLY_STOP_MIN_DELTA_PCT)

    best_ckpt = os.path.join(out_dir, "zeroshot_best.pt")
    last_ckpt = os.path.join(out_dir, "zeroshot_last.pt")
    log_path  = os.path.join(out_dir, "train_log.txt")
    open(log_path, "w").close()

    history = []
    for epoch in range(1, args.epochs + 1):
        # train
        tr_mse, tr_rmse = run_epoch(
            pipe, train_loader, target_cache, optim=optim,
            desc=f"epoch {epoch}/{args.epochs} [train]"
        )
        # val
        va_mse, va_rmse = run_epoch(
            pipe, val_loader, target_cache, optim=None,
            desc=f"epoch {epoch}/{args.epochs} [val  ]"
        )
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        log_line = (f"epoch {epoch:3d}/{args.epochs}  "
                    f"train_mse={tr_mse:.5f} rmse={tr_rmse:.4f}  "
                    f"val_mse={va_mse:.5f} rmse={va_rmse:.4f}  "
                    f"lr={lr_now:.2e}")
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")
        history.append((epoch, tr_mse, va_mse))

        # early stopping & best checkpoint
        if stopper(va_mse):
            torch.save({
                "epoch":      epoch,
                "val_mse":    va_mse,
                "proj_state": pipe.semantic.proj.state_dict(),
                "head_state": pipe.head.state_dict(),
            }, best_ckpt)
            print(f"    → best checkpoint 저장: {best_ckpt}")
            with open(log_path, "a") as f:
                f.write(f"    → best (val_mse={va_mse:.5f})\n")

        # periodic last checkpoint + panel
        if epoch % SAVE_INTERVAL == 0 or epoch == args.epochs:
            torch.save({
                "epoch":      epoch,
                "val_mse":    va_mse,
                "proj_state": pipe.semantic.proj.state_dict(),
                "head_state": pipe.head.state_dict(),
                "optim_state": optim.state_dict(),
            }, last_ckpt)
            panel_path = os.path.join(out_dir, f"panel_epoch{epoch:03d}.png")
            save_panel(pipe, val_samples, target_cache, panel_path, epoch)
            print(f"    → last checkpoint + panel 저장 (epoch {epoch})")

        if stopper.early_stop:
            msg = f"EarlyStopping 발동 (patience={args.patience}) — epoch {epoch}에서 종료"
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")
            break

    # loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs_, tr_hist, va_hist = zip(*history)
        plt.figure(figsize=(7, 4))
        plt.plot(epochs_, tr_hist, label="train MSE")
        plt.plot(epochs_, va_hist, label="val MSE")
        plt.xlabel("epoch"); plt.ylabel("MSE")
        plt.title("ZeroShot pipeline training")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120)
        print("loss curve 저장 완료")
    except ImportError:
        pass

    print(f"\n완료. best checkpoint: {best_ckpt}")
    print(f"       best val_mse:   {stopper.best_loss:.5f}")


if __name__ == "__main__":
    main()
