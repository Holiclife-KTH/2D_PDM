"""
Zero-shot similarity map pipeline (초안).

구성:
  - DINOv3 vits16          (appearance, 384-dim)  — 로컬 가중치 사용
  - SigLIP so400m-patch14  (semantic,  1152-dim)  — Gemma3 4B의 실제 vision encoder
  - ZeroShotHead           (trainable conv head)

타겟 쿼리 = DINOv3 appearance vec  +  SigLIP semantic proj (additive fusion)
씬 피처   = DINOv3 vits16 patch grid
매칭 헤드  = 학습 가능한 conv head (SimilarityMapModel 구조 동일)

학습 대상: ZeroShotHead.matching_blocks + ZeroShotHead.fuse + ZeroShotHead.aux_head
           + SiglipSemanticEncoder.proj   (Linear 1152→384 × num_layers)
고정 대상: DINOv3 backbone + SigLIP encoder

사용 예:
    pipe = ZeroShotPipeline(device="cuda")
    pred = pipe.predict_single(scene_bgr, target_bgr)   # (H,W) float32 [0,1]
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from transformers import SiglipModel, AutoTokenizer

# --------------------------------------------------------------------------- #
# 경로 상수
# --------------------------------------------------------------------------- #
_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_TH_WS     = os.path.dirname(_SRC_DIR)
MODEL_DIR  = os.path.join(_TH_WS, "model")

# --------------------------------------------------------------------------- #
# 폴더 구조 → 텍스트 라벨 자동 변환
# --------------------------------------------------------------------------- #
# 신규 계층 구조: target/<Category>/<SpecificObject>/target.png
#   Category     → coarse label  (예: "Fruit"        → "fruit")
#   SpecificObject → fine label  (예: "Apple"        → "apple")
#                                 (예: "Tomato_soup_can" → "tomato soup can")
#
# prompt에 두 레벨을 모두 담아 SigLIP의 semantic 공간을 최대한 활용:
#   "a photo of an apple, a type of fruit"
PROMPT_TEMPLATE = "a photo of {specific}, a type of {category}"
PROMPT_TEMPLATE_CATEGORY_ONLY = "a photo of a {category}"  # specific 없을 때 fallback


def _normalize_name(raw: str) -> str:
    """폴더명 → 읽기 좋은 소문자 문자열. 밑줄→공백, 끝 숫자 제거."""
    s = raw.replace('_', ' ').strip()
    s = re.sub(r'\s+\d+$', '', s)   # 끝 숫자 제거 (예: "fruit 1" → "fruit")
    return s.lower()


def path_to_labels(category_dir: str, specific_dir: str) -> Dict[str, str]:
    """
    폴더명 두 단계 → 라벨 딕셔너리.
      category_dir : "Fruit", "Packaged_food", ...
      specific_dir : "Apple", "Tomato_soup_can", "SPAM", ...
    반환:
      {"category": "fruit", "specific": "apple",
       "prompt": "a photo of an apple, a type of fruit"}
    """
    category = _normalize_name(category_dir)
    specific = _normalize_name(specific_dir)
    # "SPAM" 같은 전부 대문자는 소문자화하지 않음
    if specific_dir.isupper():
        specific = specific_dir   # 브랜드명 보존
    prompt = PROMPT_TEMPLATE.format(specific=specific, category=category)
    return {"category": category, "specific": specific, "prompt": prompt}


def discover_target_entries(target_root: str) -> List[Dict]:
    """
    target/<Category>/<SpecificObject>/target.png 패턴을 재귀 탐색.

    반환 예:
      [
        {"category": "fruit", "specific": "apple",
         "prompt": "a photo of an apple, a type of fruit",
         "path": "/…/target/Fruit/Apple/target.png",
         "category_dir": "Fruit", "specific_dir": "Apple"},
        ...
      ]
    항목이 없으면 빈 리스트.
    """
    entries = []
    if not os.path.isdir(target_root):
        return entries
    for cat_dir in sorted(os.listdir(target_root)):
        cat_path = os.path.join(target_root, cat_dir)
        if not os.path.isdir(cat_path):
            continue
        for spec_dir in sorted(os.listdir(cat_path)):
            spec_path = os.path.join(cat_path, spec_dir)
            if not os.path.isdir(spec_path):
                continue
            target_png = os.path.join(spec_path, "target.png")
            if not os.path.isfile(target_png):
                continue
            labels = path_to_labels(cat_dir, spec_dir)
            entries.append({
                **labels,
                "path":         target_png,
                "category_dir": cat_dir,
                "specific_dir": spec_dir,
                "key":          f"{cat_dir}/{spec_dir}",   # 캐시 키로 사용
            })
    return entries

DINOV3_VARIANT   = "vits16"
DINOV3_WEIGHT    = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DINOV3_EMBED_DIM = 384
DINOV3_PATCH_SZ  = 16
DINOV3_LAYERS    = (2, 5, 8, 11)   # vits16 총 12층, 4개 layer 사용

SIGLIP_MODEL_ID  = "google/siglip-so400m-patch14-384"
SIGLIP_DIM       = 1152
SIGLIP_IMG_SZ    = 384

# ImageNet normalization (DINOv3)
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# SigLIP normalization (mean=0.5, std=0.5)
_SIG_MEAN  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_SIG_STD   = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# 이미지 전처리 헬퍼
# --------------------------------------------------------------------------- #

def bgr_to_dino_tensor(bgr: np.ndarray, size: int = 224, device: str = "cuda") -> torch.Tensor:
    """BGR uint8 → (1,3,size,size) ImageNet-normalized float tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return ((t - _DINO_MEAN) / _DINO_STD).to(device)


def bgr_to_siglip_tensor(bgr: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """BGR uint8 → (1,3,384,384) SigLIP-normalized float tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SIGLIP_IMG_SZ, SIGLIP_IMG_SZ), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return ((t - _SIG_MEAN) / _SIG_STD).to(device)


def gray_fg_mask(bgr: np.ndarray, gray_thresh: float = 0.05,
                 bright_thresh: float = 0.4) -> np.ndarray:
    """BGR uint8 → (H,W) bool FG 마스크 (gray 배경 제거)."""
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    max_diff = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    brightness = (r + g + b) / 3.0
    is_gray = (max_diff < gray_thresh) & (brightness > bright_thresh)
    fg = ~is_gray
    return fg if fg.any() else np.ones_like(fg)


def crop_fg(bgr: np.ndarray, mask: np.ndarray, pad: float = 0.25):
    """FG bounding box로 tight crop (+ pad 비율 여백)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return bgr, mask
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    py, px = int(h * pad), int(w * pad)
    H, W = bgr.shape[:2]
    y0, y1 = max(0, y0 - py), min(H - 1, y1 + py)
    x0, x1 = max(0, x0 - px), min(W - 1, x1 + px)
    return bgr[y0:y1+1, x0:x1+1], mask[y0:y1+1, x0:x1+1]


# --------------------------------------------------------------------------- #
# DINOv3 vits16 backbone (frozen)
# --------------------------------------------------------------------------- #

class DinoV3Vits16(nn.Module):
    """torch.hub.load 방식으로 로컬 가중치를 쓰는 frozen vits16."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        weight_path = os.path.join(MODEL_DIR, DINOV3_WEIGHT)
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(weight_path)

        self.model = torch.hub.load(
            "facebookresearch/dinov3", f"dinov3_{DINOV3_VARIANT}",
            weights=weight_path, trust_repo=True
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.layers = list(DINOV3_LAYERS)
        self.embed_dim = DINOV3_EMBED_DIM
        self.patch_size = DINOV3_PATCH_SZ
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns [(patch (B,C,Hp,Wp), cls (B,C)), ...] per layer."""
        x = x.to(self.device)
        outs = self.model.get_intermediate_layers(
            x, n=self.layers, reshape=True, return_class_token=True, norm=True
        )
        return list(outs)


# --------------------------------------------------------------------------- #
# SigLIP semantic encoder (frozen) + trainable projection
# --------------------------------------------------------------------------- #

class SiglipSemanticEncoder(nn.Module):
    """
    SigLIP so400m (Gemma 3의 vision encoder) 풀 모델.
    이미지 + 텍스트(선택) → 1152-dim semantic embedding.
    proj: Linear(1152 → 384) × num_layers — DINOv3 space 정렬 (학습 대상).

    텍스트 없음: image embedding만 사용
    텍스트 있음: (image_embed + text_embed) / 2  — 같은 SigLIP 공간이라 단순 평균 OK
                 → 시각적 모호함을 카테고리명이 보완하여 zero-shot robustness 향상
    """

    def __init__(self, num_layers: int = len(DINOV3_LAYERS), device: str = "cuda"):
        super().__init__()
        self.device = device

        print(f"SigLIP 로딩: {SIGLIP_MODEL_ID}")
        self.model = SiglipModel.from_pretrained(SIGLIP_MODEL_ID)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID)
        self._text_cache: Dict[str, torch.Tensor] = {}   # label → (1,1152) cached

        self.proj = nn.ModuleList([
            nn.Linear(SIGLIP_DIM, DINOV3_EMBED_DIM)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def encode_image(self, siglip_tensor: torch.Tensor) -> torch.Tensor:
        """(B,3,384,384) → (B, 1152) L2-normalized image embedding."""
        with torch.no_grad():
            out = self.model.vision_model(pixel_values=siglip_tensor.to(self.device))
        return F.normalize(out.pooler_output, dim=-1)   # (B, 1152)

    def encode_text(self, label: str) -> torch.Tensor:
        """텍스트(단어 또는 완성된 prompt) → (1, 1152) L2-normalized text embedding.
        결과는 _text_cache에 저장 — 같은 label은 두 번 계산하지 않음."""
        prompt = label   # 호출 측에서 이미 완성된 prompt를 넘겨줌
        if prompt not in self._text_cache:
            tokens = self.tokenizer(
                [prompt], padding="max_length", max_length=64,
                truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model.text_model(**tokens)
            self._text_cache[prompt] = F.normalize(out.pooler_output, dim=-1)  # (1,1152)
        return self._text_cache[prompt]

    def fuse(self, img_embed: torch.Tensor,
             label: Optional[str] = None) -> torch.Tensor:
        """
        img_embed: (B, 1152)
        label    : 카테고리 텍스트 (없으면 image-only)
        returns  : (B, 1152) fused semantic embedding
        """
        if label is None:
            return img_embed
        text_embed = self.encode_text(label)          # (1, 1152)
        text_embed = text_embed.expand_as(img_embed)  # (B, 1152)
        # 두 벡터가 같은 SigLIP 공간에 정렬되어 있으므로 평균이 수학적으로 유효
        return F.normalize((img_embed + text_embed) / 2.0, dim=-1)

    def project(self, semantic: torch.Tensor) -> List[torch.Tensor]:
        """(B, 1152) → [(B, 384), ...] × num_layers."""
        return [self.proj[i](semantic) for i in range(self.num_layers)]


# --------------------------------------------------------------------------- #
# Matching head (SimilarityMapModel 구조 동일)
# --------------------------------------------------------------------------- #

class MatchingBlock(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ZeroShotHead(nn.Module):
    """
    입력: scene patch grid + target query vector (appearance + semantic 합쳐진 것)
    출력: [0,1] 유사도 맵 (patch 해상도 또는 full 해상도)

    interaction channel = scene_feat (384) + target_bcast (384) + cos_sim (1) = 769
    """

    def __init__(self, embed_dim: int = DINOV3_EMBED_DIM,
                 num_layers: int = len(DINOV3_LAYERS), hidden: int = 64):
        super().__init__()
        in_ch = embed_dim * 2 + 1   # scene + target_broadcast + cosine
        self.blocks = nn.ModuleList([MatchingBlock(in_ch, hidden) for _ in range(num_layers)])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * num_layers, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, scene_feats: List[Tuple[torch.Tensor, torch.Tensor]],
                target_vecs: List[torch.Tensor],
                out_size: Optional[Tuple[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        scene_feats : [(patch (B,C,Hp,Wp), cls), ...] — DINOv3 layer별 출력
        target_vecs : [(B,C), ...] — 이미 fusion된 쿼리 (appearance + semantic)
        out_size    : (H, W) full-res 업샘플링 목적지 (None이면 patch 해상도만)
        """
        layer_outs = []
        for (patch, _), tvec in zip(scene_feats, target_vecs):
            B, C, Hp, Wp = patch.shape
            patch_n = F.normalize(patch, dim=1)
            tvec_n  = F.normalize(tvec, dim=1)
            cos = (patch_n * tvec_n.view(B, C, 1, 1)).sum(1, keepdim=True)
            cos = (cos + 1.0) / 2.0   # [-1,1] → [0,1]
            t_bcast = tvec.view(B, C, 1, 1).expand(-1, -1, Hp, Wp)
            layer_outs.append(torch.cat([patch, t_bcast, cos], dim=1))

        matched = [block(x) for block, x in zip(self.blocks, layer_outs)]
        fused   = self.fuse(torch.cat(matched, dim=1))
        logits  = self.head(fused)
        prob_patch = torch.sigmoid(logits)   # (B,1,Hp,Wp)

        prob_full = None
        if out_size is not None:
            prob_full = F.interpolate(prob_patch, size=out_size, mode="bilinear", align_corners=False)

        return {"prob_patch": prob_patch, "prob_full": prob_full}


# --------------------------------------------------------------------------- #
# 타겟 풀링 헬퍼 (masked pool)
# --------------------------------------------------------------------------- #

def masked_pool(patch_grid: torch.Tensor, pixel_mask: np.ndarray) -> torch.Tensor:
    """
    patch_grid : (C, Hp, Wp) — DINOv3 single-sample patch feature
    pixel_mask : (H, W) bool  — FG 픽셀 마스크 (crop과 같은 해상도)
    returns    : (C,) L2-normalized appearance vector
    """
    C, Hp, Wp = patch_grid.shape
    m = torch.from_numpy(pixel_mask.astype(np.float32))[None, None].to(patch_grid.device)
    weight = F.avg_pool2d(m, kernel_size=DINOV3_PATCH_SZ, stride=DINOV3_PATCH_SZ)[0, 0]
    if weight.sum() < 1e-6:
        weight = torch.ones_like(weight)
    weight = weight / weight.sum()
    pooled = (patch_grid * weight[None]).sum(dim=(1, 2))
    return F.normalize(pooled, dim=0)


# --------------------------------------------------------------------------- #
# 전체 파이프라인
# --------------------------------------------------------------------------- #

class ZeroShotPipeline(nn.Module):
    """
    구성:
        dino    : DinoV3Vits16   (frozen)
        semantic: SiglipSemanticEncoder (SigLIP frozen, proj 학습 대상)
        head    : ZeroShotHead   (학습 대상)

    학습 파라미터: semantic.proj + head.blocks + head.fuse + head.head
    """

    def __init__(self, device: str = "cuda", hidden: int = 64):
        super().__init__()
        self.device = device

        print("DINOv3 vits16 로딩 ...")
        self.dino     = DinoV3Vits16(device=device)

        print("SigLIP so400m 로딩 (Gemma3 vision encoder) ...")
        self.semantic = SiglipSemanticEncoder(num_layers=len(DINOV3_LAYERS), device=device)
        self.semantic.to(device)

        self.head = ZeroShotHead(
            embed_dim=DINOV3_EMBED_DIM,
            num_layers=len(DINOV3_LAYERS),
            hidden=hidden,
        ).to(device)

    # ------------------------------------------------------------------ #
    # 타겟 인코딩 (캐싱 목적으로 분리)
    # ------------------------------------------------------------------ #
    def encode_target(self, target_bgr: np.ndarray,
                      label: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        타겟 이미지 한 장 → 쿼리 벡터 딕셔너리 (캐싱 가능).

        label: 카테고리 텍스트 (예: "apple", "packaged food box").
               None이면 image embedding만 사용.
               제공하면 (image_embed + text_embed) / 2 로 semantic 강화.

        반환:
            "query_vecs":   List[(1, 384)] × num_layers
            "appearances":  List[(384,)] × num_layers
            "semantic_raw": (1152,) fused SigLIP embedding (projection 전)
            "label":        입력한 label 문자열 (또는 None)
        """
        # ① FG 마스크 & crop
        fg_mask          = gray_fg_mask(target_bgr)
        crop_bgr, crop_mask = crop_fg(target_bgr, fg_mask, pad=0.25)

        # ② DINOv3 appearance: 224×224 crop
        dino_tensor = bgr_to_dino_tensor(crop_bgr, size=224, device=self.device)
        mask_224    = cv2.resize(
            crop_mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        dino_feats  = self.dino(dino_tensor)
        appearances = [masked_pool(patch[0], mask_224) for patch, _ in dino_feats]

        # ③ SigLIP semantic: image embed → (선택) text와 평균 fusion
        sig_tensor = bgr_to_siglip_tensor(crop_bgr, device=self.device)
        img_embed  = self.semantic.encode_image(sig_tensor)        # (1, 1152)
        sem_raw    = self.semantic.fuse(img_embed, label=label)    # (1, 1152)
        sem_projs  = self.semantic.project(sem_raw)                # [(1, 384)] × n

        # ④ additive fusion: appearance + semantic
        query_vecs = [
            (app.unsqueeze(0) + spr).contiguous()
            for app, spr in zip(appearances, sem_projs)
        ]

        return {
            "query_vecs":    query_vecs,
            "appearances":   appearances,
            "semantic_raw":  sem_raw[0],
            "label":         label,
        }

    def expand_query(self, cached: Dict, B: int) -> List[torch.Tensor]:
        """캐싱된 query_vecs를 배치 B로 확장."""
        return [
            v.expand(B, -1).contiguous()
            for v in cached["query_vecs"]
        ]

    def precompute_target_cache(self, target_root: str) -> Dict[str, Dict]:
        """
        target/<Category>/<SpecificObject>/target.png 구조를 자동 탐색해서
        모든 타겟을 한 번에 인코딩.

        - 텍스트 라벨: path_to_labels(Category, SpecificObject) 자동 생성
          예) Fruit/Apple → prompt="a photo of an apple, a type of fruit"
        - 텍스트 embedding: _text_cache에 저장 — 동일 카테고리는 1회만 계산
        - backbone + SigLIP 모두 frozen → 결과 결정적, 학습 중 재계산 불필요

        반환:
            {
              "Fruit/Apple":            {"query_vecs": [...], "prompt": "...", ...},
              "Fruit/Avocado":          {...},
              "Packaged_food/SPAM":     {...},
              ...
            }
        캐시 키 = "Category/SpecificObject" (entry["key"])
        """
        entries = discover_target_entries(target_root)
        if not entries:
            raise FileNotFoundError(
                f"target.png 파일을 찾을 수 없음: {target_root}\n"
                f"예상 구조: target/<Category>/<SpecificObject>/target.png"
            )

        cache: Dict[str, Dict] = {}
        for e in entries:
            bgr = cv2.imread(e["path"])
            if bgr is None:
                print(f"  [WARN] 이미지 읽기 실패: {e['path']}")
                continue
            encoded = self.encode_target(bgr, label=e["prompt"])
            cache[e["key"]] = {**encoded, "meta": e}
            print(f"  {e['key']:35s} → prompt={e['prompt']!r}")

        print(f"  총 {len(cache)}개 타겟 캐시 완료")
        return cache

    def collate_query(self, target_cache: Dict, target_names: List[str],
                      cam: str = "center") -> List[torch.Tensor]:
        """
        배치 내 각 샘플의 target_name에 맞는 query_vecs를 모아서 (B,384) 텐서로 반환.
        학습 루프의 DataLoader collate 이후 호출:
            target_query = pipe.collate_query(cache, batch_target_names)
            out = pipe(scene_batch, target_query, out_size=(H,W))
        """
        batch_vecs = [target_cache[name][cam]["query_vecs"] for name in target_names]
        # batch_vecs: List[List[(1,384)]]  →  List[(B,384)]
        return [
            torch.cat([bv[li] for bv in batch_vecs], dim=0)   # (B,384)
            for li in range(len(DINOV3_LAYERS))
        ]

    # ------------------------------------------------------------------ #
    # Forward (학습 루프용)
    # ------------------------------------------------------------------ #
    def forward(self, scene_bgr_batch: torch.Tensor,
                target_query: List[torch.Tensor],
                out_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        scene_bgr_batch : (B,3,H,W) ImageNet-normalized tensor
        target_query    : List[(B,384)] × num_layers — expand_query() 결과
        """
        scene_feats = self.dino(scene_bgr_batch)
        return self.head(scene_feats, target_query, out_size=out_size)

    # ------------------------------------------------------------------ #
    # 단발성 추론 (테스트/시각화용)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_single(self, scene_bgr: np.ndarray,
                       target_bgr: np.ndarray,
                       label: Optional[str] = None) -> np.ndarray:
        """
        scene_bgr  : (H,W,3) BGR uint8
        target_bgr : (H',W',3) BGR uint8 (gray 배경 가능)
        label      : 카테고리 텍스트 (예: "apple"). None이면 image-only.
        returns    : (H,W) float32 similarity map [0,1]
        """
        H, W = scene_bgr.shape[:2]

        target_cache = self.encode_target(target_bgr, label=label)
        query_vecs   = target_cache["query_vecs"]

        Hs = (H // DINOV3_PATCH_SZ) * DINOV3_PATCH_SZ
        Ws = (W // DINOV3_PATCH_SZ) * DINOV3_PATCH_SZ
        scene_t = torch.from_numpy(
            cv2.resize(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB), (Ws, Hs)).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
        scene_t = ((scene_t - _DINO_MEAN) / _DINO_STD).to(self.device)

        scene_feats = self.dino(scene_t)
        out = self.head(scene_feats, query_vecs, out_size=(H, W))
        return out["prob_full"][0, 0].cpu().numpy()


# --------------------------------------------------------------------------- #
# 학습 파라미터 요약 출력
# --------------------------------------------------------------------------- #

def print_param_summary(pipe: ZeroShotPipeline):
    def count(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("\n[학습 파라미터 요약]")
    print(f"  SigLIP proj (DINOv3 정렬): {count(pipe.semantic.proj):>10,} params")
    print(f"  ZeroShotHead blocks+fuse:  {count(pipe.head):>10,} params")
    total = count(pipe.semantic.proj) + count(pipe.head)
    print(f"  ─────────────────────────────────────")
    print(f"  학습 대상 합계:             {total:>10,} params")
    print(f"  DINOv3 vits16 (frozen):   {sum(p.numel() for p in pipe.dino.parameters()):>10,} params")
    enc_params = sum(p.numel() for p in pipe.semantic.model.parameters())
    print(f"  SigLIP encoder (frozen):  {enc_params:>10,} params")


# --------------------------------------------------------------------------- #
# 간단한 동작 테스트
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("ZeroShotPipeline 동작 테스트")
    print("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {DEVICE}")

    pipe = ZeroShotPipeline(device=DEVICE)
    print_param_summary(pipe)

    # 더미 이미지로 shape 검증
    dummy_scene  = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_target = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    # target 중앙에 비-gray 물체 시뮬레이션
    dummy_target[60:160, 60:160] = [200, 80, 30]

    # ── path_to_labels 자동 변환 확인 ──
    print("\n[path_to_labels 자동 변환 테스트]")
    test_cases = [
        ("Fruit",         "Apple"),
        ("Fruit",         "Avocado"),
        ("Packaged_food", "SPAM"),
        ("Packaged_food", "Tomato_soup_can"),
    ]
    for cat, spec in test_cases:
        labels = path_to_labels(cat, spec)
        print(f"  {cat}/{spec:20s} → prompt={labels['prompt']!r}")

    # ── discover_target_entries 실제 폴더 탐색 ──
    TARGET_ROOT = os.path.join(_TH_WS, "data", "target")
    print(f"\n[discover_target_entries: {TARGET_ROOT}]")
    entries = discover_target_entries(TARGET_ROOT)
    if entries:
        for e in entries:
            print(f"  {e['key']:35s} → {e['prompt']!r}")
    else:
        print("  (항목 없음 — 더미 이미지로 대체 테스트)")

    # ── precompute_target_cache (실제 폴더 기반) ──
    print("\n[precompute_target_cache 테스트]")
    if entries:
        target_cache = pipe.precompute_target_cache(TARGET_ROOT)
        # collate_query 테스트: 첫 두 항목으로 배치 구성
        keys = list(target_cache.keys())
        if len(keys) >= 2:
            batch_keys = [keys[0], keys[1], keys[0]]
            batch_vecs = [target_cache[k]["query_vecs"] for k in batch_keys]
            collated = [
                torch.cat([bv[li] for bv in batch_vecs], dim=0)
                for li in range(len(DINOV3_LAYERS))
            ]
            print(f"\n[collate_query (배치={len(batch_keys)})]")
            print(f"  keys: {batch_keys}")
            print(f"  collated: {len(collated)} layers, each {tuple(collated[0].shape)}")
    else:
        print("  target 폴더에 항목이 없어 건너뜀")

    print("\n[predict_single — 실제 타겟 이미지]")
    if entries:
        first_bgr = cv2.imread(entries[0]["path"])
        pred = pipe.predict_single(dummy_scene, first_bgr, label=entries[0]["prompt"])
    else:
        pred = pipe.predict_single(dummy_scene, dummy_target, label="apple, a type of fruit")
    print(f"  pred shape: {pred.shape}, min={pred.min():.3f}, max={pred.max():.3f}")
    print("\n테스트 완료.")
