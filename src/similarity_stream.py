import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoConfig
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ── Utility ───────────────────────────────────────────────────────────────────

def pad_to_patch_multiple_tensor(
    x: torch.Tensor, patch_size: int
) -> Tuple[torch.Tensor, Dict]:
    """
    x: [B, C, H, W]
    returns: padded tensor + meta for cropping back
    """
    _, _, H, W = x.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, {"orig_h": H, "orig_w": W, "pad_h": pad_h, "pad_w": pad_w}


def pool_target_feature(
    target_feature: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    target_feature: [B, D, H_p, W_p]
    target_mask:    optional [B, 1, H_p, W_p]
    returns q:      [B, D]
    """
    if target_mask is not None:
        mask = target_mask.clamp(min=1e-6)
        return (target_feature * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3))
    return target_feature.mean(dim=(2, 3))


def make_fg_mask(
    image: torch.Tensor,
    patch_size: int,
    gray_thresh: float = 0.05,
    bright_thresh: float = 0.4,
) -> torch.Tensor:
    """
    단색(gray) 배경을 자동 감지해 foreground mask를 생성.

    image:        [B, 3, H, W]  float [0,1]
    returns mask: [B, 1, H_p, W_p]  float  (FG=1, BG=0)

    gray 판별 기준:
      - R≈G≈B (채널 간 max차이 < gray_thresh)
      - 전체 밝기 > bright_thresh  (어두운 물체 제외)
    패치 단위로 집계: FG 픽셀 비율 > 0.3이면 FG 패치
    """
    B, C, H, W = image.shape
    H_p = H // patch_size
    W_p = W // patch_size

    r, g, b = image[:, 0], image[:, 1], image[:, 2]   # [B, H, W]
    max_diff = torch.stack([
        (r - g).abs(), (g - b).abs(), (r - b).abs()
    ], dim=1).max(dim=1).values                        # [B, H, W]
    brightness = (r + g + b) / 3.0

    # gray pixel: 채널 차이 작고 충분히 밝음
    is_gray = (max_diff < gray_thresh) & (brightness > bright_thresh)
    is_fg   = ~is_gray                                 # [B, H, W]  bool

    # patch 단위 집계: FG 픽셀 비율
    is_fg_f = is_fg.float()                            # [B, H, W]
    # unfold: dim1=H, dim2=W → [B, H_p, W_p, ps, ps]
    fg_ratio = is_fg_f.unfold(1, patch_size, patch_size) \
                      .unfold(2, patch_size, patch_size) \
                      .mean(dim=(-2, -1))              # [B, H_p, W_p]

    mask = (fg_ratio > 0.3).float().unsqueeze(1)       # [B, 1, H_p, W_p]

    # 모든 패치가 배경이면 전체를 1로 (fallback)
    if mask.sum() == 0:
        mask = torch.ones_like(mask)

    return mask


# ── DINOv3 Feature Extractor ──────────────────────────────────────────────────

class DINOFeatureExtractor(nn.Module):
    """
    Frozen DINOv3 backbone에서 지정 layer의 patch feature map을 추출.
    입력: tensor [B, 3, H, W], float [0, 1]
    출력: {layer_idx: [B, D, H_p, W_p]}
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        model_path: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        freeze: bool = True,
    ):
        super().__init__()
        if model_path is not None:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)
            ckpt = torch.load(model_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[WARN] missing keys  : {len(missing)}")
            if unexpected:
                print(f"[WARN] unexpected keys: {len(unexpected)}")
            print(f"[INFO] loaded weights from {model_path}")
        else:
            self.backbone = AutoModel.from_pretrained(model_name)

        self.patch_size = self.backbone.config.patch_size
        self.num_reg = getattr(self.backbone.config, "num_register_tokens", 0)
        num_layers = self.backbone.config.num_hidden_layers

        if layer_indices is None:
            raw = [num_layers - 7, num_layers - 4, num_layers - 1]
        else:
            raw = layer_indices
        self.layer_indices: List[int] = sorted(set(
            max(0, min(i, num_layers - 1)) for i in raw
        ))

        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(self._STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        returns:
            patch_feats: {layer: [B, D, H_p, W_p]}
            cls_feats:   {layer: [B, D]}   ← CLS token (글로벌 의미 벡터)
        """
        B, _, H, W = images.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        x = (images - self.mean) / self.std

        with torch.no_grad():
            outputs = self.backbone(
                pixel_values=x,
                output_hidden_states=True,
                interpolate_pos_encoding=True,
            )

        patch_feats: Dict[int, torch.Tensor] = {}
        cls_feats:   Dict[int, torch.Tensor] = {}
        for b in self.layer_indices:
            hs = outputs.hidden_states[b + 1]             # [B, 1+num_reg+hw, D]
            cls_feats[b]  = hs[:, 0, :]                   # [B, D]  — CLS token
            tokens = hs[:, 1 + self.num_reg:, :]          # [B, hw, D]
            D = tokens.shape[-1]
            feat = tokens.reshape(B, H_p, W_p, D).permute(0, 3, 1, 2)  # [B, D, H_p, W_p]
            patch_feats[b] = feat

        return patch_feats, cls_feats


# ── Layer-wise Matching Block ─────────────────────────────────────────────────

class LayerWiseMatchingBlock(nn.Module):
    """
    Input I_i:  [B, in_ch, H_p, W_p]   (in_ch = align_dim*4 + 1 = 513)
    Output M_i: [B, out_ch, H_p, W_p]  (out_ch = match_dim = 64)
    """

    def __init__(self, in_ch: int = 513, hidden_ch: int = 128, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,     hidden_ch, 1),
            nn.GroupNorm(8, hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, out_ch, 1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Similarity Matching Module (M_cat) ────────────────────────────────────────

class SimilarityMatchingModule(nn.Module):
    """
    Section 3–9: DINOv3 feature 추출 → Scene-Target interaction tensor →
    layer-wise matching blocks → M_cat

    Output M_cat: [B, match_dim * num_layers, H_p, W_p]
                  e.g. [B, 192, 30, 40]  (match_dim=64, 3 layers, 480×640 input)
    """

    def __init__(
        self,
        dino_model_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        dino_model_path: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        align_dim: int = 128,
        match_dim: int = 64,
        freeze_dino: bool = True,
    ):
        super().__init__()

        self.extractor = DINOFeatureExtractor(
            model_name=dino_model_name,
            model_path=dino_model_path,
            layer_indices=layer_indices,
            freeze=freeze_dino,
        )
        self.layer_indices = self.extractor.layer_indices
        n = len(self.layer_indices)
        D = self.extractor.backbone.config.hidden_size  # 4096

        self.scene_projs = nn.ModuleList([
            nn.Conv2d(D, align_dim, kernel_size=1) for _ in range(n)
        ])
        self.target_projs = nn.ModuleList([
            nn.Linear(D, align_dim) for _ in range(n)
        ])
        # CLS token을 통한 글로벌 의미 쿼리 (zero-shot 일반화용)
        # query = target_projs(patch_pool) + cls_projs(cls_token)
        self.cls_projs = nn.ModuleList([
            nn.Linear(D, align_dim) for _ in range(n)
        ])

        in_ch = align_dim * 4 + 1  # [Z, Q, |Z-Q|, Z*Q, cosine] → 513
        self.matching_blocks = nn.ModuleList([
            LayerWiseMatchingBlock(in_ch=in_ch, hidden_ch=align_dim, out_ch=match_dim)
            for _ in range(n)
        ])

    def forward(
        self,
        scene_rgb: torch.Tensor,
        target_rgb: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        target_feats: Optional[Dict[int, torch.Tensor]] = None,
        target_cls: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        target_feats: {b: [B, D, H_p, W_p]}  사전 계산된 patch features
        target_cls:   {b: [B, D]}             사전 계산된 CLS token (의미 쿼리)
        둘 다 None이면 target_rgb로부터 직접 계산.
        """
        scene_feats, _ = self.extractor(scene_rgb)
        if target_feats is None:
            assert target_rgb is not None, "target_rgb 또는 target_feats 중 하나는 필요"
            target_feats, target_cls = self.extractor(target_rgb)

        M_list = []
        debug: Dict = {}

        for k, b in enumerate(self.layer_indices):
            S_i = scene_feats[b]   # [B, D, H_p, W_p]
            T_i = target_feats[b]  # [B, D, H_p, W_p]

            # patch 기반 외관 쿼리
            q_patch = pool_target_feature(T_i, target_mask)           # [B, D]
            Z_i     = self.scene_projs[k](S_i)                        # [B, align_dim, H_p, W_p]
            p_i     = self.target_projs[k](q_patch)                   # [B, align_dim]

            # CLS 기반 의미 쿼리 — 가산 융합 (모델이 둘의 비중을 학습)
            if target_cls is not None:
                p_i = p_i + self.cls_projs[k](target_cls[b])          # [B, align_dim]

            Q_i = p_i[:, :, None, None].expand_as(Z_i).contiguous()

            cosine = F.cosine_similarity(Z_i, Q_i, dim=1, eps=1e-6).unsqueeze(1)  # [B, 1, H_p, W_p]
            I_i = torch.cat([Z_i, Q_i, (Z_i - Q_i).abs(), Z_i * Q_i, cosine], dim=1)

            M_i = self.matching_blocks[k](I_i)                        # [B, match_dim, H_p, W_p]
            M_list.append(M_i)

            debug[f"layer_{b}"] = {
                "S_i": tuple(S_i.shape),
                "q_i": tuple(q_patch.shape),
                "Z_i": tuple(Z_i.shape),
                "I_i": tuple(I_i.shape),
                "M_i": tuple(M_i.shape),
                "cosine_map": cosine.squeeze(1),  # [B, H_p, W_p]
            }

        M_cat = torch.cat(M_list, dim=1)  # [B, match_dim * n, H_p, W_p]
        return M_cat, debug


# ── Similarity Feature Head ───────────────────────────────────────────────────

class SimilarityFeatureHead(nn.Module):
    """
    Section 10: M_cat → F_s

    M_cat에서 multi-layer matching 결과를 통합해 단일 feature map으로 정제.
    Dilated conv를 사용해 receptive field를 넓히면서 해상도를 유지.

    Input:  [B, in_ch,   H_p, W_p]   (in_ch = match_dim * num_layers, e.g. 192)
    Output: [B, feat_dim, H_p, W_p]  (feat_dim = 128)
    """

    def __init__(self, in_ch: int, feat_dim: int = 128):
        super().__init__()
        mid = feat_dim * 2  # 256

        # 1×1 channel reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1),
            nn.GroupNorm(8, mid),
            nn.GELU(),
        )

        # 병렬 dilated convolution (다양한 scale의 context 포착)
        self.branch1 = nn.Conv2d(mid, feat_dim // 2, 3, padding=1,  dilation=1)
        self.branch2 = nn.Conv2d(mid, feat_dim // 2, 3, padding=2,  dilation=2)
        self.branch3 = nn.Conv2d(mid, feat_dim // 2, 3, padding=4,  dilation=4)
        self.branch4 = nn.Conv2d(mid, feat_dim // 2, 1)  # global context (1×1)

        fused_ch = feat_dim // 2 * 4  # 256
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, feat_dim, 1),
            nn.GroupNorm(8, feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.GroupNorm(8, feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        b1 = F.gelu(self.branch1(x))
        b2 = F.gelu(self.branch2(x))
        b3 = F.gelu(self.branch3(x))
        b4 = F.gelu(self.branch4(x))
        return self.fuse(torch.cat([b1, b2, b3, b4], dim=1))  # [B, feat_dim, H_p, W_p]


# ── Similarity Distribution Head ─────────────────────────────────────────────

class SimilarityDistributionHead(nn.Module):
    """
    Section 11: F_s → S_pred

    패치 해상도 [H_p, W_p] → 원본 이미지 해상도 [H, W]로 upsample하면서
    per-pixel similarity score(또는 class logit)를 출력.

    num_classes=1  → sigmoid 적용 → 이진 similarity map (target 물체 / 배경)
    num_classes>1  → softmax 적용 → multi-class category map

    Input:  [B, in_ch,      H_p, W_p]
    Output: [B, num_classes, H,   W ]
    """

    def __init__(self, in_ch: int, num_classes: int = 1):
        super().__init__()
        mid = max(in_ch // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.GroupNorm(min(8, mid), mid),
            nn.GELU(),
            nn.Conv2d(mid, mid // 2, 3, padding=1),
            nn.GroupNorm(min(8, mid // 2), mid // 2),
            nn.GELU(),
            nn.Conv2d(mid // 2, num_classes, 1),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        logits = self.net(x)  # [B, num_classes, H_p, W_p]
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        # [B, num_classes, H, W]


# ── Full Similarity Stream ────────────────────────────────────────────────────

class SimilarityStream(nn.Module):
    """
    Target 이미지와 Scene 이미지를 받아 Scene 위의 similarity distribution map을
    출력하는 전체 파이프라인.

    Pipeline:
      scene_rgb + target_rgb
        ↓ DINOFeatureExtractor (frozen)
        ↓ SimilarityMatchingModule  → M_cat  [B, match_dim*n, H_p, W_p]
        ↓ SimilarityFeatureHead     → F_s    [B, feat_dim,    H_p, W_p]
        ↓ SimilarityDistributionHead→ S_pred [B, num_classes, H,   W  ]

    S_pred interpretation:
      num_classes=1 : sigmoid(S_pred) → target 물체가 있을 확률 [0,1] per pixel
      num_classes>1 : softmax(S_pred) → 각 카테고리에 속할 확률 per pixel
    """

    def __init__(
        self,
        dino_model_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        dino_model_path: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        align_dim: int = 128,
        match_dim: int = 64,
        feat_dim: int = 128,
        num_classes: int = 1,
        freeze_dino: bool = True,
    ):
        super().__init__()

        self.matching = SimilarityMatchingModule(
            dino_model_name=dino_model_name,
            dino_model_path=dino_model_path,
            layer_indices=layer_indices,
            align_dim=align_dim,
            match_dim=match_dim,
            freeze_dino=freeze_dino,
        )
        n = len(self.matching.layer_indices)
        m_cat_ch = match_dim * n  # e.g. 192

        self.feature_head = SimilarityFeatureHead(in_ch=m_cat_ch, feat_dim=feat_dim)
        self.dist_head    = SimilarityDistributionHead(in_ch=feat_dim, num_classes=num_classes)
        self.num_classes  = num_classes

    def forward(
        self,
        scene_rgb: torch.Tensor,
        target_rgb: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        target_feats: Optional[Dict[int, torch.Tensor]] = None,
        target_cls: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        scene_rgb:    [B, 3, H, W]
        target_rgb:   [B, 3, Ht, Wt]  — target_feats 없을 때만 필요
        target_feats: {b: [B, D, Hp, Wp]} — 사전 계산된 patch features
        target_cls:   {b: [B, D]}          — 사전 계산된 CLS token
        returns:
            S_pred: [B, num_classes, H, W]
            debug:  intermediate tensor shapes
        """
        H, W = scene_rgb.shape[2:]

        M_cat, debug = self.matching(
            scene_rgb, target_rgb, target_mask,
            target_feats=target_feats, target_cls=target_cls,
        )
        F_s    = self.feature_head(M_cat)
        S_pred = self.dist_head(F_s, (H, W))

        debug["M_cat"]  = tuple(M_cat.shape)
        debug["F_s"]    = tuple(F_s.shape)
        debug["S_pred"] = tuple(S_pred.shape)

        return S_pred, debug


# ── Image Loading ─────────────────────────────────────────────────────────────

def load_image_tensor(
    path: str, patch_size: int, device: str = "cpu"
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    이미지 파일 → [1, 3, H, W] float tensor [0,1]
    H, W는 patch_size의 배수로 조정.
    """
    img = Image.open(path).convert("RGB")
    W_orig, H_orig = img.size
    new_W = (W_orig // patch_size) * patch_size
    new_H = (H_orig // patch_size) * patch_size
    img = img.resize((new_W, new_H), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device), (new_H, new_W)


def pca_rgb(feat_map: torch.Tensor) -> np.ndarray:
    """feat_map [C, H, W] → PCA 3채널 → [H, W, 3] float [0,1]"""
    C, H, W = feat_map.shape
    x = feat_map.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(x)
    for c in range(3):
        rgb[:, c] -= rgb[:, c].min()
        rgb[:, c] /= rgb[:, c].max() + 1e-8
    return rgb.reshape(H, W, 3)


# ── Main ──────────────────────────────────────────────────────────────────────

SCENE_PATH  = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/output/rgb/000002_rgb.png"
TARGET_PATH = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/output/target/rgb/000003_rgb.png"
LOCAL_PTH   = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/model/dinov3_vit7b16_hf_converted.pth"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    # ── 모델 초기화 ────────────────────────────────────────────────────────────
    stream = SimilarityStream(
        dino_model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        dino_model_path=LOCAL_PTH,
        num_classes=1,       # 단일 similarity score map
        freeze_dino=True,
    ).to(device).eval()

    patch_size = stream.matching.extractor.patch_size
    layer_indices = stream.matching.layer_indices
    print(f"[INFO] patch_size={patch_size}  layers={layer_indices}")

    # ── 이미지 로드 ────────────────────────────────────────────────────────────
    scene_t,  (H_s, W_s) = load_image_tensor(SCENE_PATH,  patch_size, device)
    target_t, (H_t, W_t) = load_image_tensor(TARGET_PATH, patch_size, device)
    print(f"[INFO] scene:  {tuple(scene_t.shape)}   target: {tuple(target_t.shape)}")

    # ── Forward ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        S_pred, debug = stream(scene_t, target_t)

    # ── Shape 출력 ─────────────────────────────────────────────────────────────
    print("\n── Intermediate shapes ──")
    for key in ["M_cat", "F_s", "S_pred"]:
        print(f"  {key:8s}: {debug[key]}")
    for b in layer_indices:
        info = debug[f"layer_{b}"]
        print(f"  Block {b}: S_i={info['S_i']}  I_i={info['I_i']}  M_i={info['M_i']}")

    # ── 확률 맵 생성 (sigmoid → [0,1]) ────────────────────────────────────────
    prob = torch.sigmoid(S_pred[0, 0]).cpu().numpy()  # [H_s, W_s]

    # DINOv3 cosine similarity map (last layer, pre-head baseline)
    last_b = layer_indices[-1]
    cosine_np = debug[f"layer_{last_b}"]["cosine_map"][0].cpu().numpy()  # [H_p, W_p]
    cosine_up = F.interpolate(
        torch.tensor(cosine_np).unsqueeze(0).unsqueeze(0),
        size=(H_s, W_s), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    # 이미지 numpy 변환
    scene_np  = scene_t[0].permute(1, 2, 0).cpu().numpy()   # [H, W, 3]
    target_np = target_t[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

    # ── 시각화 ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("SimilarityStream — target vs scene", fontsize=13)

    axes[0].imshow(target_np)
    axes[0].set_title(f"Target\n({W_t}×{H_t})")
    axes[0].axis("off")

    axes[1].imshow(scene_np)
    axes[1].set_title(f"Scene\n({W_s}×{H_s})")
    axes[1].axis("off")

    # DINOv3 cosine similarity (pre-head, meaningful without training)
    im2 = axes[2].imshow(cosine_up, cmap="hot", vmin=-1, vmax=1)
    axes[2].set_title(f"DINOv3 Cosine Sim\n(Block {last_b}, pre-head)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # SimilarityStream 출력 (head 미학습 → random, 구조 검증용)
    im3 = axes[3].imshow(prob, cmap="hot", vmin=0, vmax=1)
    axes[3].set_title("S_pred  sigmoid(logit)\n(head 미학습 — 구조 검증용)")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
