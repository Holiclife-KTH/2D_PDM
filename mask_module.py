import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA

IMAGE_PATH = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/output/rgb/000002_rgb.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vit7b16-pretrain-sat493m")
model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-sat493m").to(device)
patch_size = model.config.patch_size
num_register_tokens = model.config.num_register_tokens
num_layers = model.config.num_hidden_layers

# SegDINO 동일 방식으로 4개 중간 레이어 선택
if num_layers <= 12:
    LAYER_INDICES = [2, 5, 8, 11]
elif num_layers <= 24:
    LAYER_INDICES = [4, 11, 17, 23]
else:
    step = num_layers // 4
    LAYER_INDICES = [step - 1, 2 * step - 1, 3 * step - 1, num_layers - 1]
LAYER_INDICES = [min(b, num_layers - 1) for b in LAYER_INDICES]

# 최종 레이어가 포함되지 않은 경우 추가
final = num_layers - 1
if final not in LAYER_INDICES:
    LAYER_INDICES.append(final)

print(f"[INFO] num_layers={num_layers}  |  추출 레이어: {LAYER_INDICES}")


def extract_intermediate_features(image_path):
    image = Image.open(image_path).convert("RGB")
    W_orig, H_orig = image.size
    new_W = (W_orig // patch_size) * patch_size
    new_H = (H_orig // patch_size) * patch_size
    original = image.resize((new_W, new_H))

    inputs = processor(images=original, return_tensors="pt", do_resize=False).to(device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    h, w = new_H // patch_size, new_W // patch_size

    # hidden_states[0]=embedding, hidden_states[b+1]=block b 출력
    feature_grids = {}
    for b in LAYER_INDICES:
        hs = outputs.hidden_states[b + 1]                    # [1, 1+num_reg+hw, D]
        patch_tokens = hs[:, 1 + num_register_tokens:, :]   # [1, hw, D]
        patch_grid = patch_tokens.squeeze(0).cpu().numpy()  # [hw, D]
        feature_grids[b] = patch_grid

    return feature_grids, original, h, w


def pca_rgb(feats_hw_d):
    """[hw, D] → [hw, 3] → normalize to [0,1] per component"""
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(feats_hw_d)        # [hw, 3]
    for c in range(3):
        rgb[:, c] -= rgb[:, c].min()
        rgb[:, c] /= (rgb[:, c].max() + 1e-8)
    return rgb                                 # [hw, 3], 값 0~1


feature_grids, original, h, w = extract_intermediate_features(IMAGE_PATH)
print(f"[INFO] 패치 그리드: {h}×{w}  |  feature dim: {next(iter(feature_grids.values())).shape[-1]}")

# ── 시각화 ────────────────────────────────────────────────────────────────────
n = len(LAYER_INDICES)
fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

# 원본 이미지를 패치 해상도로 다운샘플해서 크기 맞춤
original_small = np.asarray(original.resize((w, h), Image.BILINEAR))
axes[0].imshow(original_small, interpolation="nearest")
axes[0].set_title(f"Original\n({w}×{h} patches)")
axes[0].axis("off")

for k, b in enumerate(LAYER_INDICES):
    rgb = pca_rgb(feature_grids[b]).reshape(h, w, 3)
    axes[k + 1].imshow(rgb, interpolation="nearest")
    label = f"Block {b} [Final]" if b == final else f"Block {b}"
    axes[k + 1].set_title(f"{label}\nPCA (3 comp.)")
    axes[k + 1].axis("off")

plt.suptitle(
    f"DINOv3 Feature Evolution Across Layers  |  patch_size={patch_size}  |  grid={h}×{w}",
    fontsize=12,
)
plt.tight_layout()
plt.show()
