# Similarity Stream 구현 지시서

## 0. 작업 목표

현재 repository `Holiclife-KTH/2D_PDM_DINO`에서 `mask_module.py`를 수정하거나, 가능하면 별도 파일 `similarity_stream.py`를 새로 만들어라.

목표는 다음 구조의 **Similarity Stream**을 구현하는 것이다.

```text
Scene RGB Image
Target RGB Image
        ↓
Shared Frozen DINOv3 Encoder
        ↓
Scene multi-layer patch features
Target multi-layer patch features
        ↓
Layer-wise Scene–Target Interaction
        ↓
Layer-wise Learnable Matching Blocks
        ↓
Multi-level Aggregation
        ↓
Similarity Feature F_s
        ├── Adaptive Fusion으로 전달할 feature
        └── Auxiliary Similarity Head → S_pred
```

최종적으로 forward 함수는 다음을 반환해야 한다.

```python
{
    "similarity_feature": F_s,   # [B, fusion_dim, H_p, W_p]
    "similarity_map": S_pred,    # [B, 1, H_p, W_p] or optionally upsampled [B, 1, H, W]
    "layer_features": {...},     # optional debug
    "matching_features": {...},  # optional debug
}
```

---

## 1. 현재 코드 기반

현재 `mask_module.py`에는 다음 요소가 이미 있다.

```python
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vit7b16-pretrain-sat493m")
model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-sat493m").to(device)
patch_size = model.config.patch_size
num_register_tokens = model.config.num_register_tokens
num_layers = model.config.num_hidden_layers
```

이 구조는 유지하되, `num_register_tokens`는 안전하게 다음처럼 바꿔라.

```python
num_register_tokens = getattr(model.config, "num_register_tokens", 0)
```

현재 코드는 `outputs = model(**inputs, output_hidden_states=True)`를 사용해서 hidden states를 얻고, `outputs.hidden_states[b + 1]`에서 block `b`의 출력을 가져온다. 이 방식은 유지한다.

다만 현재 코드는 patch token을 `[hw, D]` 형태로 저장하고 있다. Similarity Stream에서는 반드시 feature map 형태로 변환해야 한다.

```python
patch_tokens = hs[:, 1 + num_register_tokens:, :]  # [B, hw, D]
patch_map = patch_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)
```

출력 형태:

```text
[B, D, H_p, W_p]
```

예를 들어 입력이 `480×640`, patch size가 `16`, DINO feature dimension이 `4096`이면:

```text
[B, 4096, 30, 40]
```

---

## 2. 입력 이미지 크기 기준

우선 다음 입력 크기를 기준으로 구현하라.

```text
Scene RGB  : [B, 3, 480, 640]
Target RGB : [B, 3, 480, 640]
Patch size : 16
Patch grid : 30 × 40
```

640과 480은 16으로 나누어떨어지므로 padding 없이 가능하다.

하지만 일반성을 위해 입력 이미지 크기가 patch size의 배수가 아닐 경우 padding하는 함수를 작성하라.

```python
def pad_to_patch_multiple_tensor(x, patch_size):
    """
    x: [B, C, H, W]
    return:
        x_padded: [B, C, H_pad, W_pad]
        meta: dict with original/padded size
    """
```

주의: resize로 크기를 맞추지 말고 padding을 사용하라. GT map, depth map, PDM 좌표와의 정합성을 위해 원본 geometry를 왜곡하면 안 된다.

---

## 3. DINO feature extractor 구현

`DINOFeatureExtractor` 클래스를 작성하라.

### 역할

Scene과 Target image를 같은 frozen DINOv3 encoder에 넣고, 지정된 layer의 patch feature map을 추출한다.

### 요구사항

```python
class DINOFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vit7b16-pretrain-sat493m",
        layer_indices: Optional[List[int]] = None,
        freeze: bool = True,
    ):
        ...
```

### 기본 layer 선택

Similarity stream에서는 후반부 3개 layer를 사용한다.

예를 들어 DINO block 수가 `num_layers`일 때:

```python
layer_indices = [num_layers - 7, num_layers - 4, num_layers - 1]
```

단, index가 음수가 되지 않도록 clamp하라.

```python
indices = [max(0, min(i, num_layers - 1)) for i in indices]
indices = sorted(list(dict.fromkeys(indices)))
```

실제 사용 모델(`facebook/dinov3-vit7b16-pretrain-sat493m`)은 40-layer ViT이므로:

```text
[33, 36, 39]   # num_layers=40 기준: [40-7, 40-4, 40-1]
```

이 된다.

### forward 입력

가능하면 tensor 입력을 받도록 구현하라.

```python
def forward(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
    """
    images: [B, 3, H, W], float tensor, range can be [0,1]
    return:
        features: dict
            key: layer index
            value: [B, D, H_p, W_p]
    """
```

Hugging Face processor를 꼭 써야 한다면 PIL 기반 처리도 가능하지만, 학습 모듈로 쓰려면 tensor 입력을 받는 편이 좋다. 최소한 현재 단계에서는 PIL path 입력 버전과 tensor 입력 버전을 분리하라.

### frozen 처리

DINOv3 backbone은 기본적으로 freeze한다.

```python
if freeze:
    for p in self.model.parameters():
        p.requires_grad = False
    self.model.eval()
```

forward에서도 DINO 부분은 `torch.no_grad()` 또는 `torch.inference_mode()`를 사용해도 된다. 단, projection/matching block은 학습되어야 한다.

---

## 4. Target feature 처리

Scene feature와 Target feature는 같은 DINO encoder에서 같은 layer로 추출된다.

각 layer에서:

```text
Scene feature  S_i: [B, D, H_p, W_p]
Target feature T_i: [B, D, H_p, W_p]
```

단, matching에서는 target image가 “찾고 싶은 물체 하나”이므로 target feature map을 global query로 요약한다.

```python
q_i = global_average_pool(T_i)  # [B, D]
```

추후 target object mask가 있으면 masked pooling을 지원할 수 있도록 함수 구조를 열어둔다.

```python
def pool_target_feature(target_feature, target_mask=None):
    """
    target_feature: [B, D, H_p, W_p]
    target_mask: optional [B, 1, H_p, W_p]
    return q: [B, D]
    """
```

현재는 target mask가 없다고 가정하고 global average pooling을 사용한다.

---

## 5. Layer-wise Matching 구조

DINO 후반부 3개 layer를 사용한다고 하면:

```text
Scene:  S1, S2, S3
Target: T1, T2, T3
```

각 layer마다 다음을 수행한다.

```text
S_i, T_i
↓
Target pooling
↓
Channel alignment
↓
Target spatial expansion
↓
Interaction tensor construction
↓
Learnable Matching Block
↓
M_i
```

---

## 6. Channel Alignment

각 layer의 DINO feature dimension `D`를 `align_dim=128`로 줄인다.

Scene feature:

```python
S_i: [B, 4096, 30, 40]   # D=4096
Conv1x1(4096 → 128)
Z_i: [B, 128, 30, 40]
```

Target query:

```python
q_i: [B, 4096]            # D=4096
Linear(4096 → 128)
p_i: [B, 128]
```

그 다음 target query를 scene spatial size로 expand한다.

```python
Q_i = p_i[:, :, None, None].expand(-1, -1, H_p, W_p)
# [B, 128, 30, 40]
```

주의: `expand`는 view 기반이므로 이후 연산에서 문제가 생기면 `.contiguous()`를 적절히 사용하라.

---

## 7. Interaction Tensor Construction

각 layer마다 interaction tensor를 만든다.

```python
I_i = concat([
    Z_i,
    Q_i,
    torch.abs(Z_i - Q_i),
    Z_i * Q_i,
    cosine_map
], dim=1)
```

cosine map:

```python
cosine_map = F.cosine_similarity(Z_i, Q_i, dim=1, eps=1e-6).unsqueeze(1)
```

shape:

```text
Z_i              : [B, 128, 30, 40]
Q_i              : [B, 128, 30, 40]
abs diff         : [B, 128, 30, 40]
product          : [B, 128, 30, 40]
cosine_map       : [B, 1,   30, 40]

I_i              : [B, 513, 30, 40]
```

주의: 이 interaction construction은 학습 파라미터가 없는 non-parametric layer다. 하지만 gradient는 projection layer와 matching block으로 흐를 수 있어야 한다.

---

## 8. Learnable Matching Block

각 layer마다 matching block을 둔다.

```python
class LayerWiseMatchingBlock(nn.Module):
    def __init__(self, in_ch=513, hidden_ch=128, out_ch=64):
        ...
```

구조:

```text
Input I_i: [B, 513, 30, 40]

1×1 Conv: 513 → 128
Norm
GELU

3×3 Conv: 128 → 128
Norm
GELU

1×1 Conv: 128 → 64
Norm
GELU

Output M_i: [B, 64, 30, 40]
```

Norm은 `BatchNorm2d` 또는 `GroupNorm`을 사용하라. batch size가 작을 가능성이 크므로 `GroupNorm`을 추천한다.

예:

```python
nn.GroupNorm(num_groups=8, num_channels=hidden_ch)
```

---

## 9. Matching block은 병렬 구조로 구현

matching block을 순차적으로 쌓지 말고, layer별로 병렬로 적용한다.

```text
I1 → MatchingBlock1 → M1
I2 → MatchingBlock2 → M2
I3 → MatchingBlock3 → M3
```

그 다음 `M1, M2, M3`를 channel 방향으로 concat한다.

```python
M_cat = torch.cat([M1, M2, M3], dim=1)
```

shape:

```text
M1, M2, M3 : [B, 64, 30, 40]
M_cat      : [B, 192, 30, 40]
```

각 layer의 matching block은 일단 weight를 공유하지 말고 별도로 둔다. 즉, `nn.ModuleList`를 사용하라.

```python
self.matching_blocks = nn.ModuleList([
    LayerWiseMatchingBlock(...) for _ in layer_indices
])
```

추후 ablation으로 shared matching block을 추가할 수 있게 구조를 열어두면 좋다.

---

## 10. Similarity Feature Head

`M_cat`을 받아 fusion용 feature `F_s`를 만든다.

```python
class SimilarityFeatureHead(nn.Module):
    def __init__(self, in_ch=64*3, out_ch=128):  # 3 = len(layer_indices)
        ...
```

구조:

```text
M_cat: [B, 192, 30, 40]

1×1 Conv: 192 → 128
GroupNorm
GELU

3×3 Conv: 128 → 128
GroupNorm
GELU

Residual 3×3 Conv block, optional

Output F_s: [B, 128, 30, 40]
```

`F_s`는 이후 Adaptive Fusion에 들어가는 Similarity Stream의 메인 출력이다.

---

## 11. Auxiliary Similarity Map Head

`F_s`에서 auxiliary semantic/similarity map을 출력한다.

```python
class SimilarityAuxHead(nn.Module):
    def __init__(self, in_ch=128):
        ...
```

구조:

```text
F_s: [B, 128, 30, 40]

3×3 Conv: 128 → 64
GroupNorm
GELU

1×1 Conv: 64 → 1
Sigmoid

S_pred_low: [B, 1, 30, 40]
```

기본은 low-resolution map을 반환한다.  
옵션으로 원본 해상도까지 upsample할 수 있게 하라.

```python
if upsample_to_input:
    S_pred = F.interpolate(S_pred_low, size=(H, W), mode="bilinear", align_corners=False)
```

반환 dict에는 둘 다 넣어도 좋다.

```python
{
    "similarity_map_low": S_pred_low,
    "similarity_map": S_pred_upsampled
}
```

---

## 12. 최종 SimilarityStream 클래스

다음 클래스를 구현하라.

```python
class SimilarityStream(nn.Module):
    def __init__(
        self,
        dino_model_name="facebook/dinov3-vit7b16-pretrain-sat493m",
        layer_indices=None,
        align_dim=128,
        match_dim=64,
        fusion_dim=128,
        freeze_dino=True,
        upsample_aux=True,
    ):
        ...
```

forward:

```python
def forward(
    self,
    scene_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    scene_rgb: [B, 3, H, W]
    target_rgb: [B, 3, H, W]
    target_mask: optional [B, 1, H, W]

    return:
        similarity_feature: [B, fusion_dim, H_p, W_p]
        similarity_map_low: [B, 1, H_p, W_p]
        similarity_map: [B, 1, H, W] if upsample_aux=True
        matching_features: optional dict/list
    """
```

forward 내부 순서:

```text
1. scene_rgb, target_rgb를 patch size 배수로 padding
2. DINO feature 추출
3. 각 layer마다:
   a. scene feature S_i 가져오기
   b. target feature T_i 가져오기
   c. target feature pooling → q_i
   d. channel alignment
   e. target spatial expand
   f. interaction tensor 생성
   g. matching block → M_i
4. M_i concat → M_cat
5. Similarity Feature Head → F_s
6. Aux Similarity Head → S_pred_low
7. 필요 시 input size로 upsample/crop
8. dict 반환
```

padding한 경우, upsample된 map은 원본 `H, W`로 crop해서 반환하라.

---

## 13. Shape check 테스트 코드 추가

별도 test 또는 `if __name__ == "__main__":` 아래에 다음 shape check를 넣어라.

```python
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimilarityStream(
        dino_model_name="facebook/dinov3-vit7b16-pretrain-sat493m",
        align_dim=128,
        match_dim=64,
        fusion_dim=128,
        freeze_dino=True,
        upsample_aux=True,
    ).to(device)

    scene = torch.rand(1, 3, 480, 640, device=device)
    target = torch.rand(1, 3, 480, 640, device=device)

    with torch.no_grad():
        out = model(scene, target)

    print("similarity_feature:", out["similarity_feature"].shape)
    print("similarity_map_low:", out["similarity_map_low"].shape)
    print("similarity_map:", out["similarity_map"].shape)
```

기대 출력:

```text
similarity_feature: [1, 128, 30, 40]
similarity_map_low: [1, 1, 30, 40]
similarity_map: [1, 1, 480, 640]
```

DINO model에 따라 feature dimension `D`는 달라질 수 있으므로 코드에서 hard-code하지 말고, 첫 forward에서 feature dim을 읽거나 model config에서 가져오라.

---

## 14. Loss 계산용 인터페이스

학습 시에는 다음 loss를 계산할 수 있어야 한다.

```python
loss_sim = criterion(out["similarity_map_low"], S_gt_low)
```

또는 upsample map 기준:

```python
loss_sim = criterion(out["similarity_map"], S_gt)
```

초기에는 low-resolution map 기준 loss를 추천한다.

```python
S_gt_low = F.interpolate(S_gt, size=out["similarity_map_low"].shape[-2:], mode="bilinear", align_corners=False)
```

---

## 15. 구현 시 주의사항

1. DINO backbone은 frozen이어야 한다.
2. DINO feature extraction과 matching block은 명확히 분리하라.
3. interaction tensor 생성은 학습 파라미터 없는 고정 연산이지만, matching block 입력으로 사용된다.
4. cosine similarity만 최종 map으로 쓰면 안 된다. cosine은 interaction tensor의 일부 cue일 뿐이다.
5. target feature는 scene과 동일한 DINO layer에서 추출하되, matching 단계에서는 global pooling으로 query vector를 만든다.
6. layer-wise matching은 병렬로 수행한다.
7. `M1, M2, M3`를 concat한 뒤 `SimilarityFeatureHead`에서 `F_s`를 만든다.
8. `S_pred`는 auxiliary supervision용 map이며, 최종 PDM이 아니다.
9. 코드는 나중에 Occlusion Stream, Complexity Stream, Adaptive Fusion과 연결 가능하도록 modular하게 작성하라.
10. 현재 `mask_module.py`는 시각화용 코드가 섞여 있으므로, 가능하면 학습용 모듈은 `similarity_stream.py`로 분리하라.

---

## 16. 개발 완료 기준

다음 조건을 만족하면 완료로 본다.

```text
1. SimilarityStream 클래스가 존재한다.
2. Scene/Target RGB tensor를 입력으로 받는다.
3. DINOv3에서 지정 layer별 patch feature map을 추출한다.
4. 각 layer별 Scene–Target interaction tensor를 만든다.
5. 각 interaction tensor가 layer-wise matching block을 통과한다.
6. matching feature들이 concat된다.
7. fusion용 similarity_feature F_s가 출력된다.
8. auxiliary similarity map S_pred가 출력된다.
9. 480×640 입력 기준 output shape이 다음과 같다.
   - similarity_feature: [1, 128, 30, 40]
   - similarity_map_low: [1, 1, 30, 40]
   - similarity_map: [1, 1, 480, 640]
10. DINO backbone parameter는 requires_grad=False다.
11. Projection, MatchingBlock, FeatureHead, AuxHead는 requires_grad=True다.
```

---

## 17. 최종 구조 요약

구현하려는 Similarity Stream은 다음 구조다.

```text
Scene RGB ──→ Shared Frozen DINOv3 ──→ S1, S2, S3
Target RGB ─→ Shared Frozen DINOv3 ──→ T1, T2, T3

For each layer i:
    T_i → Target Pooling → q_i
    S_i → Scene Projection → Z_i
    q_i → Target Projection → p_i
    p_i → Spatial Expand → Q_i
    [Z_i, Q_i, |Z_i-Q_i|, Z_i⊙Q_i, cosine] → I_i
    I_i → MatchingBlock_i → M_i

M1, M2, M3
→ Concat
→ SimilarityFeatureHead
→ F_s

F_s
├→ Adaptive Fusion later
└→ SimilarityAuxHead
   → S_pred
```

이 지시서를 기준으로 우선 `similarity_stream.py`를 구현하고, 기존 `mask_module.py`는 feature 시각화/검증용으로 남겨두는 것이 좋다.
