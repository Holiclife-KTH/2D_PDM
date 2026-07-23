# Zero-Shot Similarity Map Pipeline — Architecture

## 1. 개요

임의의 타겟 이미지(또는 이미지 + 텍스트)를 쿼리로 받아,
씬 이미지 안에서 해당 물체와 얼마나 유사한지를 픽셀 단위 유사도 맵(0~1)으로 출력하는 zero-shot 파이프라인.

**핵심 특성**
- 추론 시 카테고리 레이블 불필요 — 이미지(+ 물체 이름 텍스트) 만으로 동작
- 학습 데이터에 없던 새 물체에 대해 zero-shot 일반화
- 학습 대상 파라미터 **3.58M** (전체 모델의 약 0.3%)

---

## 2. 모델 구성

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INFERENCE FLOW                              │
│                                                                     │
│  TARGET  ─────────────────────────────────────────────────────────┐ │
│  image                                                            ↓ │
│    └─► [FG mask + crop]                                           │ │
│              ↓                         ↓                          │ │
│        DINOv3 vits16              SigLIP so400m                   │ │
│        (frozen, 224px)            (frozen, 384px)                 │ │
│              ↓                         ↓                          │ │
│     appearances[L]            img_embed (1152-d)                  │ │
│     [(384,)] × 4              +(text_embed/2 선택)                │ │
│              │                    semantic_raw (1152-d)           │ │
│              │                         ↓                          │ │
│              │                  proj[L] ★학습★                    │ │
│              │                  Linear(1152→384) × 4              │ │
│              │                    sem_proj[L] (384,) × 4          │ │
│              └────────── + ─────────────┘                         │ │
│                       query[L]  (384,) × 4   ←────────────────── ┘ │
│                           ↓                                         │
│  SCENE  ──────────────────────────────────────────┐                 │
│  image                                            ↓                 │
│    └─► DINOv3 vits16 (frozen, full res)                            │
│              ↓                                                      │
│       scene_feats[L]  (B, 384, Hp, Wp) × 4                         │
│              ↓                                                      │
│         ZeroShotHead ★학습★                                        │
│              ↓                                                      │
│       prob_map  (B, 1, H, W)  [0, 1]                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 구성 요소

### 3-1. DINOv3 vits16 — Appearance Encoder (Frozen)

| 항목 | 값 |
|---|---|
| 모델 | `dinov3_vits16` |
| 가중치 | `model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| 출력 차원 | 384-d |
| 패치 크기 | 16 px |
| 사용 레이어 | L2, L5, L8, L11 (총 12층 중 4개) |
| 파라미터 | ~21M (frozen) |
| 역할 | 씬/타겟 공통 appearance 인코딩 |

`get_intermediate_layers(n=(2,5,8,11), reshape=True, return_class_token=True)`
→ 각 레이어에서 `(patch_grid: B×C×Hp×Wp, cls: B×C)` 쌍 반환

**타겟 쪽**: 224px 리사이즈 → FG 마스크로 masked average pooling → `(384,)` 벡터 × 4 레이어

**씬 쪽**: 원본 해상도(16의 배수) → 패치 그리드 그대로 사용

---

### 3-2. SigLIP so400m — Semantic Encoder (Frozen)

| 항목 | 값 |
|---|---|
| 모델 | `google/siglip-so400m-patch14-384` |
| 출처 | Gemma3 4B의 실제 vision encoder |
| 입력 해상도 | 384 × 384 |
| 출력 차원 | 1152-d (pooler_output) |
| 파라미터 | ~400M (frozen) |
| 역할 | 타겟의 semantic embedding 추출 (타겟 측만 적용) |

**이미지 임베딩**: SigLIP vision_model → L2-normalize → `(1, 1152)`

**텍스트 임베딩** (선택):
- 텍스트 입력: `"a photo of {specific}, a type of {category}"`
  - 예) `"a photo of an apple, a type of fruit"`
- SigLIP text_model → L2-normalize → `(1, 1152)`
- 같은 SigLIP 공간에 정렬되어 있으므로 평균 fusion 가능:
  `semantic_raw = L2_norm((img_embed + text_embed) / 2)`
- **텍스트 없이** 이미지만 줘도 동작 (image-only mode)
- 텍스트를 주면 모호한 타겟(어두운 조명 등)을 카테고리 정보로 보완

> **비대칭 설계**: SigLIP은 타겟 측에만 붙어 있음.  
> 씬 측은 DINOv3만 사용 → 씬 안의 새 물체는 시각적 외관으로만 매칭됨.

---

### 3-3. SigLIP Projection — semantic.proj (Trainable)

```python
proj = nn.ModuleList([
    nn.Linear(1152, 384)   # bias 포함
    for _ in range(4)      # 레이어별 독립 투영
])
```

| 항목 | 값 |
|---|---|
| 구조 | `Linear(1152 → 384) × 4` (레이어별 독립) |
| 파라미터 | **1,771,008** (학습) |
| 역할 | SigLIP 공간(1152-d)을 DINOv3 공간(384-d)으로 정렬 |

**Additive Fusion**: `query[l] = appearance[l] + proj[l](semantic_raw)`
- appearance: 시각적 디테일 (DINOv3)
- proj(semantic): 카테고리 의미 (SigLIP → 정렬)
- 합산 결과가 씬의 DINOv3 패치와 직접 비교됨

---

### 3-4. ZeroShotHead (Trainable)

```
입력 채널: scene_feat(384) + target_broadcast(384) + cosine_sim(1) = 769
```

```
scene_feats[l]: (B, 384, Hp, Wp)
query[l]:       (B, 384) → broadcast → (B, 384, Hp, Wp)
cosine:         (patch_norm · query_norm) 정규화 후 [0,1]로 shift
cat → (B, 769, Hp, Wp) → MatchingBlock[l]
```

**MatchingBlock** (레이어당 1개, 총 4개):
```
Conv2d(769, 64, 3, pad=1) → GroupNorm(8,64) → ReLU
Conv2d(64,  64, 1)        → GroupNorm(8,64) → ReLU
```

**Fusion + 출력**:
```
cat([block_0..3], dim=1): (B, 256, Hp, Wp)
fuse: Conv2d(256,64,1) → GroupNorm(8,64) → ReLU
head: Conv2d(64, 1, 1) → Sigmoid → prob_patch (B,1,Hp,Wp)
upsample(bilinear) → prob_full (B,1,H,W)
```

| 서브모듈 | 파라미터 |
|---|---|
| `blocks` (MatchingBlock × 4) | 1,789,696 |
| `fuse` (Conv+GN) | 16,576 |
| `head` (Conv 1×1) | 65 |
| **합계** | **1,806,337** |

---

### 3-5. 전체 파라미터 요약

| 모듈 | 파라미터 | 상태 |
|---|---|---|
| DINOv3 vits16 | ~21M | **Frozen** |
| SigLIP so400m encoder | ~400M | **Frozen** |
| SigLIP proj (Linear×4) | **1,771,008** | **학습** |
| ZeroShotHead | **1,806,337** | **학습** |
| **학습 대상 합계** | **3,577,345** | |

---

## 4. GT (Ground Truth) 생성

학습 GT는 precomputed 이미지가 아닌 seg + mapping.json에서 실시간 계산.

### 파일 구조
```
data/scene/<Category>/<Specific>/scene/
    rgb/scene00001_env0003_center.png
    seg/scene00001_env0003_center.png   ← 오브젝트별 BGR 색상으로 세그먼테이션
    seg/scene00001_mapping.json         ← {USD_name: {color_rgb: [B,G,R]}}
```

### GT 점수 계산 (sky_ws 동일 방식)

```
타겟 물체와의 관계        GT 점수
─────────────────────────────────
같은 USD 오브젝트 (exact)   1.0   ← same object
같은 카테고리              0.8   ← same category
유사한 카테고리             0.5   ← similar
다른 카테고리              0.2   ← different
배경 / 미등록              0.0
```

**카테고리 유사도 행렬:**

|           | fruit | pkg_food | book | toy |
|-----------|-------|----------|------|-----|
| fruit     | 0.8   | 0.5      | 0.2  | 0.2 |
| pkg_food  | 0.5   | 0.8      | 0.2  | 0.2 |
| book      | 0.2   | 0.2      | 0.8  | 0.5 |
| toy       | 0.2   | 0.2      | 0.5  | 0.8 |

**USD → 카테고리 매핑 (gt_builder.py):**
```
fruit       : Apple, Avocado01, Lime01, Orange_03
packaged_food: 005_tomato_soup_can, 006_mustard_bottle, 008_pudding_box, 010_potted_meat_can
book        : Book_02, Book_GetKnowPPU, Book_Greener, OmniConnect2015
toy         : Ball_Walnut, Shield_Controller, RubixCube, toy_truck
```

> `OmniConnect2015`는 "OMNI CONNECTS PEOPLE 2015" **책**이므로 `book` 분류.

---

## 5. 데이터 구조

```
th_ws/
├── config/
│   └── scenes.yaml          ← 씬↔타겟 매핑 설정 (새 씬 추가 시 이 파일만 수정)
├── data/
│   ├── scene/
│   │   └── <Category>/<Specific>/scene/
│   │       ├── rgb/   *.png  (학습 입력)
│   │       └── seg/   *.png + *_mapping.json  (GT 계산)
│   └── target/
│       └── <Category>/<Specific>/target.png   (타겟 이미지)
├── src/
│   ├── zeroshot_pipeline.py  (모델 정의)
│   └── gt_builder.py         (GT 계산 + 씬 설정 로더)
├── train/
│   └── train_zeroshot.py
└── validate/
    └── validate_zeroshot.py
```

현재 학습 씬 (16개):

| 카테고리 | 씬 / 타겟 | USD 이름 |
|---|---|---|
| Fruit | Apple | Apple |
| Fruit | Avocado | Avocado01 |
| Fruit | Lime | Lime01 |
| Fruit | Orange | Orange_03 |
| Packaged_food | SPAM | 010_potted_meat_can |
| Packaged_food | Tomato_soup_can | 005_tomato_soup_can |
| Packaged_food | Mustard | 006_mustard_bottle |
| Packaged_food | Pudding_box | 008_pudding_box |
| Book | Book_1 | Book_02 |
| Book | Book_2 | Book_GetKnowPPU |
| Book | Book_3 | Book_Greener |
| Book | Book_4 | OmniConnect2015 |
| Toy | Ball | Ball_Walnut |
| Toy | Gamepad | Shield_Controller |
| Toy | RubixCube | RubixCube |
| Toy | Toy_truck | toy_truck |

---

## 6. 학습

### 데이터 분할
- **장면(scene_id) 단위 분할** — 같은 씬의 여러 뷰가 train/val에 나뉘지 않도록 leakage 방지
- 기본 분할 비율: train 80% / val 20%

### Loss
```
loss = MSE(prob_patch, GT_downsampled)
```
- GT를 패치 해상도(H/16 × W/16)로 avg_pool2d 다운샘플
- prob_patch와 같은 공간에서 MSE 계산

### Optimizer / Schedule
| 항목 | 값 |
|---|---|
| Optimizer | AdamW (weight_decay=1e-4) |
| LR | 1e-3 |
| Schedule | CosineAnnealingLR |
| Early Stop | patience=10, min_delta=5% |
| Batch | 512 |
| Epochs | 200 |

### Gradient 흐름
```
scene → DINOv3(frozen, no_grad) → scene_feats
target.png → DINOv3(frozen) + SigLIP(frozen) → appearances, semantic_raw  (캐시됨)
                                                        ↓
                                         proj[l] ★gradient★ → sem_proj
                                  appearance + sem_proj = query  ★gradient★
                                              ↓
                                       head ★gradient★ → prob_patch
                                              ↓
                                           MSE loss
```

**핵심**: `target_cache`에 저장된 `appearances`와 `semantic_raw`는 frozen이므로
배치마다 재인코딩하지 않음. 단, `proj`(학습 대상)는 매 스텝 실시간 적용해야
gradient가 끊기지 않음.

### 체크포인트 형식
```python
{
    "epoch":       int,
    "val_mse":     float,
    "proj_state":  pipe.semantic.proj.state_dict(),
    "head_state":  pipe.head.state_dict(),
    # last ckpt에만 추가:
    "optim_state": optim.state_dict(),
}
```

DINOv3 / SigLIP encoder는 frozen이므로 저장하지 않음.

---

## 7. 추론 (Zero-Shot)

```python
pipe = ZeroShotPipeline(device="cuda")
# 체크포인트 복원
ckpt = torch.load("zeroshot_best.pt")
pipe.semantic.proj.load_state_dict(ckpt["proj_state"])
pipe.head.load_state_dict(ckpt["head_state"])

# 추론 (이미지만)
pred = pipe.predict_single(scene_bgr, target_bgr)

# 추론 (이미지 + 텍스트)
pred = pipe.predict_single(scene_bgr, target_bgr, label="a photo of a banana, a type of fruit")
```

- `label`은 학습 때 쓴 prompt 형식(`"a photo of {specific}, a type of {category}"`)과 동일하게 줄 때 최적
- 텍스트 없이 이미지만 줘도 동작 (image-only zero-shot)
- 학습 데이터에 없던 물체(예: Banana)에도 일반화됨

---

## 8. 씬 설정 관리

씬↔타겟 매핑은 `config/scenes.yaml`로 외부화되어 있음.

```yaml
scenes:
  - scene:  Fruit/Apple
    target: Fruit/Apple
    usd:    Apple

  - scene:  Fruit/Avocado    # enabled: false 를 추가하면 학습에서 제외
    target: Fruit/Avocado
    usd:    Avocado01
```

`gt_builder.load_scene_config(yaml_path)` → `List[SceneEntry(scene, target, usd)]`

train/validate 스크립트 모두 이 함수를 통해 설정을 로드하므로,
새 씬 추가 시 **`scenes.yaml` 한 곳만 수정**하면 됩니다.
