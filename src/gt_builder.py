"""
Segmentation map + mapping.json → 카테고리 유사도 기반 GT 생성.

sky_ws gt_similarity.py와 동일한 방식:
    타겟 카테고리가 fruit이면 scene 안의
      - fruit 픽셀      → 0.8  (same category)
      - packaged_food  → 0.5  (similar)
      - book / toy     → 0.2  (dissimilar)
      - background     → 0.0

USD 이름 → 카테고리 매핑은 USD_CATEGORY_MAP (명시적) → _infer_category (키워드 fallback) 순으로 처리.
새 오브젝트 추가 시 USD_CATEGORY_MAP에만 항목을 추가하면 됨.
"""

import json
import os
import re
from typing import NamedTuple

import cv2
import numpy as np
import yaml

# --------------------------------------------------------------------------- #
# 카테고리 유사도 점수표 (sky_ws와 동일)
# --------------------------------------------------------------------------- #
SIMILARITY_MAP: dict[str, dict[str, float]] = {
    "fruit":         {"fruit": 0.8, "packaged_food": 0.5, "book": 0.2, "toy": 0.2},
    "packaged_food": {"fruit": 0.5, "packaged_food": 0.8, "book": 0.2, "toy": 0.2},
    "book":          {"fruit": 0.2, "packaged_food": 0.2, "book": 0.8, "toy": 0.5},
    "toy":           {"fruit": 0.2, "packaged_food": 0.2, "book": 0.5, "toy": 0.8},
}

# --------------------------------------------------------------------------- #
# USD 이름 → 카테고리 명시적 매핑
# --------------------------------------------------------------------------- #
USD_CATEGORY_MAP: dict[str, str] = {
    # fruit
    "Apple":     "fruit",
    "Avocado01": "fruit",
    "Lime01":    "fruit",
    "Orange_03": "fruit",
    # packaged_food (YCB 스타일 이름)
    "005_tomato_soup_can": "packaged_food",
    "006_mustard_bottle":  "packaged_food",
    "008_pudding_box":     "packaged_food",
    "010_potted_meat_can": "packaged_food",
    # book  ← OmniConnect2015 는 "OMNI CONNECTS PEOPLE 2015" 책이므로 book
    "Book_GetKnowPPU": "book",
    "Book_Greener":    "book",
    "Book_02":         "book",
    "OmniConnect2015": "book",
    # toy
    "Shield_Controller": "toy",
    "Ball_Walnut":       "toy",
    "RubixCube":         "toy",
    "toy_truck":         "toy",
}


def _infer_category(usd_name: str) -> str:
    """USD_CATEGORY_MAP에 없는 오브젝트를 키워드로 카테고리 추론."""
    n = usd_name.lower()
    if re.search(r'apple|avocado|lime|orange|banana|fruit|grape|lemon|mango|berry', n):
        return "fruit"
    if re.search(r'book|magazine|novel|journal', n):
        return "book"
    if re.search(r'can|soup|bottle|box|spam|mustard|pudding|sauce|food|potted', n):
        return "packaged_food"
    return "toy"  # 분류 불가 → toy로 fallback


def usd_to_category(usd_name: str) -> str:
    """USD 이름 → 카테고리 문자열."""
    return USD_CATEGORY_MAP.get(usd_name, _infer_category(usd_name))


# --------------------------------------------------------------------------- #
# mapping.json 로딩
# --------------------------------------------------------------------------- #

def load_scene_mapping(mapping_json_path: str) -> dict[str, tuple]:
    """
    {usd_name: (B, G, R)} 반환.
    JSON 필드명은 color_rgb이지만 실제 저장 순서가 BGR
    (Isaac Sim 파이프라인 네이밍 버그 — sky_ws와 동일 관례).
    """
    with open(mapping_json_path) as f:
        m = json.load(f)
    return {k: tuple(v["color_rgb"]) for k, v in m.items()}


# --------------------------------------------------------------------------- #
# color_to_score 빌드 & GT 렌더링
# --------------------------------------------------------------------------- #

def build_color_to_score(scene_mapping: dict[str, tuple],
                          target_category: str,
                          target_usd_name: str | None = None) -> dict[tuple, float]:
    """
    {(B,G,R): similarity_score} 딕셔너리 생성.

    점수 체계:
        - 타겟과 같은 USD 오브젝트 → 1.0   (same object)
        - 타겟과 같은 카테고리      → 0.8   (same category)
        - 비슷한 카테고리           → 0.5   (similar, e.g. fruit↔packaged_food)
        - 다른 카테고리             → 0.2   (different)
        - 배경                    → 0.0   (not matched)

    target_usd_name: 씬 mapping.json 내 실제 USD 이름 (예: "Apple", "Avocado01").
                     None이면 same-object=1.0 구분 없이 카테고리 점수만 적용.
    """
    target_scores = SIMILARITY_MAP.get(target_category, {})
    color_to_score: dict[tuple, float] = {}
    for usd_name, bgr in scene_mapping.items():
        if target_usd_name and usd_name == target_usd_name:
            score = 1.0                                    # same object
        else:
            obj_cat = usd_to_category(usd_name)
            score = target_scores.get(obj_cat, 0.0)
        color_to_score[bgr] = score
    return color_to_score


def render_gt_map(seg_bgr: np.ndarray,
                  color_to_score: dict[tuple, float]) -> np.ndarray:
    """
    seg_bgr : (H, W, 3) BGR uint8 — Isaac Sim seg 이미지
    반환    : (H, W) float32 GT [0, 1]
    """
    h, w = seg_bgr.shape[:2]
    gt = np.zeros((h, w), dtype=np.float32)
    for (b, g, r), score in color_to_score.items():
        mask = ((seg_bgr[:, :, 0] == b) &
                (seg_bgr[:, :, 1] == g) &
                (seg_bgr[:, :, 2] == r))
        gt[mask] = score
    return gt


# --------------------------------------------------------------------------- #
# 편의 함수: 파일 경로에서 바로 GT 계산
# --------------------------------------------------------------------------- #

def compute_gt(seg_path: str,
               mapping_json_path: str,
               target_category: str,
               target_usd_name: str | None = None) -> np.ndarray | None:
    """
    seg_path         : scene/.../seg/<fname>.png
    mapping_json_path: scene/.../seg/<scene_id>_mapping.json
    target_category  : 타겟 카테고리 ("fruit", "packaged_food", ...)
    target_usd_name  : 씬 내 타겟 오브젝트의 USD 이름 (예: "Apple").
                       지정 시 해당 픽셀에 1.0 부여 (same-object 처리).
    반환             : (H, W) float32 GT, 실패 시 None
    """
    seg = cv2.imread(seg_path)
    if seg is None:
        return None
    if not os.path.isfile(mapping_json_path):
        return None
    scene_mapping = load_scene_mapping(mapping_json_path)
    color_to_score = build_color_to_score(scene_mapping, target_category, target_usd_name)
    return render_gt_map(seg, color_to_score)


def scene_id_from_fname(fname: str) -> str:
    """
    'scene00001_env0003_center.png' → 'scene00001'
    mapping.json 이름 조합에 사용.
    """
    return fname.split("_")[0]


# --------------------------------------------------------------------------- #
# scenes.yaml 로더
# --------------------------------------------------------------------------- #

class SceneEntry(NamedTuple):
    scene:  str   # "Fruit/Apple"  → data/scene/Fruit/Apple/scene/rgb|seg/
    target: str   # "Fruit/Apple"  → data/target/Fruit/Apple/target.png
    usd:    str   # "Apple"        → mapping.json USD 이름 (same-object=1.0 판정)


def load_scene_config(yaml_path: str) -> list[SceneEntry]:
    """
    config/scenes.yaml → [SceneEntry, ...] 로드.
    enabled: false 항목은 제외.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    entries = []
    for item in data.get("scenes", []):
        if not item.get("enabled", True):
            continue
        entries.append(SceneEntry(
            scene=str(item["scene"]),
            target=str(item["target"]),
            usd=str(item["usd"]),
        ))
    return entries
