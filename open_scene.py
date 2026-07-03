from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
from pathlib import Path

import cv2
import numpy as np
import omni.timeline
import omni.usd
import torch
import warp as wp
from isaacsim.core.experimental.utils.transform import euler_angles_to_quaternion
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from omni.kit.viewport.utility import get_active_viewport

SAM3_ROOT = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/sam3"
sys.path.insert(0, SAM3_ROOT)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from torchvision.transforms import v2 as T

# ── Config ───────────────────────────────────────────────────────────────────
USD_PATH    = "omniverse://192.168.0.160/Users/th_ws/test_scene.usd"
CAMERA_PATH = "/World/Camera"
RESOLUTION  = (720, 1280)        # (H, W)
CKPT_PATH   = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/sam3/weight/sam3.pt"
TEXT_PROMPT = "segment the objects in the scene"
MASK_ALPHA  = 0.5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR  = Path("/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/output")

COLORS = [
    (255,  60,  60), ( 60, 255,  60), ( 60,  60, 255),
    (255, 255,  60), (255,  60, 255), ( 60, 255, 255),
]

TOP_DOWN_ORI = euler_angles_to_quaternion(
    np.array([0.0, 0.0, 0.0]), degrees=True, extrinsic=False
).numpy().reshape(1, 4)   # wxyz

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "rgb").mkdir(exist_ok=True)
(OUTPUT_DIR / "masked").mkdir(exist_ok=True)
(OUTPUT_DIR / "masks").mkdir(exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────
def overlay_masks(image_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """image_rgb (H,W,3) uint8 RGB + masks (N,H,W) bool → BGR overlay"""
    overlay = image_rgb.copy()
    for i, mask in enumerate(masks):
        overlay[mask] = COLORS[i % len(COLORS)]
    blended = cv2.addWeighted(image_rgb, 1.0 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def save_step(step: int, image_rgb: np.ndarray, masks_np: np.ndarray | None) -> None:
    prefix = f"{step:06d}"
    cv2.imwrite(str(OUTPUT_DIR / "rgb"    / f"{prefix}_rgb.png"),
                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if masks_np is not None and masks_np.shape[0] > 0:
        cv2.imwrite(str(OUTPUT_DIR / "masked" / f"{prefix}_masked.png"),
                    overlay_masks(image_rgb, masks_np))
        for i, mask in enumerate(masks_np):
            cv2.imwrite(str(OUTPUT_DIR / "masks" / f"{prefix}_mask_{i:02d}.png"),
                        (mask * 255).astype(np.uint8))
    else:
        cv2.imwrite(str(OUTPUT_DIR / "masked" / f"{prefix}_masked.png"),
                    cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


# ── SAM3 ──────────────────────────────────────────────────────────────────────
print("[INFO] SAM3 모델 로딩 중...")
sam3_model = build_sam3_image_model(
    device=DEVICE, load_from_HF=False, checkpoint_path=CKPT_PATH
)

processor = Sam3Processor(sam3_model, device=DEVICE)



# ── 씬 로드 ──────────────────────────────────────────────────────────────────
omni.usd.get_context().open_stage(USD_PATH)
while omni.usd.get_context().get_stage_loading_status()[2]:
    simulation_app.update()
print(f"[INFO] Stage loaded: {USD_PATH}")

for _ in range(5):
    simulation_app.update()


# ── 카메라 (top-down) ─────────────────────────────────────────────────────────
rtx_camera = RtxCamera(
    CAMERA_PATH,
    translations=np.array([[0.0, 0.0, 5.0]]),
    orientations=TOP_DOWN_ORI,
)

viewport = get_active_viewport()
if viewport:
    viewport.camera_path = CAMERA_PATH

# TiledCameraSensor 대신 단일 CameraSensor 사용
sensor = CameraSensor(rtx_camera, resolution=RESOLUTION, annotators=["rgb"])
print(f"[INFO] CameraSensor | resolution: {sensor.resolution}")
print(f"[INFO] 저장 경로: {OUTPUT_DIR}")


# ── 시뮬레이션 루프 ──────────────────────────────────────────────────────────
timeline = omni.timeline.get_timeline_interface()
timeline.play()

for _ in range(10):
    simulation_app.update()

step = 0
while simulation_app.is_running():
    simulation_app.update()

    rgb_warp, _ = sensor.get_data("rgb")   # wp.array (H, W, 3) uint8
    if rgb_warp is None:
        step += 1
        continue

    # wp.array → CUDA torch.Tensor (zero-copy)
    rgb_tensor: torch.Tensor = wp.to_torch(rgb_warp)  # (H, W, 3) uint8 CUDA

    # Sam3Processor는 torchvision 규약인 CHW (3, H, W) 형식을 기대한다.
    # 💡 수정 포인트: 여기서 .to(torch.bfloat16)을 빼고 contiguous()까지만 유지해.
    rgb_chw = rgb_tensor.permute(2, 0, 1).contiguous()  

    # 💡 수정 포인트: 모델에 데이터가 들어가고 추론하는 과정을 autocast로 묶어줌
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(rgb_chw)
        state = processor.set_text_prompt(TEXT_PROMPT, state)

    masks_tensor = state.get("masks")   # (N, 1, H, W) bool | None

    if masks_tensor is not None and masks_tensor.shape[0] > 0:
        masks_np = masks_tensor[:, 0].cpu().numpy().astype(bool)  # (N, H, W)
        print(f"[Step {step:06d}] {masks_np.shape[0]} 개 마스크")
    else:
        masks_np = None
        print(f"[Step {step:06d}] 마스크 없음")

    save_step(step, rgb_tensor.cpu().numpy(), masks_np)  # HWC numpy (H,W,3)
    step += 1

timeline.stop()
simulation_app.close()
