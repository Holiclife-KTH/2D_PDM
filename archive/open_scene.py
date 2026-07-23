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


# ── Config ───────────────────────────────────────────────────────────────────
USD_PATH    = "omniverse://192.168.0.13/Users/th_ws/test_scene.usd"
CAMERA_PATH = "/World/Camera"
RESOLUTION  = (480, 640)        # (H, W)
CKPT_PATH   = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/sam3/weight/sam3.pt"
TEXT_PROMPT = "segment the objects in the scene"
MASK_ALPHA  = 0.5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR  = Path("/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/output/target")

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


def save_step(step: int, image_rgb: np.ndarray) -> None:
    prefix = f"{step:06d}"
    cv2.imwrite(str(OUTPUT_DIR / "rgb"    / f"{prefix}_rgb.png"),
                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))



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
    translations=np.array([[0.0, 0.0, 1.3]]),
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

    # Sam3Processor는 torchvision 규약인 CHW (3, H, W) 형식을 기대

    save_step(step, rgb_tensor.cpu().numpy())  # HWC numpy (H,W,3)
    step += 1

timeline.stop()
simulation_app.close()
