# import Genesis.genesis as gs
import genesis as gs
import torch
import os
import numpy as np
import cv2
import argparse
import random
from collections import defaultdict
import json

# Base directory (directory of this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. argparse 설정
parser = argparse.ArgumentParser(
    description="Isaac Sim Dynamic Stabilization Generator"
)
parser.add_argument("--target_object", type=str, required=True, help="Target USD name")
args, unknown = parser.parse_known_args()


PEN = {
    f"pen_{i}": os.path.join(
        BASE_DIR, "asset", "USD", "pen", f"Collected_pen_{i}", f"pen_{i}.usdc"
    )
    for i in range(1, 5)
}
ERASER = {
    f"eraser_{i}": os.path.join(
        BASE_DIR, "asset", "USD", "eraser", f"Collected_eraser_{i}", f"eraser_{i}.usdc"
    )
    for i in range(1, 5)
}
BOOK = {
    f"book_{i}": os.path.join(
        BASE_DIR, "asset", "USD", "book", f"Collected_book_{i}", f"book_{i}.usdc"
    )
    for i in range(1, 5)
}
NOTEBOOK = {
    f"notebook_{i}": os.path.join(
        BASE_DIR, "asset", "USD", "notebook", f"Collected_notebook_{i}", f"notebook_{i}.usdc"
    )
    for i in range(1, 5)
}

SEGMENTATION_IDX = {"pen_1": 3, "pen_2": 4, "pen_3": 5, "pen_4": 6, "eraser_1": 7, "eraser_3": 8, "eraser_4": 9, "book_1": 10, "book_2": 11, "book_3": 12, "book_4": 13, "notebook_1": 14, "notebook_2": 15, "notebook_3": 16, "notebook_4": 17}


# "pen_1": 531, "pen_2": 518, "pen_3": 589, "pen_4": 651, "eraser_1": 646, "eraser_2": 581, "eraser_3": 1385, "eraser_4": 445, "book_1": 14926, "book_2": 10030, "book_3": 26364, "book_4": 14352, "notebook_1": 3833, "notebook_2": 15892, "notebook_3": 1672, "notebook_4": 3933

class Segmentation_Pixel_Test:

    def __init__(self):
        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(
                dt=1e-2,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=1e-3,
                constraint_solver=gs.constraint_solver.Newton,
                tolerance=1e-7,
                iterations=300,
                ls_iterations=300,
                max_collision_pairs=300,
        
        ),
            viewer_options=gs.options.ViewerOptions(
                res=(1920, 1080),
                camera_pos=(8.5, 0.0, 4.5),
                camera_lookat=(3.0, 0.0, 0.5),
                camera_fov=50,
            ),
            renderer=gs.renderers.RayTracer(  # type: ignore
            env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path="textures/indoor_bright.png",
                    ),
                ),
                env_radius=15.0,
                env_euler=(0, 0, 180),
                lights=[
                    {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
                ],
            ),
        )
        self.items = {}
        self.item_types = {}
        self.segmentation_idx = {}
        self.wait_positions = {}
        self.last_orientations = {}  # 이전 프레임의 회전값 저장용
        self._scene_initialize()

    def _scene_initialize(self):
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)),
            surface=gs.surfaces.Aluminium(ior=10.0),
        )

        self.workspace = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file=os.path.join(BASE_DIR, "asset", "USD", "drawer.obj"),
                scale=1.0,
                pos=(0.0, 0.0, 0.02),
                euler=(0.0, 0.0, 0.0),
            ),
        )

        self.segmentation_idx["workspace"] = self.workspace.idx + 1

        def load_items(item_dict, type_name, y_offset):
            for i, (key, value) in enumerate(item_dict.items()):
                try:
                    wait_pos = np.array([0.8, y_offset, 0.1 * i])  # 서랍 밖 대기 위치
                    item_rigid_prim = self.scene.add_entity(
                        material=gs.materials.Rigid(rho=300),
                        morph=gs.morphs.Mesh(
                            file=value,
                            scale=1.0,
                            pos=wait_pos,
                            euler=(0.0, 0.0, 0.0),
                            convexify=True,
                        ),
                    )
                    self.items[key] = item_rigid_prim
                    self.segmentation_idx[key] = item_rigid_prim.idx + 1
                    self.item_types[key] = type_name
                    self.wait_positions[key] = wait_pos
                    # print(self.scene.segmentation_idx_dict)
                except Exception as e:
                    print(f"Failed to load {key}: {e}")

        load_items(PEN, "pen", 0.0)
        load_items(ERASER, "eraser", 0.2)
        load_items(BOOK, "book", 0.45)
        load_items(NOTEBOOK, "notebook", 0.75)

        self.cam_0 = self.scene.add_camera(
            res=(640, 480),
            pos=(0.0, 0.0, 0.8),
            lookat=(0.0, 0.0, 0.5),
            fov=60,
            GUI=False,
        )

        self.scene.build()

        self.scene.reset()

    def check_stabilization(self, threshold=1e-4):
        """
        [추가] 서랍 안에 있는 모든 물체의 회전 변화량을 체크하여 안정화 여부 판단
        """
        all_stable = True
        for key, obj in self.items.items():
            pos = obj.get_pos()
            quat = obj.get_quat()

            # 서랍 밖 대기 영역에 있는 물체는 체크 제외 (x > 0.5)
            if pos[0] > 0.5:
                continue

            if key in self.last_orientations:
                # 쿼터니언 내적을 통한 회전 차이 계산 (1.0에 가까울수록 변화 없음)
                dot_product = torch.abs(torch.dot(quat, self.last_orientations[key]))
                diff = 1.0 - dot_product

                # 속도와 회전 변화량이 모두 낮아야 안정화로 판단
                if diff > threshold:
                    all_stable = False

            # 현재 회전값 업데이트
            self.last_orientations[key] = quat

        return all_stable

    def wait_for_stabilization(self, max_steps=500):
        """안정화될 때까지 대기 (최대 max_steps)"""
        for _ in range(max_steps):
            self.scene.step()
            if self.check_stabilization():
                break

    def run(self):
        target_name = args.target_object
        print(f"--- Target {target_name} ---")
        # 1. Target 배치 및 안정화
        target_obj: gs.entities.RigidEntity = self.items[target_name]
        target_pos = np.array(
            [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.2]
        )
        target_pos = torch.tensor(target_pos, device="cuda:0")
        target_obj.set_pos(target_pos)
        self.wait_for_stabilization(max_steps=120)

        for _ in range(100):
            self.scene.step()

        rgb, depth_image, seg_image, _ = self.cam_0.render(
            rgb=True, depth=True, segmentation=True
        )
        # Count target pixels if segmentation index mapping available
        def count_target_pixels_in_image(seg_img, target, seg_idx_map, multiplier=15):
            """Return count of pixels belonging to `target` in segmentation image.

            seg_img: numpy array (H,W) or (H,W,C)
            target: target name (key in seg_idx_map) or integer index
            seg_idx_map: dict mapping names to segmentation idx (integers)
            multiplier: integer factor applied when saving segmentation images
            """
            if isinstance(target, str):
                if target not in seg_idx_map:
                    raise ValueError(f"Unknown target name: {target}")
                idx = int(seg_idx_map[target])
            else:
                idx = int(target)

            pixel_value = idx * multiplier

            # If multi-channel, convert to single channel assuming saved as grayscale*multiplier
            if seg_img.ndim == 3 and seg_img.shape[2] > 1:
                seg_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
            else:
                seg_gray = seg_img

            count = int(np.sum(seg_gray == pixel_value))
            total = seg_gray.size
            return count, total

        try:
            cnt, tot = count_target_pixels_in_image(seg_image, args.target_object, self.segmentation_idx, multiplier=1)
            print(f"Segmentation pixels for {args.target_object}: {cnt} / {tot} ({cnt/tot:.4f})")
        except Exception as e:
            print(f"Could not count target pixels: {e}")


def main():
    gs.init(precision="64", logging_level="info", backend=gs.gpu)
    generator = Segmentation_Pixel_Test()
    generator.run()

if __name__ == "__main__":  
    main()