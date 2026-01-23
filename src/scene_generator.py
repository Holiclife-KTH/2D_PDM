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

# 1. argparse 설정
parser = argparse.ArgumentParser(
    description="Isaac Sim Dynamic Stabilization Generator"
)
parser.add_argument("--target_name", type=str, required=True, help="Target USD name")
parser.add_argument("--num_images", type=int, default=1, help="Number of scenes")
args, unknown = parser.parse_known_args()

PEN = {
    f"pen_{i}": f"/home/irol/workspace/2D_PDM/src/asset/USD/pen/Collected_pen_{i}/pen_{i}.usdc"
    for i in range(1, 5)
}
ERASER = {
    f"eraser_{i}": f"/home/irol/workspace/2D_PDM/src/asset/USD/eraser/Collected_eraser_{i}/eraser_{i}.usdc"
    for i in range(1, 5) if i != 2
}
BOOK = {
    f"book_{i}": f"/home/irol/workspace/2D_PDM/src/asset/USD/book/Collected_book_{i}/book_{i}.usdc"
    for i in range(1, 5)
}
NOTEBOOK = {
    f"notebook_{i}": f"/home/irol/workspace/2D_PDM/src/asset/USD/notebook/Collected_notebook_{i}/notebook_{i}.usdc"
    for i in range(1, 5)
}

SIMILARITY_MAP = {
    "pen": {"pen": 0.8, "eraser": 0.5, "book": 0.2, "notebook": 0.2},
    "eraser": {"pen": 0.5, "eraser": 0.8, "book": 0.2, "notebook": 0.2},
    "book": {"pen": 0.2, "eraser": 0.2, "book": 0.8, "notebook": 0.5},
    "notebook": {"pen": 0.2, "eraser": 0.2, "book": 0.5, "notebook": 0.8},
}

SCORE_TO_RADIUS = {0.8: 0.03, 0.5: 0.06, 0.2: 0.09}
WORKSPACE_BOUNDS = {"x": [-0.25, 0.25], "y": [-0.25, 0.25], "z": [0.2, 0.35]}


class Cluttered_Scene_Generator:
    path = f"/home/irol/workspace/2D_PDM/src/output/{args.target_name}"
    if not os.path.exists(path + "/rgb"):
        os.makedirs(path + "/rgb")
    if not os.path.exists(path + "/depth"):
        os.makedirs(path + "/depth")
    if not os.path.exists(path + "/seg"):
        os.makedirs(path + "/seg")

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
                iterations=200,
                ls_iterations=200,
                max_collision_pairs=200,
        
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
                    {
                        "pos": (0.0, 0.0, 10.0),
                        "radius": 20.0,
                        "color": (100.0, 100.0, 100.0),
                    },
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
                file="/home/irol/workspace/2D_PDM/src/asset/USD/drawer.obj",
                scale=1.0,
                pos=(0.0, 0.0, 0.05),
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

    def reset_items(self):
        for key, obj in self.items.items():
            obj.set_pos(self.wait_positions[key])
        self.last_orientations = {}  # 회전 기록 초기화
        for _ in range(20):
            self.scene.step()

    def generate_one_scene(self, idx, start_num=0):
        target_name = args.target_name
        print(f"--- Scene {idx+1}: Target {target_name} ---")
        # self.reset_items()

        if random.random() < 0.9:
            # 1. Target 배치 및 안정화
            target_obj: gs.entities.RigidEntity = self.items[target_name]
            target_pos = np.array(
                [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.2]
            )
            target_pos = torch.tensor(target_pos, device="cuda:0")
            target_obj.set_pos(target_pos)
            self.wait_for_stabilization(max_steps=120)

            # 2. 주변 물체 순차 투하
            target_type = self.item_types[target_name]
            score_groups = defaultdict(list)
            for k, v in self.item_types.items():
                if k == target_name:
                    continue
                score = SIMILARITY_MAP[target_type][v]
                score_groups[score].append(k)

            for score in sorted(score_groups.keys(), reverse=True):
                group = score_groups[score]
                random.shuffle(group)
                for item_key in group:
                    obj: gs.entities.RigidEntity = self.items[item_key]
                    radius = SCORE_TO_RADIUS.get(score, 0.2)

                    # 가우시안 대신 원형 범위 내 스폰 로직
                    r = radius * torch.sqrt(torch.rand(1))
                    theta = torch.rand(1) * 2 * torch.pi
                    cur_t_pos = target_obj.get_pos()
                    spawn_pos = torch.tensor(
                        [
                            torch.clip(cur_t_pos[0] + r * torch.cos(theta), -0.3, 0.3),
                            torch.clip(cur_t_pos[1] + r * torch.sin(theta), -0.3, 0.3),
                            random.uniform(0.15, 0.3),
                        ]
                    )

                    obj.set_pos(spawn_pos)
                    # 물체 하나 투하할 때마다 최소한의 물리 안정화
                    self.wait_for_stabilization(max_steps=60)
        else:
            # 10% case: spawn others first
            target_pos = np.array(
                [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.2]
            )
            target_type = self.item_types[target_name]
            score_groups = defaultdict(list)
            for k, v in self.item_types.items():
                if k == target_name:
                    continue
                score = SIMILARITY_MAP[target_type][v]
                score_groups[score].append(k)

            for score in sorted(score_groups.keys(), reverse=True):
                group = score_groups[score]
                random.shuffle(group)
                for item_key in group:
                    obj: gs.entities.RigidEntity = self.items[item_key]
                    radius = SCORE_TO_RADIUS.get(score, 0.2)

                    # 가우시안 대신 원형 범위 내 스폰 로직
                    r = radius * torch.sqrt(torch.rand(1))
                    theta = torch.rand(1) * 2 * torch.pi
                    # Spawn randomly in workspace bounds
                    spawn_pos = torch.tensor(
                        [
                            torch.clip(target_pos[0] + r * torch.cos(theta), -0.3, 0.3),
                            torch.clip(target_pos[1] + r * torch.sin(theta), -0.3, 0.3),
                            random.uniform(0.15, 0.3),
                        ]
                    )
                    obj.set_pos(spawn_pos)
                    self.wait_for_stabilization(max_steps=60)

            # Then place target
            target_obj: gs.entities.RigidEntity = self.items[target_name]
            
            target_pos = torch.tensor(target_pos, device="cuda:0")
            target_obj.set_pos(target_pos)
            self.wait_for_stabilization(max_steps=120)

        # 3. 모든 물체 투하 후 최종 안정화
        print("Final stabilizing...")
        self.wait_for_stabilization(max_steps=300)
        print(f"Scene {start_num + idx+1} Stabilized.")

        for _ in range(500):
            self.scene.step()

        rgb, depth_image, seg_image, _ = self.cam_0.render(
            rgb=True, depth=True, segmentation=True
        )

        cv2.imwrite(
            f"/home/irol/workspace/2D_PDM/src/output/{target_name}/rgb/{start_num + idx+1:03d}.png",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        # Depth 이미지는 .npy 포맷으로 저장하여 원본 데이터 보존
        np.save(
            f"/home/irol/workspace/2D_PDM/src/output/{target_name}/depth/{start_num + idx+1:03d}.npy",
            depth_image,
        )
        cv2.imwrite(
            f"/home/irol/workspace/2D_PDM/src/output/{target_name}/seg/{start_num + idx+1:03d}.png",
            seg_image * 15,
        )
        self.reset_items()

    def save_segmentation_idx(self):
        seg_json = json.dumps(self.segmentation_idx)

        with open(
            f"/home/irol/workspace/2D_PDM/src/output/{args.target_name}/seg/segmentation_idx.json",
            "w",
        ) as f:
            f.write(seg_json)

    def __del__(self):
        self.save_segmentation_idx()


def main():
    gs.init(precision="64", logging_level="info", backend=gs.gpu)

    generator = Cluttered_Scene_Generator()
    start_num = 100-args.num_images
    for i in range(args.num_images):
        generator.generate_one_scene(i, start_num)

    generator.save_segmentation_idx()


if __name__ == "__main__":
    main()
