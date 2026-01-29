import genesis as gs
import torch
import os
import numpy as np
import cv2
import argparse
import random
from collections import defaultdict
import json
from genesis.utils.geom import quat_to_xyz, xyz_to_quat

# 1. argparse 설정
parser = argparse.ArgumentParser(
    description="Isaac Sim Dynamic Stabilization Generator"
)
parser.add_argument("--target_name", type=str, required=True, help="Target USD name")
args, unknown = parser.parse_known_args()


PEN = {
    f"pen_{i}": f"/home/irol/workspace/2D_PDM/src/asset/USD/pen/Collected_pen_{i}/pen_{i}.usdc"
    for i in range(1, 5)
}
ERASER = {
    f"eraser_{i}": f"/home/irol/workspace/2D_PDM/src/asset/USD/eraser/Collected_eraser_{i}/eraser_{i}.usdc"
    for i in range(1, 5)
}
BOOK = {
    f"book_{i}": f"/home/irol/workspace/2D_PDM/src/asset/USD/book/Collected_book_{i}/book_{i}.usdc"
    for i in range(1, 5)
}
NOTEBOOK = {
    f"notebook_{i}": f"/home/irol/workspace/2D_PDM/src/asset/USD/notebook/Collected_notebook_{i}/notebook_{i}.usdc"
    for i in range(1, 5)
}


class Occlusion_set:
    path = f"/home/irol/workspace/2D_PDM/src/output/{args.target_name}/target"
    if not os.path.exists(path + "/depth"):
        os.makedirs(path + "/depth")
    if not os.path.exists(path + "/rgb"):
        os.makedirs(path + "/rgb")
    if not os.path.exists(path + "/seg"):
        os.makedirs(path + "/seg")

    def __init__(self):
        self.scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(
                dt=0.001,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1920, 1080),
                camera_pos=(0.0, 0.0, 4.5),
                camera_lookat=(0.0, 0.0, 0.5),
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

        self._scene_initialize()

    def _scene_initialize(self):
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0)),
            surface=gs.surfaces.Aluminium(ior=10.0),
        )

        self.workspace = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="/home/irol/workspace/2D_PDM/src/asset/USD/drawer.obj", # change to own path
                scale=1.0,
                pos=(0.0, 0.0, 0.02),
                euler=(0.0, 0.0, 0.0),
            ),
        )

        target = args.target_name

        if target.startswith("pen"):
            file_path = PEN[target]
        elif target.startswith("eraser"):
            file_path = ERASER[target]
        elif target.startswith("book"):
            file_path = BOOK[target]
        elif target.startswith("notebook"):
            file_path = NOTEBOOK[target]
        else:
            raise ValueError(f"Unknown target name: {target}")

        try:
            init_pos = np.array([0.17, 0.17, 0.07])  # 서랍 밖 대기 위치
            self.target_prim: gs.entities.RigidEntity = self.scene.add_entity(
                material=gs.materials.Rigid(rho=300),
                morph=gs.morphs.Mesh(
                    file=file_path,
                    scale=1.0,
                    pos=init_pos,
                    euler=(0.0, 0.0, 0.0),
                    convexify=True,
                ),
            )
            # print(self.scene.segmentation_idx_dict)
        except Exception as e:
            print(f"Failed to load {target}: {e}")

        self.cam_0 = self.scene.add_camera(
            res=(640, 480),
            pos=(0.0, 0.0, 0.8),
            lookat=(0.0, 0.0, 0.5),
            fov=60,
            GUI=False,
        )

        self.scene.build()
        self.scene.reset()

    def run(self):
        count = 0
        x_min, x_max = -0.17, 0.17
        y_min, y_max = -0.17, 0.17
        z_max = 0.1
        step = 0.02

        # Start position
        current_x = x_max
        current_y = y_max
        z = 0.04  # Keep z constant

        # Rotation parameters
        yaw_step = np.pi / 6  # pi/6 radians (30 degrees)
        num_rotations = 12  # 2pi / (pi/6) = 12 steps for full rotation
        current_rotation = 0

        while True:
            if count % 2 == 0:
                # Calculate current yaw angle
                current_yaw = (current_rotation % num_rotations) * yaw_step

                # Set position and orientation
                next_pos = torch.tensor([current_x, current_y, z], device="cuda:0")
                next_euler = torch.tensor([0.0, 0.0, current_yaw], device="cuda:0")
                next_quat = xyz_to_quat(next_euler)
                print(
                    f"Position: ({self.target_prim.get_pos()}), Yaw: {current_yaw:.2f} rad ({np.degrees(current_yaw):.1f}°)"
                )
                rgb, depth_image, seg_image, _ = self.cam_0.render(
                    rgb=True, depth=True, segmentation=True
                )

                cv2.imwrite(
                    f"/home/irol/workspace/2D_PDM/src/output/{args.target_name}/target/rgb/{count//5}.png",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )
                np.save(
                    f"/home/irol/workspace/2D_PDM/src/output/{args.target_name}/target/depth/{count//5}.npy",
                    depth_image,
                )
                cv2.imwrite(
                    f"/home/irol/workspace/2D_PDM/src/output/{args.target_name}/target/seg/{count//5}.png",
                    seg_image * 15,
                )
                current_rotation += 1

                # After completing full rotation (2pi), move to next position
                if current_rotation % num_rotations == 0:
                    current_x -= step

                    # Check if out of x boundary
                    if current_x < x_min:
                        # Reset x to start position and move y
                        current_x = x_max
                        current_y -= step

                        # Check if out of y boundary (finished scanning)
                        if current_y < y_min:
                            z += 0.03
                            if z > z_max:
                                print("Finished scanning the entire workspace")
                                break
                            

            self.target_prim.set_pos(next_pos)
            self.target_prim.set_quat(next_quat)
            count += 1
            self.scene.step()


def main():
    gs.init(precision="32", logging_level="info", backend=gs.gpu)
    occlusion_set = Occlusion_set()
    occlusion_set.run()


if __name__ == "__main__":
    main()
