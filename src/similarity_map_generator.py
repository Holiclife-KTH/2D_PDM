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
args, unknown = parser.parse_known_args()

# Base directory (directory of this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SIMILARITY_MAP = {
    "pen": {"pen": 0.8, "eraser": 0.5, "book": 0.2, "notebook": 0.2},
    "eraser": {"pen": 0.5, "eraser": 0.8, "book": 0.2, "notebook": 0.2},
    "book": {"pen": 0.2, "eraser": 0.2, "book": 0.8, "notebook": 0.5},
    "notebook": {"pen": 0.2, "eraser": 0.2, "book": 0.5, "notebook": 0.8},
}

ITEM_CATEGORIES = {
    "pen_1": "pen",
    "pen_2": "pen",
    "pen_3": "pen",
    "pen_4": "pen",
    "eraser_1": "eraser",
    "eraser_2": "eraser",
    "eraser_3": "eraser",
    "eraser_4": "eraser",
    "book_1": "book",
    "book_2": "book",
    "book_3": "book",
    "book_4": "book",
    "notebook_1": "notebook",
    "notebook_2": "notebook",
    "notebook_3": "notebook",
    "notebook_4": "notebook",
}


class Similarity_Map_Generator:
    def __init__(self):
        self.similarity_map = SIMILARITY_MAP
        self.target_name = args.target_name
        self.dataset_folder_path = os.path.join(BASE_DIR, "output", self.target_name)
        if not os.path.exists(self.dataset_folder_path):
            raise Exception(
                f"Dataset folder does not exist: {self.dataset_folder_path}"
            )

        with open(
            os.path.join(self.dataset_folder_path, "seg", "segmentation_idx.json"),
            "r",
        ) as f:
            self.segmentation_idx = json.load(f)

    def get_segmentation_images(self):
        """Get list of all segmentation images in the folder"""
        seg_images = []
        seg_dir = os.path.join(self.dataset_folder_path, "seg")
        for filename in sorted(os.listdir(seg_dir)):
            if filename.endswith(".png"):
                seg_images.append(os.path.join(seg_dir, filename))
        return seg_images

    def check_target_pixels(self, seg_image_path, target_idx):
        """Check if target_idx exists in segmentation image"""
        seg_image = cv2.imread(seg_image_path, cv2.IMREAD_UNCHANGED)

        # Check if target_idx exists in the image
        has_target = np.any(seg_image == target_idx)

        # Count pixels with target_idx
        pixel_count = np.sum(seg_image == target_idx)

        # Get all unique pixel values in the image
        unique_values = np.unique(seg_image)

        return {
            "has_target": has_target,
            "pixel_count": pixel_count,
            "unique_values": unique_values.tolist(),
        }

    def get_similarity(self, item_a, item_b):

        target_idx = self.segmentation_idx.get(item_a)

        seg_images = self.get_segmentation_images()

        # Check target pixels in each segmentation image
        for seg_img_path in seg_images:
            result = self.check_target_pixels(seg_img_path, target_idx)
            print(f"\n{os.path.basename(seg_img_path)}:")
            print(f"  Has target_idx {target_idx}: {result['has_target']}")
            print(f"  Target pixel count: {result['pixel_count']}")
            print(f"  Unique pixel values: {result['unique_values']}")

        return self.similarity_map.get(item_a, {}).get(item_b, 0.0)

    def create_similarity_segmentation(self, target_item):
        """
        Create similarity-based segmentation images where pixel values represent similarity to target

        Args:
            target_item: Target item name (e.g., 'notebook_4')
        """
        # Get target category and index
        target_category = ITEM_CATEGORIES.get(target_item)
        if not target_category:
            raise ValueError(f"Target item {target_item} not found in ITEM_CATEGORIES")

        target_idx = self.segmentation_idx.get(target_item)
        if target_idx is None:
            raise ValueError(f"Target item {target_item} not found in segmentation_idx")

        # Create reverse mapping: idx -> item_name
        idx_to_item = {v: k for k, v in self.segmentation_idx.items()}

        seg_images = self.get_segmentation_images()

        for idx, seg_img_path in enumerate(seg_images):
            # Load original segmentation
            seg_image = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)

            # Create new similarity-based segmentation
            similarity_seg = np.zeros_like(seg_image, dtype=np.uint8)

            # Get unique pixel values
            unique_values = np.unique(seg_image)
            print(unique_values)

            print(f"\nProcessing {os.path.basename(seg_img_path)}:")

            for pixel_val in unique_values:
                if pixel_val == 15:  # Background
                    similarity_seg[seg_image == pixel_val] = 0
                    continue

                # Get item name for this pixel value
                item_name = idx_to_item.get(pixel_val / 15)

                if item_name is None:
                    # Unknown item, set to 0
                    similarity_seg[seg_image == pixel_val] = 0
                    continue

                # Check if this is the exact target instance
                if item_name == target_item:
                    new_val = 255
                    print(f"  {item_name} (target) -> {new_val}")
                else:
                    # Get category of this item
                    item_category = ITEM_CATEGORIES.get(item_name)

                    if item_category is None:
                        similarity_seg[seg_image == pixel_val] = 0
                        continue

                    # Get similarity score
                    similarity_score = self.similarity_map.get(target_category, {}).get(
                        item_category, 0.0
                    )
                    new_val = int(255 * similarity_score)
                    print(
                        f"  {item_name} ({item_category}) -> {new_val} (similarity: {similarity_score})"
                    )

                # Set pixels
                similarity_seg[seg_image == pixel_val] = new_val

            # Save similarity segmentation
            sim_map_dir = os.path.join(self.dataset_folder_path, "similarity_map")
            os.makedirs(sim_map_dir, exist_ok=True)

            output_path = os.path.join(sim_map_dir, f"{idx+1:03d}.png")
            cv2.imwrite(output_path, similarity_seg)
            print(f"Saved: {output_path}")


def main():
    similarity_map_generator = Similarity_Map_Generator()

    # Create similarity segmentation for target
    similarity_map_generator.create_similarity_segmentation(args.target_name)


if __name__ == "__main__":
    main()
