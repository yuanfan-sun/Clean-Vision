import os
from pathlib import Path
import requests
from tqdm import tqdm

import cv2
import torch
import numpy as np

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

def download_file(url: str, dest_path: str):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.split('/')[-1],
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

class SAM_Segmenter:
    """
    A class dedicated to segmenting objects within bounding boxes using the Segment Anything Model (SAM).
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SAM_Segmenter using device: {self.device}")

        # --- SAM Setup ---
        self.sam_checkpoint_path = self._download_sam_checkpoint()
        self.sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint_path).to(self.device)
        self.sam_predictor = SamPredictor(self.sam)
        print("SAM model initialized.")

    def _download_sam_checkpoint(self):
        """Downloads the SAM ViT-H checkpoint if it doesn't exist."""
        checkpoint_dir = Path("weights")
        checkpoint_dir.mkdir(exist_ok=True)
        ckpt_path = checkpoint_dir / "sam_vit_h_4b8939.pth"
        if not ckpt_path.exists():
            print("Downloading SAM (ViT-H) checkpoint...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            download_file(url, str(ckpt_path))
            print("SAM checkpoint downloaded.")
        return str(ckpt_path)

    def segment_from_boxes(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray):
        """
        Generates segmentation masks for a given set of bounding boxes.
        """
        if boxes_xyxy.shape[0] == 0:
            return []

        self.sam_predictor.set_image(image_rgb)
        input_boxes = torch.from_numpy(boxes_xyxy).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=False,
        )

        masked_objects_rgb = []
        for i in range(masks.shape[0]):
            bool_mask = masks[i, 0, :, :].cpu().numpy()
            masked_img_rgb = np.zeros_like(image_rgb)
            masked_img_rgb[bool_mask] = image_rgb[bool_mask]
            masked_objects_rgb.append(masked_img_rgb)

        return masked_objects_rgb
