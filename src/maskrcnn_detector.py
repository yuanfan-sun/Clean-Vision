# This file contains a reusable class for Mask R-CNN object detection.
# It encapsulates the model loading and prediction logic so it can be easily
# used by other scripts, like a Gradio web app.

import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

class MaskRCNNDetector:
    """
    A class to handle Mask R-CNN object detection.
    """
    def __init__(self):
        """
        Initializes the model and configuration.
        """
        self.cfg = get_cfg()
        # Load a pre-trained model configuration
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
        
        # Load the pre-trained model weights
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        # Set the device
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"MaskRCNN using device: {self.cfg.MODEL.DEVICE}")
        
        # Create the predictor
        self.predictor = DefaultPredictor(self.cfg)
        
        # Get metadata for class names
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

    def run_detection(self, image_bgr):
        """
        Runs inference on a given image.
        
        Args:
            image_bgr: The input image as a NumPy array in BGR format.
            
        Returns:
            The Detectron2 Instances object containing the predictions.
        """
        outputs = self.predictor(image_bgr)
        return outputs["instances"].to("cpu")

    def get_unique_class_names(self, instances):
        """
        Gets a unique, sorted list of class names from the detected instances.
        
        Args:
            instances: The Detectron2 Instances object.
            
        Returns:
            A sorted list of unique class names.
        """
        detected_classes = set()
        for i in range(len(instances)):
            class_id = instances.pred_classes[i].item()
            class_name = self.metadata.get("thing_classes", [])[class_id]
            detected_classes.add(class_name)
        return sorted(list(detected_classes))

    def visualize_instances(self, image_bgr, instances, classes_to_show=None):
        """
        Draws annotations for a given set of instances on an image.
        
        Args:
            image_bgr: The original image as a NumPy array in BGR format.
            instances: The Detectron2 Instances object to draw.
            classes_to_show (list, optional): A list of class names to filter for. 
                                              If None, all instances are drawn.
                                              
        Returns:
            The annotated image as a NumPy array in BGR format.
        """
        instances_to_draw = instances
        if classes_to_show:
            all_class_names = self.metadata.get("thing_classes", [])
            keep_indices = {i for i, name in enumerate(all_class_names) if name in classes_to_show}
            
            if keep_indices:
                keep_mask = torch.tensor([c.item() in keep_indices for c in instances.pred_classes])
                instances_to_draw = instances[keep_mask]
            else:
                # If no classes match, draw nothing
                instances_to_draw = instances[torch.zeros(len(instances), dtype=torch.bool)]

        # Create a new visualizer
        v = Visualizer(image_bgr[:, :, ::-1], self.metadata, scale=1.2)
        out = v.draw_instance_predictions(instances_to_draw)
        return out.get_image()[:, :, ::-1]
