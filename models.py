import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from pathlib import Path
import random
import os
from tqdm import tqdm
import matplotlib.patches as mpatches
import torch
from ultralytics import YOLO
from PIL import Image
import io
import base64

# Set matplotlib backend for web app
plt.switch_backend('Agg')

class PyTorchObjectDetector:
    """
    PyTorch-based object detector using YOLOv8
    Thay thế cho GluonCV model_zoo
    """
    
    def __init__(self, model_name='./model_weights/yolov8n.pt'):
        """
        Initialize YOLO model
        model_name: 'yolov8n.pt' (nano), 'yolov8s.pt' (small), 'yolov8m.pt' (medium), 'yolov8l.pt' (large)
        """
        print(f"🔄 Loading PyTorch YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.class_names = self.model.names  # COCO class names
        print(f"✅ Model loaded successfully! Classes: {len(self.class_names)}")
    
    def detect_objects(self, image, confidence_threshold=0.6, image_size=640):
        """
        Detect objects in image using PyTorch YOLO
        
        Args:
            image: numpy array (BGR format from cv2)
            confidence_threshold: minimum confidence score
            image_size: input size for YOLO model
            
        Returns:
            detections: list of detection results
            processing_time: inference time in seconds
        """
        start_time = time.time()
        
        # Convert BGR to RGB for YOLO
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(rgb_image, imgsz=image_size, conf=confidence_threshold, verbose=False)
        
        processing_time = time.time() - start_time
        
        # Parse results
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # x, y, width, height
                    'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],    # x1, y1, x2, y2
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id]
                })
        
        return detections, processing_time
    
    def draw_detections(self, image, detections, confidence_threshold=0.6):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: numpy array (BGR format)
            detections: list of detection results
            confidence_threshold: minimum confidence to display
            
        Returns:
            annotated_image: image with bounding boxes drawn
        """
        annotated_image = image.copy()
        
        # Color palette for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (64, 64, 64), (192, 192, 192), (255, 165, 0)
        ]
        
        for detection in detections:
            if detection['confidence'] >= confidence_threshold:
                x1, y1, x2, y2 = detection['bbox_xyxy']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Choose color based on class_id
                color = colors[detection['class_id'] % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(annotated_image, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated_image, label, 
                          (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image

# Compatibility functions to replace GluonCV functions
def test_model_pytorch(image, confidence_threshold=0.6):
    """
    PyTorch version of test_model function
    Thay thế cho GluonCV test_model
    
    Args:
        image: numpy array (BGR format from cv2.imread)
        confidence_threshold: minimum confidence score
        
    Returns:
        result_image_base64: base64 encoded result image
        processing_time: inference time
        detections: list of detection results
    """
    # Initialize detector (will download model if not exists)
    detector = PyTorchObjectDetector('yolov8n.pt')  # Use nano model for speed
    
    # Detect objects
    detections, processing_time = detector.detect_objects(image, confidence_threshold)
    
    # Draw bounding boxes
    result_image = detector.draw_detections(image, detections, confidence_threshold)
    
    # Convert to base64 for web response
    _, buffer = cv2.imencode('.png', result_image)
    result_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{result_image_base64}", processing_time, detections

def quick_test_pytorch(image_path, confidence_threshold=0.6, save_result=True):
    """
    Quick test function for PyTorch model
    
    Args:
        image_path: path to test image
        confidence_threshold: minimum confidence
        save_result: whether to save result image
        
    Returns:
        detections: list of detected objects
        processing_time: inference time
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    print(f"Testing image: {image_path}")
    print(f"Image size: {image.shape}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Initialize detector
    detector = PyTorchObjectDetector('yolov8n.pt')
    
    # Detect objects
    detections, processing_time = detector.detect_objects(image, confidence_threshold)
    
    # Print detection details
    for i, det in enumerate(detections):
        print(f"   {i+1}. {det['class_name']}: {det['confidence']:.3f}")
    
    # Draw and save result
    if save_result and detections:
        result_image = detector.draw_detections(image, detections, confidence_threshold)
        result_path = f"result_pytorch_{Path(image_path).stem}.jpg"
        cv2.imwrite(result_path, result_image)    
    return detections, processing_time

# Global model instance (to avoid reloading)
pytorch_detector = None

def get_pytorch_detector():
    """Get global PyTorch detector instance"""
    global pytorch_detector
    if pytorch_detector is None:
        pytorch_detector = PyTorchObjectDetector('yolov8n.pt')
    return pytorch_detector