import time
import cv2
import numpy as np
import base64
from models import get_pytorch_detector

# Global detector instance
detector = None

def load_pytorch_model():
    """Load PyTorch YOLO model at startup"""
    global detector
    try:
        detector = get_pytorch_detector()
        print("PyTorch model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return False

def process_image_with_pytorch(image_array, confidence_threshold=0.6):
    """
    Process image using PyTorch YOLO model
    
    Args:
        image_array: numpy array from uploaded image
        confidence_threshold: minimum confidence score
        
    Returns:
        dict: processing results
    """
    try:
        start_time = time.time()
        
        # Use global detector or create new one
        global detector
        if detector is None:
            detector = get_pytorch_detector()
        
        # Detect objects
        detections, inference_time = detector.detect_objects(
            image_array, 
            confidence_threshold=confidence_threshold
        )
        
        # Draw bounding boxes
        result_image = detector.draw_detections(
            image_array, 
            detections, 
            confidence_threshold=confidence_threshold
        )
        
        # Convert result to base64
        _, buffer = cv2.imencode('.png', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        total_time = time.time() - start_time
        
        # Format response
        return {
            'status': 'success',
            'result_image': f"data:image/png;base64,{result_base64}",
            'objects_count': len(detections),
            'processing_time': round(total_time, 3),
            'inference_time': round(inference_time, 3),
            'detections': [
                {
                    'class_name': det['class_name'],
                    'confidence': round(det['confidence'], 3),
                    'bbox': det['bbox']
                }
                for det in detections
            ]
        }
        
    except Exception as e:
        print(f"Error in process_image_with_pytorch: {e}")
        return {
            'status': 'error',
            'message': f"Processing failed: {str(e)}"
        }