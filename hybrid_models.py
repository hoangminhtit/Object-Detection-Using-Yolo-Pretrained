from ultralytics import YOLO
import os
import cv2
import numpy as np

class HybridDetector:
    """
    Hybrid detector sử dụng cả model gốc YOLOv8 (80 classes COCO) và model custom (desk)
    để detect tất cả các object types
    """
    
    def __init__(self, original_model_path="./model_weights/yolov8n.pt", custom_model_path='./model_weights/best.pt'):
        """
        Args:
            original_model_path: Path to original YOLOv8 model (80 COCO classes)
            custom_model_path: Path to custom trained model (desk class)
        """
        self.original_model = YOLO(original_model_path)
        self.custom_model = None
        
        if custom_model_path and os.path.exists(custom_model_path):
            self.custom_model = YOLO(custom_model_path)
            print(f"Loaded custom model: {custom_model_path}")
        else:
            print("Custom model not found, chỉ sử dụng model gốc")
            
        # COCO classes (80 classes)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
            'toothbrush'
        ]

    def detect_hybrid(self, image_input, conf=0.30, save=True, show=False):
        """
        Detect objects using both original and custom models
        
        Args:
            image_input: path to input image or numpy array
            conf: confidence threshold
            save: save result image
            show: show result window
            
        Returns:
            tuple: (result_image, detections_list, inference_time)
        """
        import time
        start_time = time.time()
        
        # Handle both file path and numpy array inputs
        if isinstance(image_input, str):
            # It's a file path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image_path = image_input
            original_image = cv2.imread(image_path)
        else:
            # It's a numpy array
            original_image = image_input.copy()
            # Save temporarily for YOLO processing
            temp_path = "temp_hybrid_input.jpg"
            cv2.imwrite(temp_path, original_image)
            image_path = temp_path
        
        # Detect với model gốc (COCO classes)
        original_results = self.original_model.predict(
            source=image_path,
            conf=conf,
            save=False,
            show=False,
            verbose=False
        )
        
        # Detect với custom model (desk)
        custom_results = None
        if self.custom_model:
            custom_results = self.custom_model.predict(
                source=image_path,
                conf=conf,
                save=False,
                show=False,
                verbose=False
            )
        
        # Combine results
        combined_detections = self._combine_results(original_results, custom_results)
        
        # Convert to format expected by Flask app
        detections_list = self._format_detections_for_flask(combined_detections)
        
        # Draw results on original image
        result_image = self._draw_combined_results_on_image(original_image, combined_detections)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Clean up temp file if created
        if not isinstance(image_input, str) and os.path.exists("temp_hybrid_input.jpg"):
            os.remove("temp_hybrid_input.jpg")
        
        # Optional save and show
        if save and isinstance(image_input, str):
            save_path = self._save_result(image_input, result_image)
            print(f"💾 Saved result: {save_path}")
            
        if show:
            cv2.imshow('Hybrid Detection Results', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result_image, detections_list, inference_time

    def _combine_results(self, original_results, custom_results):
        """Combine detections from both models"""
        combined = {
            'original_detections': [],
            'custom_detections': [],
            'total_objects': 0
        }
        
        # Process original model results
        if original_results and len(original_results) > 0:
            result = original_results[0]
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    detection = {
                        'class': self.coco_classes[class_id],
                        'confidence': confidence,
                        'bbox': bbox,
                        'source': 'COCO'
                    }
                    combined['original_detections'].append(detection)
        
        # Process custom model results
        if custom_results and len(custom_results) > 0:
            result = custom_results[0]
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    detection = {
                        'class': 'desk',
                        'confidence': confidence,
                        'bbox': bbox,
                        'source': 'Custom'
                    }
                    combined['custom_detections'].append(detection)
        
        combined['total_objects'] = len(combined['original_detections']) + len(combined['custom_detections'])
        
        # Print summary
        print(f"📊 Results:")
        print(f"   - COCO objects: {len(combined['original_detections'])}")
        print(f"   - Custom objects (desk): {len(combined['custom_detections'])}")
        print(f"   - Total: {combined['total_objects']}")
        
        return combined

    def _format_detections_for_flask(self, combined_detections):
        """Convert combined detections to format expected by Flask app"""
        detections_list = []
        
        # Add original detections
        for det in combined_detections['original_detections']:
            detections_list.append({
                'class_name': det['class'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'model_source': det['source']
            })
        
        # Add custom detections
        for det in combined_detections['custom_detections']:
            detections_list.append({
                'class_name': det['class'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'model_source': det['source']
            })
        
        return detections_list

    def _draw_combined_results_on_image(self, image, detections):
        """Draw all detections on numpy array image"""
        result_image = image.copy()
        
        # Colors for different sources
        colors = {
            'COCO': (0, 255, 0),    # Green for COCO objects
            'Custom': (255, 0, 0)   # Blue for custom objects
        }
        
        # Draw original detections
        for det in detections['original_detections']:
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            label = f"{det['class']} {det['confidence']:.2f}"
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), colors['COCO'], 2)
            cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['COCO'], 2)
        
        # Draw custom detections
        for det in detections['custom_detections']:
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            label = f"{det['class']} {det['confidence']:.2f}"
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), colors['Custom'], 2)
            cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['Custom'], 2)
        
        return result_image

    def _draw_combined_results(self, image_path, detections):
        """Draw all detections on image (for file path input)"""
        image = cv2.imread(image_path)
        return self._draw_combined_results_on_image(image, detections)

    # def _save_result(self, image_path, result_image):
    #     """Save result image"""
    #     # Create output directory
    #     output_dir = "runs/detect/hybrid"
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Find next available predict folder
    #     predict_num = 1
    #     while os.path.exists(f"{output_dir}/predict{predict_num}"):
    #         predict_num += 1
        
    #     output_path = f"{output_dir}/predict{predict_num}"
    #     os.makedirs(output_path, exist_ok=True)
        
    #     # Save image
    #     filename = os.path.basename(image_path)
    #     save_path = os.path.join(output_path, filename)
    #     cv2.imwrite(save_path, result_image)
        
    #     return save_path


def detect_image_hybrid(original_model="./model_weights/yolov8n.pt", custom_model='./model_weights/best.pt', image_path=None, conf=0.30, show=False):
    """
    Convenience function for hybrid detection
    """
    detector = HybridDetector(original_model, custom_model)
    result_image, detections_list, inference_time = detector.detect_hybrid(image_path, conf=conf, save=True, show=show)
    return detections_list


# # --------- TEST DEMO ---------
# if __name__ == "__main__":
#     # Paths
#     original_model = "yolov8n.pt"  # Model gốc COCO 80 classes
#     custom_model = r"D:\code_vs\detect\detect\runs\detect\train\weights\best.pt"  # Model custom (desk)
#     # image_path = r"D:\code_vs\Python\ai_subject\test_data\di-bo-sang-duong.jpg"  # Test image
#     image_path = r"D:\code_vs\Python\ai_subject\test_data\di-bo-pham-luat_0207092143.jpg"
    
#     # Detect
#     results = detect_image_hybrid(
#         original_model=original_model,
#         custom_model=custom_model, 
#         image_path=image_path,
#         conf=0.30,
#         show=True
#     )
    
#     print("\n🎯 Detailed Results:")
#     for det in results['original_detections']:
#         print(f"   COCO: {det['class']} ({det['confidence']:.2f})")
    
#     for det in results['custom_detections']:
#         print(f"   Custom: {det['class']} ({det['confidence']:.2f})")