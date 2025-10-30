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
        """
        Combine detections from both models with intelligent decision making
        
        Decision Logic:
        1. Keep all COCO detections (80 classes)
        2. Keep custom detections (desk) that don't overlap significantly with COCO
        3. If overlap detected, prioritize based on confidence and class relevance
        4. Remove duplicate detections using NMS (Non-Maximum Suppression)
        """
        combined = {
            'original_detections': [],
            'custom_detections': [],
            'filtered_detections': [],  # Final smart-filtered results
            'total_objects': 0
        }
        
        # Process original model results (COCO)
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
                        'source': 'COCO',
                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    }
                    combined['original_detections'].append(detection)
        
        # Process custom model results (desk detection)
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
                        'source': 'Custom',
                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    }
                    combined['custom_detections'].append(detection)
        
        # Apply intelligent combination logic
        combined['filtered_detections'] = self._smart_combine_detections(
            combined['original_detections'], 
            combined['custom_detections']
        )
        
        combined['total_objects'] = len(combined['filtered_detections'])
        
        # Print detailed summary
        print(f"📊 Hybrid Detection Results:")
        print(f"   - COCO detections: {len(combined['original_detections'])}")
        print(f"   - Custom detections: {len(combined['custom_detections'])}")
        print(f"   - After smart filtering: {len(combined['filtered_detections'])}")
        print(f"   - Final total: {combined['total_objects']}")
        
        return combined

    def _smart_combine_detections(self, coco_detections, custom_detections):
        """
        Intelligently combine detections from both models
        
        Strategy:
        1. Always keep COCO detections (they are general purpose)
        2. Add custom detections that don't significantly overlap with COCO
        3. For overlapping regions, decide based on confidence and class relevance
        4. Apply Non-Maximum Suppression to remove duplicates
        """
        final_detections = []
        
        # Always include all COCO detections first
        for detection in coco_detections:
            final_detections.append(detection)
        
        # Process custom detections and decide whether to include them
        for custom_det in custom_detections:
            should_include = True
            overlap_reason = None
            
            # Check for significant overlap with existing COCO detections
            for coco_det in coco_detections:
                iou = self._calculate_iou(custom_det['bbox'], coco_det['bbox'])
                
                # Decision logic based on overlap
                if iou > 0.3:  # Significant overlap threshold
                    # Check if COCO detection is related to desks/tables
                    if coco_det['class'].lower() in ['desk', 'table', 'dining table', 'chair']:
                        # Related classes - decide based on confidence
                        if custom_det['confidence'] > coco_det['confidence'] + 0.1:
                            # Custom model is more confident - keep custom, mark COCO for removal
                            overlap_reason = f"Higher confidence vs {coco_det['class']} (IoU: {iou:.2f})"
                            # Remove the overlapping COCO detection
                            if coco_det in final_detections:
                                final_detections.remove(coco_det)
                        else:
                            # COCO is more confident - skip custom
                            should_include = False
                            overlap_reason = f"Lower confidence vs {coco_det['class']} (IoU: {iou:.2f})"
                            break
                    else:
                        # Different class types - keep both if confidence is reasonable
                        if custom_det['confidence'] < 0.5:
                            should_include = False
                            overlap_reason = f"Low confidence near {coco_det['class']} (IoU: {iou:.2f})"
                            break
            
            # Add custom detection if it passed all checks
            if should_include:
                final_detections.append(custom_det)
                if overlap_reason:
                    print(f"Included custom 'desk': {overlap_reason}")
            else:
                print(f"Filtered custom 'desk': {overlap_reason}")
        
        # Apply final NMS to remove any remaining duplicates
        final_detections = self._apply_nms(final_detections, iou_threshold=0.4)
        
        return final_detections

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # box format: [x1, y1, x2, y2]
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Calculate intersection area
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _apply_nms(self, detections, iou_threshold=0.4):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        while detections:
            # Take the detection with highest confidence
            current = detections.pop(0)
            final_detections.append(current)
            
            # Remove detections that significantly overlap with current
            remaining = []
            for det in detections:
                iou = self._calculate_iou(current['bbox'], det['bbox'])
                # Only remove if same class and high overlap
                if current['class'] == det['class'] and iou > iou_threshold:
                    print(f"   🔄 NMS removed duplicate {det['class']} (IoU: {iou:.2f})")
                else:
                    remaining.append(det)
            
            detections = remaining
        
        return final_detections

    def _format_detections_for_flask(self, combined_detections):
        """Convert combined detections to format expected by Flask app"""
        detections_list = []
        
        # Use filtered detections (smart combination result)
        for det in combined_detections['filtered_detections']:
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
        
        # Draw filtered detections (smart combination result)
        for det in detections['filtered_detections']:
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            color = colors.get(det['source'], (0, 255, 255))  # Default yellow
            
            # Create detailed label with source info
            label = f"{det['class']} {det['confidence']:.2f}"
            source_label = f"[{det['source']}]"
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw main label
            cv2.putText(result_image, label, (x1, y1-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw source label
            cv2.putText(result_image, source_label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add summary info at top
        summary = f"Smart Hybrid: {len(detections['filtered_detections'])} objects"
        cv2.putText(result_image, summary, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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