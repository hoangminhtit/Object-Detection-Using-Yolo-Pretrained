import cv2
import numpy as np
from ultralytics import YOLO
import sys
import argparse

def run_realtime_detection(confidence=0.6):
    """
    Chạy real-time detection với webcam
    
    Args:
        confidence: ngưỡng confidence (default: 0.6)
    """
    
    try:
        # Load YOLO model
        model = YOLO("yolov8n.pt")  # Model nhẹ cho real-time
        
        # Mở camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open camera!")
            return False
            
        # Thiết lập camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("📝 Instructions:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'c' to change confidence threshold")
        print("")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from camera!")
                break
            
            frame_count += 1
            
            # Chạy detection
            results = model(frame, conf=confidence, verbose=False)
            
            # Vẽ kết quả lên frame
            annotated_frame = results[0].plot()
            
            # Hiển thị thông tin
            info_text = f"Frame: {frame_count} | Confidence: {confidence:.1f} | Objects: {len(results[0].boxes) if results[0].boxes is not None else 0}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị các phím tắt
            controls_text = "Press: 'q'=quit, 's'=save, 'c'=change confidence"
            cv2.putText(annotated_frame, controls_text, (10, annotated_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Hiển thị frame
            cv2.imshow("Real-time Object Detection (YOLOv8)", annotated_frame)
            
            # Xử lý phím nhấn
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quit requested by user")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"realtime_capture_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved frame: {filename}")
            elif key == ord('c'):
                # Change confidence (cycling through common values)
                conf_values = [0.3, 0.5, 0.6, 0.7, 0.8]
                current_idx = conf_values.index(confidence) if confidence in conf_values else 2
                next_idx = (current_idx + 1) % len(conf_values)
                confidence = conf_values[next_idx]
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection stopped successfully!")
        return True
        
    except Exception as e:
        print(f"Error in real-time detection: {e}")
        return False

def main():
    """Main function với argument parsing"""
    parser = argparse.ArgumentParser(description='Real-time Object Detection with YOLO')
    parser.add_argument('--confidence', '-c', type=float, default=0.6,
                        help='Confidence threshold (0.1-0.9, default: 0.6)')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without display (for debugging)')
    
    args = parser.parse_args()
    
    # Validate confidence
    confidence = max(0.1, min(0.9, args.confidence))
    
    if args.no_display:
        print("No display mode - only processing (for testing)")
        return True
    
    # Chạy real-time detection
    success = run_realtime_detection(confidence)
    
    if success:
        print("Real-time detection completed successfully!")
        sys.exit(0)
    else:
        print("Real-time detection failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()