import cv2
import json
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import sys
import os
import subprocess
import argparse
from collections import Counter

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
        
        print("Instructions:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'c' to change confidence threshold")
        print("   - Press 'r' to show object count report")
        print("")
        
        frame_count = 0
        total_object_counts = Counter()  # Count tổng từ khi bắt đầu
        session_detections = []  # Lưu history detections
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from camera!")
                break
            
            frame_count += 1
            
            # Chạy detection
            results = model(frame, conf=confidence, verbose=False)
            
            # Count objects by class trong frame hiện tại
            current_frame_objects = Counter()
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Count objects trong frame hiện tại
                    current_frame_objects[class_name] += 1
                    
                    # Update total count (accumulated)
                    total_object_counts[class_name] += 1
                    
                    # Save detection info
                    session_detections.append({
                        'frame': frame_count,
                        'class': class_name,
                        'confidence': conf,
                        'timestamp': frame_count  # Simple timestamp
                    })
            
            # Vẽ kết quả lên frame
            annotated_frame = results[0].plot()
            
            # Hiển thị thông tin frame
            total_current = sum(current_frame_objects.values())
            info_text = f"Frame: {frame_count} | Conf: {confidence:.1f} | Objects: {total_current}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Hiển thị count theo class trong frame hiện tại
            if current_frame_objects:
                y_offset = 60
                cv2.putText(annotated_frame, "Current Frame:", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                for i, (class_name, count) in enumerate(current_frame_objects.items()):
                    if i >= 5:  # Chỉ hiển thị max 5 classes để không làm đầy màn hình
                        break
                    y_offset += 25
                    count_text = f"  {class_name}: {count}"
                    cv2.putText(annotated_frame, count_text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                
                if len(current_frame_objects) > 5:
                    y_offset += 25
                    cv2.putText(annotated_frame, f"  ... +{len(current_frame_objects)-5} more", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Hiển thị các phím tắt
            controls_text = "Press: 'q'=quit, 's'=save, 'c'=confidence, 'r'=report"
            cv2.putText(annotated_frame, controls_text, (10, annotated_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # Hiển thị frame
            cv2.imshow("Real-time Object Detection with Counter (YOLOv8)", annotated_frame)
            
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
                print(f"Confidence changed to: {confidence}")
            elif key == ord('r'):
                # Show object count report
                print("="*60)
                print("OBJECT DETECTION REPORT")
                print("="*60)
                print(f"Total frames processed: {frame_count}")
                print(f"Current confidence threshold: {confidence}")
                
                if total_object_counts:
                    print(f"Total objects detected (accumulated):")
                    sorted_counts = sorted(total_object_counts.items(), key=lambda x: x[1], reverse=True)
                    for class_name, count in sorted_counts:
                        percentage = (count / sum(total_object_counts.values())) * 100
                        print(f"   {class_name}: {count} ({percentage:.1f}%)")
                    
                    print(f"Summary:")
                    print(f"   Grand total: {sum(total_object_counts.values())} detections")
                    print(f"   Unique classes: {len(total_object_counts)}")
                    print(f"   Average per frame: {sum(total_object_counts.values())/frame_count:.1f}")
                else:
                    print("No objects detected yet")
                
                # Store object
                df_object_count = {}
                if current_frame_objects:
                    print(f"Current frame objects:")
                    for class_name, count in current_frame_objects.items():
                        print(f"   {class_name}: {count}")
                        df_object_count[class_name] = count
                
                print("="*60 + "\n")
                print("res:\n",df_object_count)
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # tạo chuỗi thời gian hợp lệ
                    with open(f'./data/object_count_{timestamp}.json', 'w') as f:
                        json.dump(df_object_count, f, indent=4, ensure_ascii=False)  # ✅ Ghi dict vào file JSON

                except:
                    pass
        
        # Cleanup và final summary
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final summary
        print("\n" + "🎉 FINAL SESSION SUMMARY")
        print("="*70)
        print(f"Total frames processed: {frame_count}")
        print(f"Final confidence threshold: {confidence}")
        
        if total_object_counts:
            print(f"Objects detected throughout session:")
            sorted_counts = sorted(total_object_counts.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_counts:
                percentage = (count / sum(total_object_counts.values())) * 100
                print(f"   {class_name}: {count} detections ({percentage:.1f}%)")
            
            print(f"Final Statistics:")
            print(f"   Grand total: {sum(total_object_counts.values())} object detections")
            print(f"   Unique object types: {len(total_object_counts)}")
            print(f"   Average objects per frame: {sum(total_object_counts.values())/frame_count:.2f}")
            print(f"   Most detected: {sorted_counts[0][0]} ({sorted_counts[0][1]} times)")
        else:
            print("No objects were detected during this session")
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
    
    try:
        subprocess.run([
            'python', 'statistic_object.py', 
        ], cwd=os.getcwd(), check=True, capture_output=True)
        print("✅ statistic_object.py executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Statistic object failed: {e}")
    except Exception as e:
        print(f"Error launching: {e}")
    
    if success:
        print("Real-time detection completed successfully!")
        sys.exit(0)
    else:
        print("Real-time detection failed!")
        sys.exit(1)
        

if __name__ == "__main__":
    main()    