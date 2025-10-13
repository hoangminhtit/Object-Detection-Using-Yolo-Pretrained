from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
import base64
import os
from models import test_model_pytorch, get_pytorch_detector
import time

app = Flask(__name__)
CORS(app)

# option

OPS = {
    'imageDetection': get_pytorch_detector,
    'realTimeDetection': '',
}

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

@app.route('/')
def index():
    """Main page with simple_detection.html template"""
    return render_template('index.html')

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

@app.route('/api/', methods=['POST'])
def api_object_detection():
    """
    API endpoint for object detection using PyTorch
    """
    try:
        option = request.form.get("option", "imageDetection")
        
        if option == 'imageDetection':
            # Check if file is present
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'No file uploaded'
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                }), 400
            
            # Get confidence threshold
            confidence = float(request.form.get('confidence', 0.6))
            confidence = max(0.1, min(0.9, confidence))  # Clamp between 0.1-0.9
            
            # Read and validate image
            file_bytes = file.read()
            if len(file_bytes) == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'Empty file'
                }), 400
            
            # Convert to OpenCV format
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid image format'
                }), 400
            
            # Process with PyTorch
            result = process_image_with_pytorch(image, confidence)
        
        else :
            pass
        
        out = OPS[option](result)
        
        return jsonify(out)
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({
            'status': 'error',
            'message': f"Server error: {str(e)}"
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'framework': 'PyTorch + Ultralytics YOLO'
    })

if __name__ == '__main__':
    print("Starting Flask app with PyTorch YOLO...")
    
    # Load model at startup
    model_loaded = load_pytorch_model()
    
    if model_loaded:
        print("Ready to serve!")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Please check dependencies.")
        print("Run: pip install -r requirements.txt")