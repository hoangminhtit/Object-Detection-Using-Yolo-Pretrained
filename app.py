from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
import base64
import os
import imageDetection
import time
import subprocess
import threading

app = Flask(__name__)
CORS(app)

# Processor functions for different options
def process_image_detection(image, confidence):
    """Process static image detection"""
    return imageDetection.process_image_with_pytorch(image, confidence)

def process_hybrid_detection(image, confidence):
    """Process hybrid COCO + Custom detection"""
    return imageDetection.process_image_with_hybrid(image, confidence)

# Options mapping
OPS = {
    'imageDetection': process_image_detection,
    'hybridDetection': process_hybrid_detection,
    # 'realTimeDetection': process_realtime_detection,
}

@app.route('/')
def index():
    """Main page with simple_detection.html template"""
    return render_template('index.html')

@app.route('/api/', methods=['POST'])
def api_object_detection():
    """
    API endpoint for object detection using PyTorch
    """
    try:
        option = request.form.get("option", "imageDetection")
        
        if option in ['imageDetection', 'hybridDetection']:
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
            
            # Process based on selected option
            if option in OPS:
                result = OPS[option](image, confidence)
                return jsonify(result)
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Unknown option: {option}'
                }), 400
        
        elif option == 'realTimeDetection':
            # Real-time detection can process uploaded images with RT model
            if option in OPS:
                result = OPS[option](image, confidence)
                return jsonify(result)
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Real-time detection processor not found'
                }), 500
        
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported option: {option}'
            }), 400
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({
            'status': 'error',
            'message': f"Server error: {str(e)}"
        }), 500

@app.route('/api/launch-realtime', methods=['POST'])
def api_launch_realtime():
    """Launch standalone real-time detection app"""
    try:
        # Get parameters from request
        data = request.get_json() or {}
        confidence = float(data.get('confidence', 0.6))
        confidence = max(0.1, min(0.9, confidence))
        use_hybrid = data.get('hybrid', False)
        
        # Launch real-time script in separate process
        def launch_script():
            try:
                cmd = [
                    'python', 'run_realtime.py', 
                    '--confidence', str(confidence)
                ]
                
                # Add hybrid flag if requested
                if use_hybrid:
                    cmd.append('--hybrid')
                
                subprocess.run(cmd, cwd=os.getcwd(), check=True)
            except subprocess.CalledProcessError as e:
                print(f"Real-time script failed: {e}")
            except Exception as e:
                print(f"Error launching real-time: {e}")
        
        # Start in background thread to not block web server
        thread = threading.Thread(target=launch_script)
        thread.daemon = True
        thread.start()
        
        model_type = "Hybrid (COCO + Custom)" if use_hybrid else "Standard YOLO"
        return jsonify({
            'status': 'success',
            'message': f'Real-time detection launched with {model_type} model, confidence {confidence}',
            'instruction': 'Check your desktop for the camera window. Press Q to quit, H to toggle hybrid mode.'
        })
        
    except Exception as e:
        print(f"Launch Real-time Error: {e}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to launch real-time detection: {str(e)}"
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'image_model_loaded': imageDetection.detector is not None,
        'framework': 'PyTorch + Ultralytics YOLO',
        'available_options': list(OPS.keys()),
        'endpoints': {
            'image_detection': '/api/',
            'webcam_realtime': '/api/webcam',
            'webcam_start': '/api/webcam/start', 
            'webcam_stop': '/api/webcam/stop',
            'launch_realtime_app': '/api/launch-realtime'
        }
    })

if __name__ == '__main__':
    
    # Load models at startup
    image_model_loaded = imageDetection.load_pytorch_model()
    hybrid_model_loaded = imageDetection.load_hybrid_model()
    
    if image_model_loaded:
        print("Application is running")
        if hybrid_model_loaded:
            print("Both standard and hybrid models are available")
        else:
            print("Standard model only - hybrid detection may not work")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load models. Please check dependencies.")
