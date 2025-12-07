"""
============================================================
DETECTION ROUTES - Face Detection Endpoints
============================================================
"""

from flask import Blueprint, request, jsonify
from PIL import Image
from datetime import datetime
import base64
import io

from ..model import face_model

detection_bp = Blueprint('detection', __name__)


@detection_bp.route('/api/detect', methods=['POST'])
def detect():
    """
    Detect faces from base64 encoded image
    
    Request body:
    {
        "image": "base64_encoded_image_string",
        "ignore_zones": [{"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3}]  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Get ignore zones (normalized coordinates 0-1)
        ignore_zones = data.get('ignore_zones', [])
        
        # Process
        results = face_model.process_image(image)
        
        # Filter out faces in ignore zones
        if ignore_zones:
            img_width, img_height = image.size
            filtered_results = []
            
            for result in results:
                bbox = result.get('bbox', {})
                # Get face center (normalized)
                face_center_x = (bbox.get('x1', 0) + bbox.get('x2', 0)) / 2 / img_width
                face_center_y = (bbox.get('y1', 0) + bbox.get('y2', 0)) / 2 / img_height
                
                # Check if face center is in any ignore zone
                in_ignore_zone = False
                for zone in ignore_zones:
                    if (zone['x1'] <= face_center_x <= zone['x2'] and
                        zone['y1'] <= face_center_y <= zone['y2']):
                        in_ignore_zone = True
                        break
                
                if not in_ignore_zone:
                    filtered_results.append(result)
            
            results = filtered_results
        
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "faces_detected": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@detection_bp.route('/api/detect/upload', methods=['POST'])
def detect_upload():
    """Detect faces from uploaded image file"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read image
        image = Image.open(file.stream).convert('RGB')
        
        # Process
        results = face_model.process_image(image)
        
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "faces_detected": len(results),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
