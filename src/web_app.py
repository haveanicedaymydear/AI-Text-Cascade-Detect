                      

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import json
import time
from optimized_dtd_v2 import UltraFastDTD
from typing import Dict, List
import logging

      
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

                  
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)          

           
dtd_model = None

def initialize_model():
    global dtd_model
    try:
        dtd_model = UltraFastDTD()
                      
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "optimized_dtd_model.pkl")
        
        if os.path.exists(model_path):
            dtd_model.load_model(model_path)
            logger.info("DTDModel loaded successfully")
        else:
            logger.warning("Model file not found. Please train the model first.")
            return False
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return False
    
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_text():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Please provide the text to be detected.'
            }), 400
        
        text = data['text'].strip()
        
        if len(text) < 10:
            return jsonify({
                'success': False,
                'error': 'The text is too short. Please provide at least 10 characters.'
            }), 400
        
        if len(text) > 10000:
            return jsonify({
                'success': False,
                'error': 'The text is too long. Please provide no more than 10000 characters.'
            }), 400
        
              
        if dtd_model is None:
            return jsonify({
                'success': False,
                'error': 'DTD model is not initialized.'
            }), 500
        
        start_time = time.time()
        result = dtd_model.predict_fast(text)
        total_time = (time.time() - start_time) * 1000
        
               
        response = {
            'success': True,
            'result': {
                'prediction': result['prediction'],
                'ai_probability': float(result['ai_probability']),
                'human_probability': float(1 - result['ai_probability']),
                'confidence': float(result['confidence']),
                'prediction_time_ms': float(total_time),
                'text_length': len(text),
                'word_count': len(text.split())
            },
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return jsonify({
            'success': False,
            'error': f'Error during detection: {str(e)}'
        }), 500

@app.route('/api/batch_detect', methods=['POST'])
def batch_detect():
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'Please provide the list of texts to be detected.'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'Texts must be provided as a list.'
            }), 400
        
        if len(texts) > 10:
            return jsonify({
                'success': False,
                'error': 'Batch detection supports up to 10 texts.'
            }), 400
        
              
        results = []
        total_start_time = time.time()
        
        for i, text in enumerate(texts):
            if len(text.strip()) < 10:
                results.append({
                    'index': i,
                    'success': False,
                    'error': 'The text is too short.'
                })
                continue
            
            try:
                result = dtd_model.predict_fast(text.strip())
                results.append({
                    'index': i,
                    'success': True,
                    'prediction': result['prediction'],
                    'ai_probability': float(result['ai_probability']),
                    'confidence': float(result['confidence']),
                    'prediction_time_ms': float(result['prediction_time'])
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        total_time = (time.time() - total_start_time) * 1000
        
        return jsonify({
            'success': True,
            'results': results,
            'total_time_ms': total_time,
            'processed_count': len([r for r in results if r['success']]),
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        return jsonify({
            'success': False,
            'error': f'Batch detection error: {str(e)}'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'success': True,
        'stats': {
            'model_status': 'loaded' if dtd_model else 'not_loaded',
            'version': '2.0',
            'features': [
                'High-accuracy detection (AUC: 0.9899)',
                'Fast response (23 texts/sec)',
                'Batch processing support',
                'Mixed-authorship analysis',
                'REST API interface'
            ],
            'supported_languages': ['English'],
            'max_text_length': 10000,
            'max_batch_size': 10
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'status': 'healthy',
        'model_loaded': dtd_model is not None,
        'timestamp': time.time()
    })

if __name__ == '__main__':
           
    if initialize_model():
        print("=== DTD Web app startup ===")
        print("Model loaded successfully")
        print("Access URL: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Model loading failed. Please train the model first.")
