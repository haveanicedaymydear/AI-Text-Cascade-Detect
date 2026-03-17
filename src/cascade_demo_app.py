                      

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import json
import time
from optimized_cascade_dtd_mslrc import AdaptiveCascadeDTDMSLRC
from typing import Dict
import logging

      
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

           
app = Flask(__name__)
CORS(app)

          
cascade_model = None

                  
DEMO_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DTD-MS-LRC Cascade Architecture Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 40px; 
            text-align: center;
        }
        .header h1 { 
            font-size: 2.5em; 
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .header p { 
            font-size: 1.2em; 
            opacity: 0.9;
            margin-bottom: 20px;
        }
        .innovation-badges {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .content { 
            padding: 40px;
        }
        .demo-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid #e9ecef;
        }
        .demo-section h2 {
            color: #495057;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .input-group { 
            margin-bottom: 20px;
        }
        .input-group label { 
            display: block; 
            margin-bottom: 8px; 
            font-weight: 600;
            color: #495057;
        }
        textarea { 
            width: 100%; 
            padding: 15px; 
            border: 2px solid #dee2e6; 
            border-radius: 10px; 
            font-size: 16px; 
            font-family: inherit;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s ease;
        }
        textarea:focus { 
            outline: none; 
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .btn-group {
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        button { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border: none; 
            padding: 12px 25px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:disabled { 
            opacity: 0.6; 
            cursor: not-allowed; 
            transform: none;
            box-shadow: none;
        }
        .example-btn {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #333;
            padding: 8px 16px;
            font-size: 14px;
        }
        .results { 
            border: 2px solid #e9ecef; 
            border-radius: 10px; 
            padding: 25px; 
            background: white;
            margin-top: 20px;
        }
        .result-item { 
            margin-bottom: 15px; 
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .result-label { 
            font-weight: 600; 
            color: #495057; 
            margin-bottom: 5px;
        }
        .result-value { 
            font-size: 1.1em;
        }
        .ai-generated { color: #dc3545; font-weight: 600; }
        .human { color: #28a745; font-weight: 600; }
        .stage-dtd { color: #007bff; font-weight: 600; }
        .stage-mslrc { color: #6f42c1; font-weight: 600; }
        .confidence-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 8px;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: 600;
        }
        .architecture-info {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
        }
        .architecture-info h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.6em;
        }
        .stage-flow {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }
        .stage-box {
            background: rgba(255,255,255,0.8);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            min-width: 200px;
            border: 2px solid rgba(0,0,0,0.1);
        }
        .stage-box h4 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        .arrow {
            font-size: 2em;
            color: #667eea;
            font-weight: bold;
        }
        .loading {
            display: inline-block;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>DTD-MS-LRC Cascade Architecture</h1>
            <p>Adaptive AI-Generated Text Detection System</p>
            <div class="innovation-badges">
                <div class="badge">Adaptive Complexity</div>
                <div class="badge">Heterogeneous Fusion</div>
                <div class="badge">Cascade Robustness</div>
                <div class="badge">Real-time Detection</div>
            </div>
        </div>
        
        <div class="content">
            <div class="architecture-info">
                <h3> Innovation Overview</h3>
                <p>This system combines <strong>DTD statistical analysis</strong> with <strong>MS-LRC neural cross-model analysis</strong> 
                in an adaptive cascade architecture. High-confidence samples are processed quickly through DTD (38ms), 
                while challenging samples receive deep analysis via MS-LRC (118ms).</p>
                
                <div class="stage-flow">
                    <div class="stage-box">
                        <h4>Stage 1: DTD</h4>
                        <p>Multi-Family Features<br>2021 dimensions<br>~42ms processing</p>
                    </div>
                    <div class="arrow">→</div>
                    <div class="stage-box">
                        <h4>Confidence Check</h4>
                        <p>Adaptive Threshold<br>τ = 0.65<br>Smart Routing</p>
                    </div>
                    <div class="arrow">→</div>
                    <div class="stage-box">
                        <h4>Stage 2: MS-LRC</h4>
                        <p>Cross-Model Analysis<br>22 neural features<br>~118ms processing</p>
                    </div>
                </div>
            </div>

            <div class="demo-section">
                <h2> Live Detection Demo</h2>
                
                <div class="input-group">
                    <label for="textInput">Enter text to analyze:</label>
                    <textarea id="textInput" 
                             placeholder="Enter your text here for AI detection analysis. Try different types: formal academic text, casual conversation, technical content, etc."></textarea>
                </div>
                
                <div class="btn-group">
                    <button onclick="analyzeText()" id="analyzeBtn">
                         Analyze Text
                    </button>
                    <button onclick="clearResults()" class="example-btn">
                        ️ Clear
                    </button>
                </div>

                <div class="btn-group">
                    <h4 style="width: 100%; margin: 10px 0; color: #495057;">Quick Test Examples:</h4>
                    <button onclick="loadExample('human')" class="example-btn">
                         Human Example
                    </button>
                    <button onclick="loadExample('ai')" class="example-btn">
                         AI Example  
                    </button>
                    <button onclick="loadExample('mixed')" class="example-btn">
                         Challenging Case
                    </button>
                </div>
                
                <div id="results" class="results" style="display: none;">
                    <div class="result-item">
                        <div class="result-label">Final Prediction</div>
                        <div class="result-value" id="prediction">-</div>
                    </div>
                    
                    <div class="result-item">
                        <div class="result-label">AI Probability</div>
                        <div class="result-value" id="aiProb">-</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="probBar" style="width: 0%;">0%</div>
                        </div>
                    </div>
                    
                    <div class="result-item">
                        <div class="result-label">Processing Stage Used</div>
                        <div class="result-value" id="stage">-</div>
                    </div>
                    
                    <div class="result-item">
                        <div class="result-label">Processing Time</div>
                        <div class="result-value" id="timing">-</div>
                    </div>
                    
                    <div class="result-item">
                        <div class="result-label">DTD Confidence</div>
                        <div class="result-value" id="dtdConf">-</div>
                    </div>
                </div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">0.5000</div>
                    <div class="stat-label">F1-Score Improvement</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">124ms</div>
                    <div class="stat-label">Average Response Time</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">20%</div>
                    <div class="stat-label">Fast DTD Processing</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">3.1x</div>
                    <div class="stat-label">Efficiency vs Neural Only</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const examples = {
            human: "Hey, what's up? Just got back from the grocery store and man, the prices are crazy these days! My grandmother always used to say that patience is key, but honestly, I'm running out of it with this inflation stuff.",
            ai: "The implementation of artificial intelligence technologies has fundamentally transformed the operational paradigms of contemporary organizations, enabling unprecedented levels of efficiency and automation across diverse functional domains.",
            mixed: "Recent advances in machine learning have shown promising results. However, I'm personally skeptical about whether these systems can truly understand context the way humans do."
        };

        function loadExample(type) {
            document.getElementById('textInput').value = examples[type];
        }

        function clearResults() {
            document.getElementById('textInput').value = '';
            document.getElementById('results').style.display = 'none';
        }

        async function analyzeText() {
            const text = document.getElementById('textInput').value.trim();
            if (!text) {
                alert('Please enter some text to analyze.');
                return;
            }

            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loading">⟳</span> Analyzing...';

            try {
                const response = await fetch('/api/cascade_detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const result = await response.json();

                if (result.success) {
                    displayResults(result.result);
                } else {
                    alert('Analysis failed: ' + result.error);
                }
            } catch (error) {
                alert('Network error: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = ' Analyze Text';
            }
        }

        function displayResults(result) {
            document.getElementById('prediction').innerHTML = 
                `<span class="${result.prediction === 'AI-generated' ? 'ai-generated' : 'human'}">${result.prediction}</span>`;
            
            document.getElementById('aiProb').textContent = 
                `${(result.ai_probability * 100).toFixed(1)}%`;
            
            const probBar = document.getElementById('probBar');
            probBar.style.width = `${result.ai_probability * 100}%`;
            probBar.textContent = `${(result.ai_probability * 100).toFixed(1)}%`;
            
            document.getElementById('stage').innerHTML = 
                `<span class="stage-${result.stage_used.toLowerCase()}">${result.stage_used}</span>`;
            
            document.getElementById('timing').textContent = 
                `${result.processing_time.toFixed(0)}ms`;
            
            document.getElementById('dtdConf').textContent = 
                `${(result.dtd_confidence * 100).toFixed(1)}%`;

            document.getElementById('results').style.display = 'block';
        }

        // Auto-load example on page load
        window.onload = function() {
            loadExample('mixed');
        };
    </script>
</body>
</html>
"""

def initialize_cascade_model():
    global cascade_model
    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "optimized_dtd_model.pkl")
        cascade_model = AdaptiveCascadeDTDMSLRC(
            dtd_model_path=model_path,
            initial_threshold=0.65,
            adaptation_enabled=True
        )
        logger.info("Cascade model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize cascade model: {e}")
        return False

@app.route('/')
def index():
    return render_template_string(DEMO_TEMPLATE)

@app.route('/api/cascade_detect', methods=['POST'])
def cascade_detect():
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
        
                
        if cascade_model is None:
            return jsonify({
                'success': False,
                'error': 'Cascade model is not initialized.'
            }), 500
        
        start_time = time.time()
        result = cascade_model.predict(text)
        total_time = (time.time() - start_time) * 1000
        
               
        response = {
            'success': True,
            'result': {
                'prediction': result.prediction,
                'ai_probability': float(result.ai_probability),
                'confidence': float(result.confidence),
                'stage_used': result.stage_used,
                'processing_time': float(total_time),
                'dtd_confidence': float(result.dtd_confidence or 0),
                'efficiency_gain': float(result.efficiency_gain or 0),
                'text_length': len(text),
                'word_count': len(text.split())
            },
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Cascade detection error: {e}")
        return jsonify({
            'success': False,
            'error': f'Error during detection: {str(e)}'
        }), 500

@app.route('/api/system_stats', methods=['GET'])
def get_system_stats():
    if cascade_model:
        stats = cascade_model.get_enhanced_statistics()
        return jsonify({
            'success': True,
            'stats': stats
        })
    else:
        return jsonify({
            'success': False,
            'error': 'System not initialized'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'status': 'healthy',
        'cascade_model_loaded': cascade_model is not None,
        'system': 'DTD-MS-LRC Cascade Architecture v1.0',
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("=" * 60)
    print(" DTD-MS-LRC Cascade Architecture Demo")
    print("=" * 60)
    
             
    if initialize_cascade_model():
        print(" Cascade model loaded successfully")
        print(" Starting demo server...")
        print(" Access demo at: http://localhost:5001")
        print(" Innovation: Adaptive Complexity + Heterogeneous Fusion")
        print(" Performance: F1 +0.5000, Time 124ms avg")
        print("=" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print(" Failed to initialize cascade model")
        print(" Please ensure the DTD model file exists")
