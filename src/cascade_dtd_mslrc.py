                      

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from optimized_dtd_v2 import UltraFastDTD
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CascadeResult:
    prediction: str
    confidence: float 
    ai_probability: float
    stage_used: str                     
    processing_time: float
    dtd_confidence: Optional[float] = None
    mslrc_scores: Optional[Dict] = None

class MSLRCAnalyzer:
    
    def __init__(self, model_configs: List[Dict]):
        self.models = {}
        self.tokenizers = {}
        self.model_names = []
        
               
        for config in model_configs:
            name = config['name']
            model_name = config['model_name']
            
            try:
                logger.info(f"Loading model: {name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                self.tokenizers[name] = tokenizer
                self.models[name] = model
                self.model_names.append(name)
                
                         
                model.eval()
                logger.info(f"Successfully loaded: {name}")
                
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
    
    def compute_cross_model_matrix(self, text: str) -> np.ndarray:
        n_models = len(self.model_names)
        matrix = np.zeros((n_models, n_models))
        
                     
        model_probs = {}
        
        for i, model_name in enumerate(self.model_names):
            try:
                tokenizer = self.tokenizers[model_name]
                model = self.models[model_name]
                
                       
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[0, -1, :]                    
                    probs = F.softmax(logits, dim=-1)
                    model_probs[model_name] = probs
                    
            except Exception as e:
                logger.warning(f"Error computing probabilities for {model_name}: {e}")
                model_probs[model_name] = torch.zeros(50257)                    
        
                  
        for i, model_i in enumerate(self.model_names):
            for j, model_j in enumerate(self.model_names):
                if model_i in model_probs and model_j in model_probs:
                                       
                    prob_i = model_probs[model_i]
                    prob_j = model_probs[model_j]
                    
                                              
                    k = 1000
                    top_indices = torch.topk(prob_i + prob_j, k=k).indices
                    
                    prob_i_top = prob_i[top_indices]
                    prob_j_top = prob_j[top_indices]
                    
                    similarity = F.cosine_similarity(
                        prob_i_top.unsqueeze(0), 
                        prob_j_top.unsqueeze(0)
                    ).item()
                    
                    matrix[i, j] = similarity
        
        return matrix
    
    def analyze_ladder_response_curves(self, matrix: np.ndarray) -> Dict[str, float]:
        features = {}
        
                         
        diagonal = np.diag(matrix)
        features['diagonal_mean'] = float(np.mean(diagonal))
        features['diagonal_std'] = float(np.std(diagonal))
        features['diagonal_min'] = float(np.min(diagonal))
        
                           
        off_diagonal = matrix[np.triu_indices_from(matrix, k=1)]
        features['off_diagonal_mean'] = float(np.mean(off_diagonal))
        features['off_diagonal_std'] = float(np.std(off_diagonal))
        features['off_diagonal_max'] = float(np.max(off_diagonal))
        
                   
        features['matrix_trace'] = float(np.trace(matrix))
        features['matrix_det'] = float(np.linalg.det(matrix + 1e-6 * np.eye(matrix.shape[0])))
        features['matrix_rank'] = float(np.linalg.matrix_rank(matrix))
        
                           
        eigenvals = np.linalg.eigvals(matrix)
        eigenvals_real = eigenvals.real
        features['eigenval_max'] = float(np.max(eigenvals_real))
        features['eigenval_second'] = float(np.sort(eigenvals_real)[-2])
        features['eigenval_ratio'] = float(features['eigenval_max'] / (features['eigenval_second'] + 1e-6))
        
                              
        row_vars = np.var(matrix, axis=1)
        col_vars = np.var(matrix, axis=0) 
        features['row_var_mean'] = float(np.mean(row_vars))
        features['col_var_mean'] = float(np.mean(col_vars))
        features['asymmetry'] = float(np.mean(np.abs(matrix - matrix.T)))
        
        return features
    
    def predict(self, text: str) -> Dict:
        start_time = time.time()
        
        try:
                     
            matrix = self.compute_cross_model_matrix(text)
            
                      
            features = self.analyze_ladder_response_curves(matrix)
            
                                   
                                 
            
                                
            consistency_score = features['off_diagonal_mean'] 
            eigenval_dominance = features['eigenval_ratio']
            asymmetry_penalty = features['asymmetry']
            
                    
            ai_score = (
                0.4 * consistency_score + 
                0.3 * min(eigenval_dominance / 10.0, 1.0) + 
                0.3 * (1.0 - min(asymmetry_penalty * 10, 1.0))
            )
            
            ai_probability = max(0.0, min(1.0, ai_score))
            confidence = abs(ai_probability - 0.5) * 2            
            prediction = "AI-generated" if ai_probability > 0.5 else "Human"
            
            processing_time = time.time() - start_time
            
            return {
                'prediction': prediction,
                'ai_probability': ai_probability,
                'confidence': confidence,
                'processing_time': processing_time,
                'matrix_features': features,
                'cross_model_matrix': matrix.tolist()
            }
            
        except Exception as e:
            logger.error(f"MS-LRC prediction failed: {e}")
                     
            return {
                'prediction': 'Unknown',
                'ai_probability': 0.5,
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

class CascadeDTDMSLRC:
    
    def __init__(self, 
                 dtd_model_path: str,
                 confidence_threshold: float = 0.9,
                 mslrc_models: Optional[List[Dict]] = None):
        self.confidence_threshold = confidence_threshold
        
                  
        logger.info("Initializing DTD model...")
        self.dtd = UltraFastDTD()
        self.dtd.load_model(dtd_model_path)
        
                      
        if mslrc_models is None:
            mslrc_models = [
                {'name': 'gpt2', 'model_name': 'gpt2'},
                {'name': 'distilgpt2', 'model_name': 'distilgpt2'},
            ]
        
        logger.info("Initializing MS-LRC analyzer...")
        self.mslrc = MSLRCAnalyzer(mslrc_models)
        
              
        self.stats = {
            'total_predictions': 0,
            'dtd_direct': 0,
            'mslrc_secondary': 0,
            'dtd_avg_time': 0.0,
            'mslrc_avg_time': 0.0
        }
        
        logger.info("Cascade DTD-MS-LRC system initialized successfully!")
    
    def predict(self, text: str, return_detailed: bool = False) -> CascadeResult:
        start_time = time.time()
        
                          
        dtd_result = self.dtd.predict_fast(text)
        dtd_confidence = dtd_result['confidence']
        dtd_time = time.time() - start_time
        
        self.stats['total_predictions'] += 1
        self.stats['dtd_avg_time'] = (
            (self.stats['dtd_avg_time'] * (self.stats['total_predictions'] - 1) + dtd_time) / 
            self.stats['total_predictions']
        )
        
                       
        if dtd_confidence >= self.confidence_threshold:
            self.stats['dtd_direct'] += 1
            
            return CascadeResult(
                prediction=dtd_result['prediction'],
                confidence=dtd_result['confidence'],
                ai_probability=dtd_result['ai_probability'],
                stage_used="DTD",
                processing_time=dtd_time,
                dtd_confidence=dtd_confidence
            )
        
                             
        logger.info(f"DTD confidence {dtd_confidence:.3f} below threshold {self.confidence_threshold}, using MS-LRC")
        
        mslrc_start = time.time()
        mslrc_result = self.mslrc.predict(text)
        mslrc_time = time.time() - mslrc_start
        
        self.stats['mslrc_secondary'] += 1
        self.stats['mslrc_avg_time'] = (
            (self.stats['mslrc_avg_time'] * (self.stats['mslrc_secondary'] - 1) + mslrc_time) / 
            self.stats['mslrc_secondary']
        )
        
        total_time = time.time() - start_time
        
        return CascadeResult(
            prediction=mslrc_result['prediction'],
            confidence=mslrc_result['confidence'],
            ai_probability=mslrc_result['ai_probability'],
            stage_used="MS-LRC",
            processing_time=total_time,
            dtd_confidence=dtd_confidence,
            mslrc_scores=mslrc_result.get('matrix_features') if return_detailed else None
        )
    
    def batch_predict(self, texts: List[str]) -> List[CascadeResult]:
        return [self.predict(text) for text in texts]
    
    def get_statistics(self) -> Dict:
        total = self.stats['total_predictions']
        if total == 0:
            return self.stats
            
        dtd_ratio = self.stats['dtd_direct'] / total
        mslrc_ratio = self.stats['mslrc_secondary'] / total
        
        return {
            **self.stats,
            'dtd_usage_ratio': dtd_ratio,
            'mslrc_usage_ratio': mslrc_ratio,
            'efficiency_gain': f"DTD handles {dtd_ratio:.1%} samples in {self.stats['dtd_avg_time']:.0f}ms, "
                             f"MS-LRC handles {mslrc_ratio:.1%} in {self.stats['mslrc_avg_time']:.0f}ms"
        }

      
if __name__ == "__main__":
             
    model_path = r"C:\Users\blc\Desktop\B\B2-DeepFake Text Detector (DTD)\optimized_dtd_model.pkl"
    
    cascade = CascadeDTDMSLRC(
        dtd_model_path=model_path,
        confidence_threshold=0.85,                 
        mslrc_models=[
            {'name': 'gpt2', 'model_name': 'gpt2'},
            {'name': 'distilgpt2', 'model_name': 'distilgpt2'}
        ]
    )
    
          
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",          
        "Artificial intelligence has revolutionized the way we approach complex problem-solving tasks, enabling unprecedented levels of automation and efficiency across various domains.",           
        "In my experience, working on this project has been both challenging and rewarding.",        
    ]
    
    print("=== DTD-MS-LRC Cascade Testing ===")
    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1} ---")
        print(f"Text: {text}")
        
        result = cascade.predict(text, return_detailed=True)
        print(f"Stage: {result.stage_used}")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"AI Probability: {result.ai_probability:.3f}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        if result.dtd_confidence:
            print(f"DTD Confidence: {result.dtd_confidence:.3f}")
    
    print(f"\n=== System Statistics ===")
    stats = cascade.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
