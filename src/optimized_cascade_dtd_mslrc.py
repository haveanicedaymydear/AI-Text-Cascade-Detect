                      

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from optimized_dtd_v2 import UltraFastDTD
import logging
import json
from pathlib import Path

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
    mslrc_confidence: Optional[float] = None
    feature_analysis: Optional[Dict] = None
    efficiency_gain: Optional[float] = None                             

class EnhancedMSLRCAnalyzer:
    
    def __init__(self, 
                 model_configs: List[Dict],
                 classifier_path: Optional[str] = None):
        self.models = {}
        self.tokenizers = {}
        self.model_names = []
        self.feature_scaler = StandardScaler()
        
                   
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
                
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        
                   
        if classifier_path and Path(classifier_path).exists():
            logger.info(f"Loading trained MS-LRC classifier from {classifier_path}")
            with open(classifier_path, 'rb') as f:
                classifier_data = pickle.load(f)
                self.classifier = classifier_data['classifier']
                self.feature_scaler = classifier_data['scaler']
        else:
            logger.info("Using heuristic MS-LRC classifier (will train when data available)")
            self.classifier = None
    
    def compute_enhanced_cross_model_features(self, text: str) -> np.ndarray:
        n_models = len(self.model_names)
        
                      
        matrix = np.zeros((n_models, n_models))
        model_perplexities = []
        model_entropies = []
        model_probs = {}               
        
                       
        for i, model_name in enumerate(self.model_names):
            try:
                tokenizer = self.tokenizers[model_name]
                model = self.models[model_name]
                
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    logits = outputs.logits
                    
                            
                    probs = F.softmax(logits, dim=-1)
                    
                                                   
                    if isinstance(loss, torch.Tensor):
                        perplexity = torch.exp(loss).item()
                    else:
                        perplexity = np.exp(float(loss))
                    model_perplexities.append(perplexity)
                    
                                              
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                    if isinstance(entropy, torch.Tensor):
                        model_entropies.append(entropy.item())
                    else:
                        model_entropies.append(float(entropy))
                    
                                       
                    last_token_probs = probs[0, -1, :].detach().cpu().numpy()
                    model_probs[model_name] = last_token_probs
                    
            except Exception as e:
                logger.warning(f"Error processing {model_name}: {e}")
                model_perplexities.append(100.0)        
                model_entropies.append(5.0)       
                model_probs[model_name] = np.random.rand(50257)        
        
                        
        for i, model_i in enumerate(self.model_names):
            for j, model_j in enumerate(self.model_names):
                if i == j:
                    matrix[i, j] = 1.0       
                else:
                    if model_i in model_probs and model_j in model_probs:
                                      
                        prob_i = model_probs[model_i]
                        prob_j = model_probs[model_j]
                        
                                                      
                        k = min(1000, len(prob_i))
                        top_indices = np.argsort(prob_i + prob_j)[-k:]
                        
                        prob_i_top = prob_i[top_indices]
                        prob_j_top = prob_j[top_indices]
                        
                                 
                        dot_product = np.dot(prob_i_top, prob_j_top)
                        norm_i = np.linalg.norm(prob_i_top)
                        norm_j = np.linalg.norm(prob_j_top)
                        
                        if norm_i > 0 and norm_j > 0:
                            similarity = dot_product / (norm_i * norm_j)
                        else:
                            similarity = 0.0
                        
                        matrix[i, j] = max(0.0, min(1.0, similarity))
                    else:
                        matrix[i, j] = 0.5         
        
                       
        features = self._extract_advanced_features(matrix, model_perplexities, model_entropies)
        
        return features
    
    def _extract_advanced_features(self, 
                                   matrix: np.ndarray, 
                                   perplexities: List[float],
                                   entropies: List[float]) -> np.ndarray:
        
        features = []
        
                     
        features.extend([
            np.mean(np.diag(matrix)),         
            np.std(np.diag(matrix)),           
            np.mean(matrix[np.triu_indices_from(matrix, k=1)]),         
            np.std(matrix[np.triu_indices_from(matrix, k=1)]),           
            np.trace(matrix),     
            np.linalg.det(matrix + 1e-6 * np.eye(matrix.shape[0])),       
        ])
        
                  
        eigenvals = np.linalg.eigvals(matrix).real
        eigenvals_sorted = np.sort(eigenvals)[::-1]
        features.extend([
            eigenvals_sorted[0],         
            eigenvals_sorted[1] if len(eigenvals_sorted) > 1 else 0,         
            eigenvals_sorted[0] / (eigenvals_sorted[1] + 1e-6),         
            np.sum(eigenvals_sorted > 0.1),           
        ])
        
                  
        features.extend([
            np.mean(perplexities),
            np.std(perplexities),
            np.min(perplexities),
            np.max(perplexities),
            np.max(perplexities) - np.min(perplexities),         
        ])
        
                
        features.extend([
            np.mean(entropies),
            np.std(entropies),
            np.min(entropies),
            np.max(entropies),
        ])
        
                      
        off_diagonal = matrix[np.triu_indices_from(matrix, k=1)]
        features.extend([
            np.mean(off_diagonal),
            np.std(off_diagonal),
            np.median(off_diagonal),
            len(off_diagonal[off_diagonal > 0.7]),          
        ])
        
                   
        features.extend([
            np.linalg.norm(matrix, 'fro'),               
            np.mean(np.abs(matrix - matrix.T)),        
            np.linalg.matrix_rank(matrix),       
        ])
        
        return np.array(features)
    
    def predict_with_ml(self, features: np.ndarray) -> Dict:
        
        if self.classifier is None:
                         
            return self._heuristic_prediction(features)
        
        try:
                   
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
                
            prediction_proba = self.classifier.predict_proba(features_scaled)[0]
            ai_probability = prediction_proba[1]            
            
            confidence = max(prediction_proba)
            prediction = "AI-generated" if ai_probability > 0.5 else "Human"
            
            return {
                'prediction': prediction,
                'ai_probability': ai_probability,
                'confidence': confidence,
                'method': 'ML_classifier'
            }
            
        except Exception as e:
            logger.warning(f"ML prediction failed, using heuristic: {e}")
            return self._heuristic_prediction(features)
    
    def _heuristic_prediction(self, features: np.ndarray) -> Dict:
        
                
        diagonal_mean = features[0]              
        diagonal_std = features[1]                 
        off_diagonal_mean = features[2]            
        off_diagonal_std = features[3]               
        eigenval_max = features[8]               
        eigenval_ratio = features[10]             
        perplexity_mean = features[11]            
        perplexity_std = features[7]               
        entropy_mean = features[16]             
        asymmetry = features[21]                   
        
                      
                                
                            
                           
                               
                            
        
                       
        consistency_score = min(1.0, max(0.0, off_diagonal_mean))
        predictability_score = min(1.0, max(0.0, 1.0 / (1.0 + perplexity_std)))
        certainty_score = min(1.0, max(0.0, 1.0 / (1.0 + entropy_mean)))
        symmetry_score = min(1.0, max(0.0, 1.0 / (1.0 + asymmetry * 10)))
        dominance_score = min(1.0, max(0.0, eigenval_ratio / 20.0))
        
                                   
        ai_score = (
            0.25 * consistency_score +              
            0.20 * predictability_score +             
            0.20 * certainty_score +               
            0.20 * symmetry_score +                
            0.15 * dominance_score                  
        )
        
                           
        ai_threshold = 0.6             
        ai_probability = max(0.0, min(1.0, ai_score))
        
                        
        distance_from_threshold = abs(ai_probability - ai_threshold)
        confidence = min(1.0, distance_from_threshold * 2.5)
        
              
        prediction = "AI-generated" if ai_probability > ai_threshold else "Human"
        
        return {
            'prediction': prediction,
            'ai_probability': ai_probability,
            'confidence': confidence,
            'method': 'improved_heuristic',
            'feature_scores': {
                'consistency': consistency_score,
                'predictability': predictability_score,
                'certainty': certainty_score,
                'symmetry': symmetry_score,
                'dominance': dominance_score
            }
        }
    
    def predict(self, text: str) -> Dict:
        start_time = time.time()
        
        try:
                     
            features = self.compute_enhanced_cross_model_features(text)
            
                     
            result = self.predict_with_ml(features)
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['feature_vector'] = features.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced MS-LRC prediction failed: {e}")
            return {
                'prediction': 'Unknown',
                'ai_probability': 0.5,
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

class AdaptiveCascadeDTDMSLRC:
    
    def __init__(self, 
                 dtd_model_path: str,
                 initial_threshold: float = 0.8,
                 mslrc_models: Optional[List[Dict]] = None,
                 adaptation_enabled: bool = True):
        
        self.confidence_threshold = initial_threshold
        self.adaptation_enabled = adaptation_enabled
        
                 
        self.threshold_history = []
        self.performance_history = []
        self.adaptation_window = 50              
        
                  
        logger.info("Initializing enhanced DTD model...")
        self.dtd = UltraFastDTD()
        self.dtd.load_model(dtd_model_path)
        
                        
        if mslrc_models is None:
            mslrc_models = [
                {'name': 'gpt2', 'model_name': 'gpt2'},
                {'name': 'distilgpt2', 'model_name': 'distilgpt2'},
            ]
        
        logger.info("Initializing enhanced MS-LRC analyzer...")
        self.mslrc = EnhancedMSLRCAnalyzer(mslrc_models)
        
              
        self.stats = {
            'total_predictions': 0,
            'dtd_direct': 0,
            'mslrc_secondary': 0,
            'hybrid_decisions': 0,
            'avg_dtd_time': 0.0,
            'avg_mslrc_time': 0.0,
            'efficiency_improvement': 0.0,
            'threshold_adaptations': 0
        }
        
        logger.info("Adaptive Cascade DTD-MS-LRC system ready!")
    
    def predict(self, text: str, ground_truth: Optional[str] = None) -> CascadeResult:
        start_time = time.time()
        
                          
        dtd_result = self.dtd.predict_fast(text)
        dtd_confidence = dtd_result['confidence']
        dtd_time = time.time() - start_time
        
        self.stats['total_predictions'] += 1
        
                 
        self._update_dtd_stats(dtd_time)
        
                       
        if dtd_confidence >= self.confidence_threshold:
            self.stats['dtd_direct'] += 1
            
            result = CascadeResult(
                prediction=dtd_result['prediction'],
                confidence=dtd_result['confidence'],
                ai_probability=dtd_result['ai_probability'],
                stage_used="DTD",
                processing_time=dtd_time,
                dtd_confidence=dtd_confidence,
                efficiency_gain=self._calculate_efficiency_gain("DTD", dtd_time)
            )
            
                   
            if ground_truth and self.adaptation_enabled:
                self._adaptive_learning(result, ground_truth)
            
            return result
        
                             
        logger.info(f"DTD confidence {dtd_confidence:.3f} below threshold {self.confidence_threshold:.3f}, using MS-LRC")
        
        mslrc_start = time.time()
        mslrc_result = self.mslrc.predict(text)
        mslrc_time = time.time() - mslrc_start
        
        self.stats['mslrc_secondary'] += 1
        self._update_mslrc_stats(mslrc_time)
        
        total_time = time.time() - start_time
        
        result = CascadeResult(
            prediction=mslrc_result['prediction'],
            confidence=mslrc_result['confidence'],
            ai_probability=mslrc_result['ai_probability'],
            stage_used="MS-LRC",
            processing_time=total_time,
            dtd_confidence=dtd_confidence,
            mslrc_confidence=mslrc_result['confidence'],
            feature_analysis={
                'method': mslrc_result.get('method'),
                'feature_count': len(mslrc_result.get('feature_vector', []))
            },
            efficiency_gain=self._calculate_efficiency_gain("MS-LRC", total_time)
        )
        
               
        if ground_truth and self.adaptation_enabled:
            self._adaptive_learning(result, ground_truth)
        
        return result
    
    def _calculate_efficiency_gain(self, stage: str, actual_time: float) -> float:
        estimated_full_mslrc_time = 0.5                  
        
        if stage == "DTD":
            return (estimated_full_mslrc_time - actual_time) / estimated_full_mslrc_time
        else:
            return 0.0                 
    
    def _update_dtd_stats(self, time_taken: float):
        total = self.stats['total_predictions']
        self.stats['avg_dtd_time'] = (
            (self.stats['avg_dtd_time'] * (total - 1) + time_taken) / total
        )
    
    def _update_mslrc_stats(self, time_taken: float):
        count = self.stats['mslrc_secondary']
        if count > 1:
            self.stats['avg_mslrc_time'] = (
                (self.stats['avg_mslrc_time'] * (count - 1) + time_taken) / count
            )
        else:
            self.stats['avg_mslrc_time'] = time_taken
    
    def _adaptive_learning(self, result: CascadeResult, ground_truth: str):
        
              
        is_correct = (result.prediction.lower() == ground_truth.lower())
        self.performance_history.append({
            'threshold': self.confidence_threshold,
            'stage': result.stage_used,
            'correct': is_correct,
            'dtd_confidence': result.dtd_confidence,
            'processing_time': result.processing_time
        })
        
                     
        if len(self.performance_history) >= self.adaptation_window:
            self._adjust_threshold()
    
    def _adjust_threshold(self):
        
        recent_performance = self.performance_history[-self.adaptation_window:]
        
                         
        dtd_results = [p for p in recent_performance if p['stage'] == 'DTD']
        mslrc_results = [p for p in recent_performance if p['stage'] == 'MS-LRC']
        
        if dtd_results:
            dtd_accuracy = sum(r['correct'] for r in dtd_results) / len(dtd_results)
            avg_dtd_confidence = np.mean([r['dtd_confidence'] for r in dtd_results])
            
                                    
            if dtd_accuracy > 0.95 and avg_dtd_confidence > self.confidence_threshold + 0.1:
                new_threshold = max(0.7, self.confidence_threshold - 0.05)
                logger.info(f"Lowering threshold from {self.confidence_threshold:.3f} to {new_threshold:.3f} (DTD accuracy: {dtd_accuracy:.3f})")
                self.confidence_threshold = new_threshold
                self.stats['threshold_adaptations'] += 1
            
                            
            elif dtd_accuracy < 0.85:
                new_threshold = min(0.95, self.confidence_threshold + 0.05)
                logger.info(f"Raising threshold from {self.confidence_threshold:.3f} to {new_threshold:.3f} (DTD accuracy: {dtd_accuracy:.3f})")
                self.confidence_threshold = new_threshold
                self.stats['threshold_adaptations'] += 1
    
    def get_enhanced_statistics(self) -> Dict:
        total = self.stats['total_predictions']
        if total == 0:
            return self.stats
        
        dtd_ratio = self.stats['dtd_direct'] / total
        mslrc_ratio = self.stats['mslrc_secondary'] / total
        
                
        if self.stats['avg_mslrc_time'] > 0:
            avg_efficiency = dtd_ratio * (self.stats['avg_mslrc_time'] - self.stats['avg_dtd_time']) / self.stats['avg_mslrc_time']
        else:
            avg_efficiency = 0
        
        enhanced_stats = {
            **self.stats,
            'dtd_usage_ratio': dtd_ratio,
            'mslrc_usage_ratio': mslrc_ratio,
            'current_threshold': self.confidence_threshold,
            'avg_efficiency_gain': avg_efficiency,
            'theoretical_max_efficiency': dtd_ratio,
            'system_adaptation': 'active' if self.adaptation_enabled else 'static'
        }
        
        return enhanced_stats

def run_comprehensive_test():
    
    model_path = r"C:\Users\blc\Desktop\B\B2-DeepFake Text Detector (DTD)\optimized_dtd_model.pkl"
    
                
    cascade = AdaptiveCascadeDTDMSLRC(
        dtd_model_path=model_path,
        initial_threshold=0.65,                         
        adaptation_enabled=True
    )
    
             
    test_cases = [
        {
            'text': "The quick brown fox jumps over the lazy dog.",
            'expected': 'Human',
            'description': 'Simple human text'
        },
        {
            'text': "In accordance with the aforementioned stipulations and pursuant to the comprehensive analysis conducted, it is hereby determined that the implementation of advanced technological solutions represents a paradigm shift in operational efficiency.",
            'expected': 'AI-generated',
            'description': 'Formal/bureaucratic style (likely AI)'
        },
        {
            'text': "Honestly, I'm not sure what to think about this whole situation. It's been really confusing and I keep going back and forth on it.",
            'expected': 'Human',
            'description': 'Informal human style with uncertainty'
        },
        {
            'text': "The integration of artificial intelligence technologies has fundamentally transformed the landscape of modern computational systems, enabling unprecedented levels of automation and optimization across diverse application domains.",
            'expected': 'AI-generated',
            'description': 'Technical academic style (likely AI)'
        },
        {
            'text': "My grandmother always used to say that the best apple pie needs a pinch of love and a dash of patience. She was right - I tried her recipe last weekend and wow, what a difference!",
            'expected': 'Human',
            'description': 'Personal story with emotional content'
        }
    ]
    
    print("=== EnhancedDTD-MS-LRC cascade system test ===\n")
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"--- Test {i}: {case['description']} ---")
        print(f"Text: {case['text'][:80]}{'...' if len(case['text']) > 80 else ''}")
        print(f"Expected: {case['expected']}")
        
        result = cascade.predict(case['text'], ground_truth=case['expected'])
        results.append(result)
        
        print(f"Stage: {result.stage_used}")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"AI probability: {result.ai_probability:.3f}")
        print(f"Processing time: {result.processing_time:.3f}s")
        if result.efficiency_gain:
            print(f"Efficiency gain: {result.efficiency_gain:.1%}")
        print(f"DTD confidence: {result.dtd_confidence:.3f}")
        
               
        is_correct = result.prediction.replace('-generated', '').lower() == case['expected'].replace('-generated', '').lower()
        print(f"Prediction correct: {'' if is_correct else ''}")
        print()
    
          
    print("=== System performance statistics ===")
    stats = cascade.get_enhanced_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    return results, stats

if __name__ == "__main__":
    results, stats = run_comprehensive_test()
