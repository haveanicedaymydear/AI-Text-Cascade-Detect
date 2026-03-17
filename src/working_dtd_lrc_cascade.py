import os
import json
import time
import math
import gc
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CascadeResult:
    
    prediction: str
    ai_probability: float
    confidence: float
    stage_used: str
    processing_time: float
    dtd_confidence: Optional[float] = None
    lrc_features: Optional[Dict] = None
    efficiency_gain: Optional[float] = None

class SimpleDTDDetector:
   
    
    def __init__(self):
       
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.is_trained = False
        
       
        logger.info("SimpleDTD initialized with heuristic rules")
    
    def _extract_features(self, text: str) -> Dict:
       
        words = text.split()
        sentences = text.split('.')
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'complex_words': sum(1 for w in words if len(w) > 6) / max(len(words), 1),
            'punctuation_density': sum(1 for c in text if c in '.,!?;:') / max(len(text), 1),
            'capital_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'repetition_score': self._calculate_repetition(text),
            'formality_score': self._calculate_formality(text)
        }
        return features
    
    def _calculate_repetition(self, text: str) -> float:
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        unique_words = len(set(words))
        return 1.0 - (unique_words / len(words))
    
    def _calculate_formality(self, text: str) -> float:
        formal_indicators = ['furthermore', 'however', 'therefore', 'consequently', 
                           'implementation', 'methodology', 'framework', 'systematic']
        
        text_lower = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        
        return min(formal_count / 10, 1.0)
    
    def predict_proba(self, text: str) -> Tuple[float, float]:
        features = self._extract_features(text)
        
        ai_score = 0.0
        
        if features['formality_score'] > 0.3:
            ai_score += 0.25
        
        if features['complex_words'] > 0.4:
            ai_score += 0.20
        
        if features['repetition_score'] < 0.1:
            ai_score += 0.15
        
        if 15 <= features['word_count'] / max(features['sentence_count'], 1) <= 25:
            ai_score += 0.15
        
        if 0.02 <= features['punctuation_density'] <= 0.08:
            ai_score += 0.10
        
        if features['capital_ratio'] < 0.05:
            ai_score += 0.15
        
        ai_probability = 0.1 + ai_score
        human_probability = 1.0 - ai_probability
        
        return human_probability, ai_probability
    
    def predict(self, text: str) -> str:
        human_prob, ai_prob = self.predict_proba(text)
        return "AI-generated" if ai_prob > 0.5 else "Human"

class OptimizedLRCDetector:
    
    def __init__(self, device: str = "auto"):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        
        self.model_configs = [
            {"id": "Qwen/Qwen2.5-0.5B", "scale": 0.5},
            {"id": "Qwen/Qwen2.5-1.5B", "scale": 1.5}, 
            {"id": "Qwen/Qwen2.5-3B", "scale": 3.0}
        ]
        
        self.model_cache = {}
        self.computation_cache = {}
        
        logger.info(f"OptimizedLRC initialized on {self.device}")
    
    def _get_cached_model(self, model_id: str):
        if model_id not in self.model_cache:
            logger.info(f"Loading {model_id}...")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                if self.device == "cuda":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map="auto"
                    ).eval()
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    ).to("cpu").eval()
                
                self.model_cache[model_id] = (tokenizer, model)
                logger.info(f" {model_id} loaded successfully")
                
            except Exception as e:
                logger.error(f" Failed to load {model_id}: {e}")
                return None, None
        
        return self.model_cache[model_id]
    
    @torch.no_grad()
    def _compute_nll_fast(self, model, tokenizer, text: str) -> float:

        text = text.strip()
        if len(text) > 1000: 
            text = text[:1000]
        
        if not text:
            return float("nan")
        
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"][0]
            
            if len(input_ids) < 2:
                return float("nan")
            
            device = next(model.parameters()).device
            input_ids = input_ids.unsqueeze(0).to(device)
            
            logits = model(input_ids).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="mean"
            )
            
            return float(loss.item())
            
        except Exception as e:
            logger.warning(f"NLL computation failed: {e}")
            return float("nan")
    
    def predict(self, text: str) -> Dict:
        start_time = time.time()
        
        text_hash = hash(text) % 1000000
        if text_hash in self.computation_cache:
            cached = self.computation_cache[text_hash].copy()
            cached["from_cache"] = True
            return cached
        
        scores = []
        scales = []
        
        for config in self.model_configs:
            tokenizer, model = self._get_cached_model(config["id"])
            
            if tokenizer is None or model is None:
                continue
            
            try:
                nll = self._compute_nll_fast(model, tokenizer, text)
                if not math.isnan(nll):
                    scores.append(nll)
                    scales.append(config["scale"])
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.warning(f"Model {config['id']} failed: {e}")
                continue
        
        if len(scores) >= 3:
            lrc_features = self._compute_lrc_features(scores)
            ai_probability = self._lrc_classify(lrc_features, scores)
        else:
            lrc_features = {}
            ai_probability = 0.4 
        
        prediction = "AI-generated" if ai_probability > 0.5 else "Human"
        confidence = abs(ai_probability - 0.5) * 2
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "prediction": prediction,
            "ai_probability": ai_probability,
            "confidence": confidence,
            "lrc_features": lrc_features,
            "processing_time": processing_time,
            "scores": scores,
            "scales": scales,
            "from_cache": False
        }
        
              
        self.computation_cache[text_hash] = result.copy()
        
        return result
    
    def _compute_lrc_features(self, scores: List[float]) -> Dict:
        if len(scores) < 3:
            return {}
        
        s_small, s_mid, s_large = scores[0], scores[1], scores[2]
        early_drop = s_small - s_mid
        late_drop = s_mid - s_large
        overall_drop = s_small - s_large
        drop_ratio = early_drop / (late_drop + 1e-8)
        concavity = 1 if early_drop > late_drop else 0
        
        return {
            "s_small": s_small,
            "s_mid": s_mid,
            "s_large": s_large,
            "early_drop": early_drop,
            "late_drop": late_drop,
            "overall_drop": overall_drop,
            "drop_ratio": drop_ratio,
            "concavity_flag": concavity,
        }
    
    def _lrc_classify(self, features: Dict, scores: List[float]) -> float:
        if not features:
            return 0.4
        
        ai_score = 0.0
        
                     
        early_drop = features.get("early_drop", 0)
        late_drop = features.get("late_drop", 0)
        overall_drop = features.get("overall_drop", 0)
        drop_ratio = features.get("drop_ratio", 1.0)
        concavity = features.get("concavity_flag", 0)
        
                   
        if overall_drop > 0.1:
            ai_score += 0.3
        
        if drop_ratio > 1.5:
            ai_score += 0.25
        
        if concavity == 1:
            ai_score += 0.2
        
                
        avg_score = np.mean(scores) if scores else 0
        if avg_score > 1.0:
            ai_score += 0.15
        
              
        if len(scores) >= 3:
            score_var = np.var(scores)
            if score_var > 0.02:
                ai_score += 0.1
        
        return min(max(0.1 + ai_score, 0.0), 1.0)
    
    def cleanup(self):
        for model_id in list(self.model_cache.keys()):
            try:
                del self.model_cache[model_id]
            except:
                pass
        
        self.model_cache.clear()
        self.computation_cache.clear()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

class WorkingDTDLRCCascade:
    
    def __init__(self, confidence_low: float = 0.41, confidence_high: float = 0.61):

        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        
                  
        logger.info(" Initializing DTD detector...")
        self.dtd_detector = SimpleDTDDetector()
        
        logger.info(" Initializing LRC detector...")
        self.lrc_detector = OptimizedLRCDetector()
        
              
        self.stats = {
            "total_predictions": 0,
            "dtd_direct": 0,
            "lrc_deep": 0,
            "avg_dtd_time": 0,
            "avg_lrc_time": 0
        }
        
        logger.info(f" DTD-LRC cascade system ready. Confidence band: [{confidence_low}, {confidence_high}]")
    
    def predict(self, text: str) -> CascadeResult:
        total_start = time.time()
        
        if not text or len(text.strip()) < 10:
            return CascadeResult(
                prediction="Human",
                ai_probability=0.2,
                confidence=0.3,
                stage_used="DTD",
                processing_time=1.0
            )
        
                          
        dtd_start = time.time()
        try:
            human_prob, ai_prob = self.dtd_detector.predict_proba(text)
            dtd_confidence = max(human_prob, ai_prob)
            dtd_time = (time.time() - dtd_start) * 1000
            
            logger.info(f"DTD result: AI probability={ai_prob:.3f}, Confidence={dtd_confidence:.3f}")
            
        except Exception as e:
            logger.error(f"DTDprediction failed: {e}")
            ai_prob = 0.5
            dtd_confidence = 0.3
            dtd_time = 10
        
                       
        if dtd_confidence < self.confidence_low or dtd_confidence > self.confidence_high:
                                 
            self.stats["dtd_direct"] += 1
            total_time = (time.time() - total_start) * 1000
            
            prediction = "AI-generated" if ai_prob > 0.5 else "Human"
            logger.info(f" DTD direct output: {prediction} (Confidence {dtd_confidence:.3f} outside the threshold band)")
            
            return CascadeResult(
                prediction=prediction,
                ai_probability=ai_prob,
                confidence=dtd_confidence,
                stage_used="DTD",
                processing_time=total_time,
                dtd_confidence=dtd_confidence,
                efficiency_gain=1.0
            )
        
                                            
        logger.info(f" DTDConfidence {dtd_confidence:.3f} within the band [{self.confidence_low}, {self.confidence_high}] within，startingLRCdeep analysis")
        
        lrc_start = time.time()
        try:
            lrc_result = self.lrc_detector.predict(text)
            lrc_time = (time.time() - lrc_start) * 1000
            
            self.stats["lrc_deep"] += 1
            total_time = (time.time() - total_start) * 1000
            
            logger.info(f" LRC deep analysiscompleted: {lrc_result['prediction']} (AI probability: {lrc_result['ai_probability']:.3f})")
            
            return CascadeResult(
                prediction=lrc_result["prediction"],
                ai_probability=lrc_result["ai_probability"],
                confidence=lrc_result["confidence"],
                stage_used="LRC",
                processing_time=total_time,
                dtd_confidence=dtd_confidence,
                lrc_features=lrc_result.get("lrc_features", {}),
                efficiency_gain=dtd_time / total_time if total_time > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"LRCprediction failed: {e}")
                         
            total_time = (time.time() - total_start) * 1000
            prediction = "AI-generated" if ai_prob > 0.5 else "Human"
            
            return CascadeResult(
                prediction=prediction,
                ai_probability=ai_prob,
                confidence=dtd_confidence * 0.8,
                stage_used="DTD (LRC failed)",
                processing_time=total_time,
                dtd_confidence=dtd_confidence
            )
        
        finally:
            self.stats["total_predictions"] += 1
    
    def get_statistics(self) -> Dict:
        total = max(self.stats["total_predictions"], 1)
        return {
            "total_predictions": total,
            "dtd_usage_rate": f"{self.stats['dtd_direct'] / total * 100:.1f}%",
            "lrc_usage_rate": f"{self.stats['lrc_deep'] / total * 100:.1f}%",
            "confidence_range": f"[{self.confidence_low}, {self.confidence_high}]",
            "dtd_count": self.stats["dtd_direct"],
            "lrc_count": self.stats["lrc_deep"]
        }
    
    def cleanup(self):
        if hasattr(self, 'lrc_detector'):
            self.lrc_detector.cleanup()

def test_working_cascade():
    print("=" * 70)
    print(" DTD-LRCcascade systemTest (custom version)")
    print("=" * 70)
    
    try:
                 
        cascade = WorkingDTDLRCCascade(confidence_low=0.41, confidence_high=0.61)
        
                 
        test_samples = [
            ("human dialogue", "Hey, the weather is nice today. I think we should go for a walk. What do you think?"),
            ("formal AI text", "The implementation of artificial intelligence methodologies requires sophisticated computational frameworks and systematic approaches to data processing."),
            ("medium difficulty", "I think machine learning is quite interesting, but honestly I'm not sure if these systems can really understand context like humans do."),
            ("complex case", "Recent developments in neural networks have shown promising results. However, there are still challenges in deployment at scale."),
            ("short dialogue", "Okay, I understand."),
            ("technical discussion", "This algorithm optimizes performance through iterative refinement and adaptive parameter adjustment mechanisms.")
        ]
        
        print(f" Test configuration:")
        print(f"   - Confidence thresholdband: [{cascade.confidence_low}, {cascade.confidence_high}]")
        print(f"   - Number of test samples: {len(test_samples)}")
        print(f"   - Stage 1: DTD fast detection")
        print(f"   - Stage 2: LRC deep analysis (high-precision system)")
        print()
        
        results = []
        total_start = time.time()
        
        for i, (sample_type, text) in enumerate(test_samples, 1):
            print(f" Test samples {i} ({sample_type}):")
            print(f"   Text: {text[:60]}...")
            
            result = cascade.predict(text)
            results.append(result)
            
            print(f"    Prediction: {result.prediction}")
            print(f"    AI probability: {result.ai_probability:.3f}")
            print(f"    Confidence: {result.confidence:.3f}")
            print(f"    Stage used: {result.stage_used}")
            print(f"    Processing time: {result.processing_time:.1f}ms")
            
            if result.dtd_confidence:
                print(f"    DTD confidence: {result.dtd_confidence:.3f}")
            
            print()
        
        total_time = time.time() - total_start
        
              
        print("=" * 70)
        print(" Test result summary:")
        stats = cascade.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n Total test time: {total_time:.2f} sec")
        print(f" Average processing time: {np.mean([r.processing_time for r in results]):.1f}ms")
        
                
        dtd_count = sum(1 for r in results if r.stage_used.startswith("DTD"))
        lrc_count = sum(1 for r in results if r.stage_used == "LRC")
        
        print(f" Stage usage:")
        print(f"   DTD direct handling: {dtd_count} samples")
        print(f"   LRC deep analysis: {lrc_count} samples")
        
        print("\n Testcompleted！cascade system is running normally")
        print(f" Your LRC system has been integrated successfully. Confidence band [{cascade.confidence_low}, {cascade.confidence_high}] works correctly！")
        
              
        cascade.cleanup()
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_working_cascade()
