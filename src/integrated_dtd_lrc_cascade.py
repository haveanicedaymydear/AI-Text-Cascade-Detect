                      

import os
import json
import time
import pickle
import logging
import math
import gc
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

         
from optimized_dtd_v2 import AdvancedFeatureExtractor, UltraFastDTD

      
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

                    
class OptimizedLRCDetector:
    
    def __init__(self, device: str = "auto", enable_cache: bool = True):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.enable_cache = enable_cache
        self.model_cache = {}               
        self.tokenizer_cache = {}
        
                               
        self.families = {
            "qwen_efficient": [
                "Qwen/Qwen2.5-0.5B",         
                "Qwen/Qwen2.5-1.5B",           
                "Qwen/Qwen2.5-3B",           
            ]
        }
        
        logger.info(f"LRC Detector initialized with device: {self.device}")
    
    def _get_model_key(self, model_id: str) -> str:
        return f"{model_id}_{self.device}"
    
    def _load_model_cached(self, model_id: str) -> Tuple[Any, Any]:
        cache_key = self._get_model_key(model_id)
        
        if self.enable_cache and cache_key in self.model_cache:
            logger.debug(f"Using cached model: {model_id}")
            return self.model_cache[cache_key]
        
        try:
            logger.info(f"Loading LRC model: {model_id}")
            
                     
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
                      
            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto"          
                ).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                ).to("cpu").eval()
            
            if self.enable_cache:
                self.model_cache[cache_key] = (tokenizer, model)
                
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    @torch.no_grad()
    def _compute_nll_per_byte(self, model, tokenizer, text: str, 
                             max_length: int = 512, stride: int = 128) -> float:
        text = (text or "").strip()
        if not text or len(text) < 10:
            return float("nan")
        
                              
        if len(text) > 2000:
            text = text[:2000]
        
        try:
            input_ids = tokenizer(text, return_tensors="pt", truncation=True, 
                                max_length=max_length)["input_ids"][0]
            L = int(input_ids.size(0))
            
            if L < 2:
                return float("nan")
            
            device = next(model.parameters()).device
            total_nll = 0.0
            total_count = 0
            
                     
            for start in range(0, L - 1, stride):
                end = min(start + max_length, L)
                chunk = input_ids[start:end].unsqueeze(0).to(device)
                
                          
                logits = model(chunk).logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = chunk[:, 1:].contiguous()
                
                        
                if start == 0:
                    valid_from = 0
                else:
                    overlap = max_length - stride
                    valid_from = max(0, overlap - 1)
                
                if valid_from >= shift_labels.size(1):
                    if end == L:
                        break
                    continue
                
                      
                loss_sum = F.cross_entropy(
                    shift_logits[:, valid_from:, :].reshape(-1, shift_logits.size(-1)),
                    shift_labels[:, valid_from:].reshape(-1),
                    reduction="sum"
                )
                
                total_nll += float(loss_sum.item())
                total_count += int(shift_labels[:, valid_from:].numel())
                
                if end == L:
                    break
            
            byte_len = len(text.encode("utf-8"))
            if byte_len == 0 or total_count == 0:
                return float("nan")
            
            return total_nll / byte_len
            
        except Exception as e:
            logger.error(f"NLL computation failed: {e}")
            return float("nan")
    
    def _extract_scale_b(self, model_id: str) -> float:
        import re
        m = re.search(r"-([0-9]+(?:\.[0-9]+)?)B\b", model_id)
        return float(m.group(1)) if m else float("nan")
    
    def _compute_lrc_features(self, scores: List[float], eps: float = 1e-8) -> Dict:
        if len(scores) < 3:
            return {}
        
        s_small, s_mid, s_large = scores[:3]
        early = s_small - s_mid
        late = s_mid - s_large
        overall = s_small - s_large
        ratio = early / (late + eps)
        concavity = 1 if early > late else 0
        
        return {
            "s_small": s_small,
            "s_mid": s_mid, 
            "s_large": s_large,
            "early_drop": early,
            "late_drop": late,
            "overall_drop": overall,
            "drop_ratio": ratio,
            "concavity_flag": concavity,
        }
    
    def predict(self, text: str) -> Dict:
        start_time = time.time()
        
        try:
                            
            family = "qwen_efficient"
            model_ids = self.families[family]
            
            scores = []
            scales = []
            
            for model_id in model_ids:
                try:
                    tokenizer, model = self._load_model_cached(model_id)
                    score = self._compute_nll_per_byte(model, tokenizer, text)
                    scale = self._extract_scale_b(model_id)
                    
                    if not math.isnan(score):
                        scores.append(score)
                        scales.append(scale)
                    
                          
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.warning(f"Model {model_id} failed: {e}")
                    continue
            
            if len(scores) < 3:
                return {
                    "prediction": "Human",        
                    "ai_probability": 0.3,
                    "confidence": 0.5,
                    "lrc_features": {},
                    "error": "Insufficient valid scores"
                }
            
                     
            lrc_features = self._compute_lrc_features(scores)
            
                                   
            ai_probability = self._lrc_classify(lrc_features, scores)
            
            prediction = "AI-generated" if ai_probability > 0.5 else "Human"
            confidence = abs(ai_probability - 0.5) * 2             
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "prediction": prediction,
                "ai_probability": ai_probability,
                "confidence": confidence,
                "lrc_features": lrc_features,
                "processing_time": processing_time,
                "scores": scores,
                "scales": scales
            }
            
        except Exception as e:
            logger.error(f"LRC prediction failed: {e}")
            return {
                "prediction": "Human",
                "ai_probability": 0.3,
                "confidence": 0.2,
                "lrc_features": {},
                "error": str(e)
            }
    
    def _lrc_classify(self, features: Dict, scores: List[float]) -> float:
        if not features:
            return 0.3
        
                
        early_drop = features.get("early_drop", 0)
        late_drop = features.get("late_drop", 0)
        overall_drop = features.get("overall_drop", 0)
        drop_ratio = features.get("drop_ratio", 1.0)
        concavity = features.get("concavity_flag", 0)
        
                                 
        ai_score = 0.0
        
                                        
        if overall_drop > 0.1:
            ai_score += 0.25
        elif overall_drop > 0.05:
            ai_score += 0.15
        
                         
        if drop_ratio > 1.2:           
            ai_score += 0.20
        elif drop_ratio < 0.8:           
            ai_score += 0.10
        
                    
        if concavity == 1:        
            ai_score += 0.15
        
                     
        avg_score = np.mean(scores) if scores else 0
        if avg_score > 0.8:         
            ai_score += 0.15
        elif avg_score < 0.3:         
            ai_score += 0.10
        
                   
        if len(scores) >= 3:
            score_var = np.var(scores)
            if score_var > 0.05:       
                ai_score += 0.15
        
                
        ai_probability = 0.2 + ai_score
        
                  
        return max(0.0, min(1.0, ai_probability))
    
    def cleanup(self):
        for model_key in list(self.model_cache.keys()):
            try:
                _, model = self.model_cache[model_key]
                del model
            except:
                pass
        
        self.model_cache.clear()
        self.tokenizer_cache.clear()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()

class IntegratedDTDLRCCascade:
    
    def __init__(self, dtd_model_path: str, confidence_low: float = 0.41, 
                 confidence_high: float = 0.61, device: str = "auto"):
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        self.device = device
        
                   
        logger.info("Initializing DTD detector...")
        self.dtd_detector = self._load_dtd_model(dtd_model_path)
        
                   
        logger.info("Initializing LRC detector...")
        self.lrc_detector = OptimizedLRCDetector(device=device)
        
              
        self.stats = {
            "total_predictions": 0,
            "dtd_direct": 0,
            "lrc_deep": 0,
            "dtd_skip": 0,
            "avg_dtd_time": 0,
            "avg_lrc_time": 0
        }
        
        logger.info(f"DTD-LRC Cascade initialized! Confidence range: [{confidence_low}, {confidence_high}]")
    
    def _load_dtd_model(self, model_path: str) -> UltraFastDTD:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"DTD model not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                dtd_model = pickle.load(f)
            
            if hasattr(dtd_model, 'predict') and hasattr(dtd_model, 'predict_proba'):
                logger.info("DTD model loaded successfully")
                return dtd_model
            else:
                raise ValueError("Invalid DTD model format")
                
        except Exception as e:
            logger.error(f"Failed to load DTD model: {e}")
            raise
    
    def predict(self, text: str) -> CascadeResult:
        total_start = time.time()
        
        if not text or len(text.strip()) < 10:
            return CascadeResult(
                prediction="Human",
                ai_probability=0.1,
                confidence=0.2,
                stage_used="DTD",
                processing_time=1.0,
                dtd_confidence=0.2
            )
        
                          
        dtd_start = time.time()
        try:
                  
            feature_extractor = AdvancedFeatureExtractor()
            features = feature_extractor.extract_features(text)
            
                   
            dtd_proba = self.dtd_detector.predict_proba([features])[0]
            dtd_ai_prob = dtd_proba[1]          
            dtd_confidence = max(dtd_proba)
            
            dtd_time = (time.time() - dtd_start) * 1000
            self.stats["avg_dtd_time"] = (self.stats["avg_dtd_time"] * self.stats["total_predictions"] + dtd_time) / (self.stats["total_predictions"] + 1)
            
        except Exception as e:
            logger.error(f"DTD prediction failed: {e}")
            dtd_ai_prob = 0.5
            dtd_confidence = 0.3
            dtd_time = 50
        
                       
        if dtd_confidence < self.confidence_low or dtd_confidence > self.confidence_high:
                                   
            self.stats["dtd_direct"] += 1
            total_time = (time.time() - total_start) * 1000
            
            prediction = "AI-generated" if dtd_ai_prob > 0.5 else "Human"
            return CascadeResult(
                prediction=prediction,
                ai_probability=dtd_ai_prob,
                confidence=dtd_confidence,
                stage_used="DTD",
                processing_time=total_time,
                dtd_confidence=dtd_confidence,
                efficiency_gain=1.0
            )
        
                          
        logger.info(f"DTD confidence {dtd_confidence:.3f} in range [{self.confidence_low}, {self.confidence_high}], using LRC")
        
        lrc_start = time.time()
        try:
            lrc_result = self.lrc_detector.predict(text)
            lrc_time = (time.time() - lrc_start) * 1000
            self.stats["avg_lrc_time"] = (self.stats["avg_lrc_time"] * self.stats["lrc_deep"] + lrc_time) / (self.stats["lrc_deep"] + 1)
            
            self.stats["lrc_deep"] += 1
            total_time = (time.time() - total_start) * 1000
            
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
            logger.error(f"LRC prediction failed: {e}")
                           
            total_time = (time.time() - total_start) * 1000
            prediction = "AI-generated" if dtd_ai_prob > 0.5 else "Human"
            
            return CascadeResult(
                prediction=prediction,
                ai_probability=dtd_ai_prob,
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
            "dtd_usage_rate": self.stats["dtd_direct"] / total,
            "lrc_usage_rate": self.stats["lrc_deep"] / total,
            "avg_dtd_time_ms": round(self.stats["avg_dtd_time"], 2),
            "avg_lrc_time_ms": round(self.stats["avg_lrc_time"], 2),
            "confidence_range": [self.confidence_low, self.confidence_high],
            "efficiency_summary": f"DTD: {self.stats['dtd_direct']}, LRC: {self.stats['lrc_deep']}"
        }
    
    def batch_predict(self, texts: List[str], max_workers: int = 4) -> List[CascadeResult]:
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def cleanup(self):
        if hasattr(self, 'lrc_detector'):
            self.lrc_detector.cleanup()

         
def create_simple_detector(dtd_model_path: str = None, 
                          confidence_low: float = 0.41,
                          confidence_high: float = 0.61) -> IntegratedDTDLRCCascade:
    if dtd_model_path is None:
                   
        possible_paths = [
            "../optimized_dtd_model.pkl",
            "optimized_dtd_model.pkl", 
            "models/optimized_dtd_model.pkl"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                dtd_model_path = path
                break
        
        if dtd_model_path is None:
            raise FileNotFoundError("Could not find DTD model file")
    
    return IntegratedDTDLRCCascade(
        dtd_model_path=dtd_model_path,
        confidence_low=confidence_low,
        confidence_high=confidence_high
    )

         
def test_cascade_system():
    print("=" * 60)
    print(" DTD-LRC cascade detection systemTest")
    print("=" * 60)
    
    try:
               
        detector = create_simple_detector()
        
              
        test_samples = [
            "Hello! How are you doing today? I just wanted to chat and see what's up.",
            "The implementation of artificial intelligence in modern computational systems requires sophisticated algorithmic frameworks and methodological approaches.",
            "I think the weather is nice today, but my personal opinion is that we should go outside and enjoy it.",
        ]
        
        print(f" Confidence threshold: [{detector.confidence_low}, {detector.confidence_high}]")
        print(f" Test samples: {len(test_samples)} ")
        print("\n" + "=" * 60)
        
        for i, text in enumerate(test_samples, 1):
            print(f"\n Sample {i}: {text[:50]}...")
            
            start_time = time.time()
            result = detector.predict(text)
            total_time = time.time() - start_time
            
            print(f" Prediction result: {result.prediction}")
            print(f" AI probability: {result.ai_probability:.3f}")
            print(f" Confidence: {result.confidence:.3f}")
            print(f" Stage used: {result.stage_used}")
            print(f" Processing time: {result.processing_time:.1f}ms")
            
            if result.dtd_confidence:
                print(f" DTD confidence: {result.dtd_confidence:.3f}")
        
        print("\n" + "=" * 60)
        stats = detector.get_statistics()
        print(" System statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n Testing completed. The DTD-LRC cascade system is running normally.")
        
              
        detector.cleanup()
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cascade_system()
