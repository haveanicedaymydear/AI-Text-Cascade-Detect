import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse


@dataclass
class DetectionResult:
    """Container for a single detection result."""
    prediction: str
    ai_probability: float
    confidence: float
    stage_used: str
    processing_time: float
    dtd_confidence: Optional[float] = None
    lrc_features: Optional[Dict] = None


class DTDDetector:
    """Stage 1 detector: fast DTD screening."""
    
    def predict_proba(self, text: str) -> Tuple[float, float]:
        """Return probabilities in the form [Human probability, AI probability]."""
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        ai_score = 0.0
        
        # Lexical complexity
        if len(words) > 0:
            complex_words = sum(1 for w in words if len(w) > 8)
            complex_ratio = complex_words / len(words)
            if complex_ratio > 0.3:
                ai_score += 0.25
        
        # Sentence length regularity
        if len(sentences) > 1:
            word_counts = [len(s.split()) for s in sentences]
            avg_words = np.mean(word_counts)
            if 12 <= avg_words <= 22:
                ai_score += 0.20
        
        # Formal language indicators
        formal_words = [
            'implementation', 'methodology', 'framework', 'systematic',
            'furthermore', 'consequently', 'therefore', 'however', 'comprehensive'
        ]
        formal_count = sum(1 for word in formal_words if word in text.lower())
        if formal_count >= 2:
            ai_score += 0.15
        
        # Punctuation regularity
        punct_count = sum(1 for c in text if c in '.,!?;:')
        punct_density = punct_count / max(len(text), 1)
        if 0.02 <= punct_density <= 0.06:
            ai_score += 0.15
        
        # Repetition analysis
        word_set = set(w.lower() for w in words)
        repetition = 1.0 - (len(word_set) / max(len(words), 1))
        if repetition < 0.15:
            ai_score += 0.15
        
        # Uppercase usage pattern
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / max(len(text), 1)
        if uppercase_ratio < 0.03:
            ai_score += 0.10
        
        ai_probability = 0.1 + ai_score
        human_probability = 1.0 - ai_probability
        
        return human_probability, ai_probability


class LRCDetector:
    """Stage 2 detector: higher-precision analysis using LRC-style evidence."""
    
    def predict(self, text: str) -> Dict:
        """Run the LRC detector and return prediction outputs."""
        start_time = time.time()
        
        # Extract text features
        features = self._extract_features(text)
        
        # Simulate LRC response curve scoring
        lrc_scores = self._compute_lrc_scores(features)
        
        # Derive curve-based LRC features
        lrc_features = self._compute_lrc_features(lrc_scores)
        
        # Final LRC classification
        ai_probability = self._lrc_classify(lrc_features, features)
        
        prediction = "AI-generated" if ai_probability > 0.5 else "Human"
        confidence = abs(ai_probability - 0.5) * 2
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "prediction": prediction,
            "ai_probability": ai_probability,
            "confidence": confidence,
            "lrc_features": lrc_features,
            "processing_time": processing_time
        }
    
    def _extract_features(self, text: str) -> Dict:
        """Extract basic text analysis features."""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'vocabulary_diversity': len(set(words)) / max(len(words), 1),
            'technical_terms': sum(1 for w in words if len(w) > 10),
            'formal_indicators': sum(
                1 for w in ['however', 'therefore', 'furthermore', 'consequently']
                if w in text.lower()
            ),
            'punctuation_variety': len(set(c for c in text if c in '.,!?;:()')),
        }
    
    def _compute_lrc_scores(self, features: Dict) -> List[float]:
        """Compute simulated LRC response scores across model scales."""
        base_score = 1.8
        
        complexity_factor = features['avg_word_length'] * 0.1
        diversity_factor = features['vocabulary_diversity'] * 0.3
        technical_factor = features['technical_terms'] * 0.02
        formal_factor = features['formal_indicators'] * 0.1
        
        # Simulated scores for three model scales
        small_score = base_score + complexity_factor + diversity_factor + technical_factor + np.random.normal(0, 0.05)
        mid_score = small_score - 0.15 - formal_factor + np.random.normal(0, 0.03)
        large_score = mid_score - 0.12 + diversity_factor * 0.5 + np.random.normal(0, 0.02)
        
        return [small_score, mid_score, large_score]
    
    def _compute_lrc_features(self, scores: List[float]) -> Dict:
        """Compute curve-shape features from LRC scores."""
        s_small, s_mid, s_large = scores
        
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
    
    def _lrc_classify(self, lrc_features: Dict, text_features: Dict) -> float:
        """Apply the rule-based LRC classification logic."""
        ai_score = 0.0
        
        early_drop = lrc_features.get("early_drop", 0)
        late_drop = lrc_features.get("late_drop", 0)
        overall_drop = lrc_features.get("overall_drop", 0)
        drop_ratio = lrc_features.get("drop_ratio", 1.0)
        concavity = lrc_features.get("concavity_flag", 0)
        
        if overall_drop > 0.08:
            ai_score += 0.35
        
        if drop_ratio > 1.3:
            ai_score += 0.25
        elif drop_ratio < 0.8:
            ai_score += 0.10
        
        if concavity == 1:
            ai_score += 0.20
        
        # Auxiliary text features
        vocab_div = text_features.get('vocabulary_diversity', 0.5)
        if vocab_div > 0.75:
            ai_score += 0.15
        
        tech_terms = text_features.get('technical_terms', 0)
        if tech_terms >= 2:
            ai_score += 0.10
        
        formal_count = text_features.get('formal_indicators', 0)
        if formal_count >= 2:
            ai_score += 0.15
        
        return min(max(0.15 + ai_score, 0.0), 1.0)


class DTDLRCCascade:
    """Final DTD-LRC cascade system."""
    
    def __init__(self, confidence_low: float = 0.41, confidence_high: float = 0.61):
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        
        self.dtd_detector = DTDDetector()
        self.lrc_detector = LRCDetector()
        
        self.stats = {
            "total": 0,
            "dtd_direct": 0,
            "lrc_deep": 0,
            "dtd_time": [],
            "lrc_time": []
        }
    
    def detect(self, text: str) -> DetectionResult:
        """Main detection entry point."""
        total_start = time.time()
        
        if not text or len(text.strip()) < 3:
            return DetectionResult(
                prediction="Human",
                ai_probability=0.1,
                confidence=0.2,
                stage_used="DTD",
                processing_time=1.0
            )
        
        # Stage 1: DTD
        dtd_start = time.time()
        human_prob, ai_prob = self.dtd_detector.predict_proba(text)
        dtd_confidence = max(human_prob, ai_prob)
        dtd_time = (time.time() - dtd_start) * 1000
        
        # Gate decision
        if dtd_confidence < self.confidence_low or dtd_confidence > self.confidence_high:
            self.stats["dtd_direct"] += 1
            self.stats["dtd_time"].append(dtd_time)
            total_time = (time.time() - total_start) * 1000
            
            prediction = "AI-generated" if ai_prob > 0.5 else "Human"
            
            return DetectionResult(
                prediction=prediction,
                ai_probability=ai_prob,
                confidence=dtd_confidence,
                stage_used="DTD",
                processing_time=total_time,
                dtd_confidence=dtd_confidence
            )
        
        # Stage 2: LRC deep analysis
        lrc_start = time.time()
        lrc_result = self.lrc_detector.predict(text)
        lrc_time = (time.time() - lrc_start) * 1000
        
        self.stats["lrc_deep"] += 1
        self.stats["lrc_time"].append(lrc_time)
        total_time = (time.time() - total_start) * 1000
        
        return DetectionResult(
            prediction=lrc_result["prediction"],
            ai_probability=lrc_result["ai_probability"],
            confidence=lrc_result["confidence"],
            stage_used="LRC",
            processing_time=total_time,
            dtd_confidence=dtd_confidence,
            lrc_features=lrc_result.get("lrc_features", {})
        )
    
    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """Run batch detection."""
        results = []
        for text in texts:
            result = self.detect(text)
            results.append(result)
            self.stats["total"] += 1
        return results
    
    def get_stats(self) -> Dict:
        """Return summary statistics for the current session."""
        total = self.stats["total"]
        if total == 0:
            return {"message": "No detections performed yet"}
        
        return {
            "total_detections": total,
            "dtd_direct": self.stats["dtd_direct"],
            "lrc_deep": self.stats["lrc_deep"],
            "dtd_rate": f"{self.stats['dtd_direct'] / total * 100:.1f}%",
            "lrc_rate": f"{self.stats['lrc_deep'] / total * 100:.1f}%",
            "avg_dtd_time": f"{np.mean(self.stats['dtd_time']):.1f}ms" if self.stats["dtd_time"] else "N/A",
            "avg_lrc_time": f"{np.mean(self.stats['lrc_time']):.1f}ms" if self.stats["lrc_time"] else "N/A",
            "thresholds": f"[{self.confidence_low}, {self.confidence_high}]"
        }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="DTD-LRC Cascade Detection System")
    parser.add_argument("--text", "-t", type=str, help="Text to detect")
    parser.add_argument("--file", "-f", type=str, help="Path to a text file")
    parser.add_argument("--batch", "-b", type=str, help="Batch text file (one text per line)")
    parser.add_argument("--low", type=float, default=0.41, help="Lower confidence threshold")
    parser.add_argument("--high", type=float, default=0.61, help="Upper confidence threshold")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--stats", "-s", action="store_true", help="Show summary statistics")
    
    args = parser.parse_args()
    
    detector = DTDLRCCascade(confidence_low=args.low, confidence_high=args.high)
    
    print("=" * 60)
    print("DTD-LRC Cascade Detection System")
    print("=" * 60)
    print(f"Confidence threshold band: [{args.low}, {args.high}]")
    print()
    
    if args.text:
        result = detector.detect(args.text)
        print_result(args.text, result)
        
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            result = detector.detect(text)
            print_result(f"File: {args.file}", result)
        except Exception as e:
            print(f"Failed to read file: {e}")
            
    elif args.batch:
        try:
            with open(args.batch, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Running batch detection on {len(texts)} texts...")
            results = detector.detect_batch(texts)
            
            for i, (text, result) in enumerate(zip(texts, results), 1):
                print(f"\nText {i}: {text[:40]}...")
                print_result_brief(result)
                
        except Exception as e:
            print(f"Failed to read batch file: {e}")
            
    elif args.interactive:
        print("Interactive mode (type 'quit' to exit)")
        while True:
            try:
                text = input("\nEnter text to detect: ").strip()
                if text.lower() in ['quit', 'exit', '退出']:
                    break
                if not text:
                    continue
                    
                result = detector.detect(text)
                print_result(text, result)
                
            except KeyboardInterrupt:
                break
                
    else:
        demo_texts = [
            "Hey, the weather is nice today. Where do you think we should go?",
            "The comprehensive implementation of machine learning algorithms requires systematic optimization of computational resources.",
            "I think this problem is interesting, but I am still not fully sure about the best solution."
        ]
        
        print("Running demo...")
        for i, text in enumerate(demo_texts, 1):
            print(f"\nDemo {i}: {text}")
            result = detector.detect(text)
            print_result_brief(result)
    
    if args.stats:
        print("\n" + "=" * 60)
        print("System statistics:")
        stats = detector.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")


def print_result(text: str, result: DetectionResult):
    """Print a detailed detection result."""
    print(f"Text: {text[:50]}..." if len(text) > 50 else f"Text: {text}")
    print(f"Prediction: {result.prediction}")
    print(f"AI probability: {result.ai_probability:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Stage used: {result.stage_used}")
    print(f"Processing time: {result.processing_time:.1f}ms")
    if result.dtd_confidence:
        print(f"DTD confidence: {result.dtd_confidence:.3f}")


def print_result_brief(result: DetectionResult):
    """Print a brief detection result."""
    print(
        f"{result.prediction} | "
        f"AI probability: {result.ai_probability:.3f} | "
        f"Stage: {result.stage_used} | "
        f"Time: {result.processing_time:.0f}ms"
    )


if __name__ == "__main__":
    main()