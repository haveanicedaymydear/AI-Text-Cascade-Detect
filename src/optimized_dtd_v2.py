                      

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time
import re
import pickle
from typing import List, Dict, Tuple
import warnings
from functools import lru_cache
import hashlib

warnings.filterwarnings('ignore')

class AdvancedFeatureExtractor:
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),              
            sublinear_tf=True
        )
        self.feature_cache = {}
        
    @lru_cache(maxsize=1000)
    def extract_linguistic_features(self, text_hash: str, text: str) -> Tuple[float, ...]:
        
              
        text_len = len(text)
        word_count = len(text.split())
        
        if word_count == 0:
            return tuple([0.0] * 20)          
        
        words = text.split()
        
                           
        avg_word_len = np.mean([len(word) for word in words])
        word_len_var = np.var([len(word) for word in words])
        long_word_ratio = sum(1 for word in words if len(word) > 7) / word_count
        unique_word_ratio = len(set(words)) / word_count
        
                    
        sentences = re.split(r'[.!?]', text)
        valid_sentences = [s.strip() for s in sentences if s.strip()]
        
        if valid_sentences:
            avg_sent_len = np.mean([len(sent.split()) for sent in valid_sentences])
            sent_len_var = np.var([len(sent.split()) for sent in valid_sentences])
            max_sent_len = max([len(sent.split()) for sent in valid_sentences])
        else:
            avg_sent_len = sent_len_var = max_sent_len = 0
        
                            
        punct_density = len(re.findall(r'[.!?]', text)) / word_count
        comma_density = text.count(',') / word_count
        colon_density = text.count(':') / word_count
        semicolon_density = text.count(';') / word_count
        
                           
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        bigram_repetition = 1 - len(set(bigrams)) / max(len(bigrams), 1)
        trigram_repetition = 1 - len(set(trigrams)) / max(len(trigrams), 1)
        
                            
        adj_like = sum(1 for word in words if word.endswith(('ing', 'ed', 'er', 'est'))) / word_count
        adv_like = sum(1 for word in words if word.endswith('ly')) / word_count
        
                   
        paragraph_count = text.count('\n\n') + 1
        avg_paragraph_len = word_count / paragraph_count
        
                   
        char_variety = len(set(text.lower())) / max(len(text), 1)
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        
                            
        academic_indicators = ['however', 'moreover', 'furthermore', 'therefore', 'consequently']
        academic_score = sum(text.lower().count(word) for word in academic_indicators) / word_count
        
        return (
            text_len, word_count, avg_word_len, word_len_var, long_word_ratio,
            unique_word_ratio, avg_sent_len, sent_len_var, max_sent_len,
            punct_density, comma_density, colon_density, semicolon_density,
            bigram_repetition, trigram_repetition, adj_like, adv_like,
            avg_paragraph_len, char_variety, digit_ratio, academic_score
        )
    
    def extract_features_batch(self, texts: List[str]) -> np.ndarray:
                  
        tfidf_features = self.vectorizer.transform(texts).toarray()
        
               
        linguistic_features = []
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            features = self.extract_linguistic_features(text_hash, text)
            linguistic_features.append(features)
        
              
        combined_features = np.hstack([tfidf_features, np.array(linguistic_features)])
        return combined_features

class UltraFastDTD:
    
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.classifier = None
        self.scaler = StandardScaler()
        self.model_pipeline = None
        
    def train_optimized(self, data_path: str, use_ensemble: bool = True) -> Dict:
        print("=== Optimized DTD training v2.0 ===")
        start_time = time.time()
        
                    
        texts, labels = self.load_balanced_data(data_path, max_per_class=800)
        
              
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )
        
                 
        print("Training feature extractor...")
        self.feature_extractor.vectorizer.fit(X_train)
        
                
        print("Extracting training features...")
        X_train_features = self.feature_extractor.extract_features_batch(X_train)
        X_test_features = self.feature_extractor.extract_features_batch(X_test)
        
               
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        X_test_scaled = self.scaler.transform(X_test_features)
        
              
        if use_ensemble:
            print("Training random forest ensemble model...")
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        else:
            print("Training logistic regression model...")
            self.classifier = LogisticRegression(
                max_iter=200,
                random_state=42,
                class_weight='balanced'
            )
        
              
        self.classifier.fit(X_train_scaled, y_train)
        
            
        y_pred = self.classifier.predict(X_test_scaled)
        y_prob = self.classifier.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
        
        training_time = time.time() - start_time
        
        print(f"\n=== Optimized training completed ===")
        print(f"Training time: {training_time:.2f} sec")
        print(f"AUC score: {auc_score:.4f}")
        print(f"Feature dimension: {X_train_features.shape[1]}")
        print("\nDetailed classification report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
        
        return {
            'training_time': training_time,
            'auc_score': auc_score,
            'n_features': X_train_features.shape[1],
            'model_type': 'RandomForest' if use_ensemble else 'LogisticRegression'
        }
    
    def load_balanced_data(self, data_path: str, max_per_class: int = 800) -> Tuple[List[str], List[int]]:
        print(f"Loading data: {data_path}")
        
        ai_texts, human_texts = [], []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(ai_texts) >= max_per_class and len(human_texts) >= max_per_class:
                    break
                
                try:
                    sample = json.loads(line.strip())
                    text = sample['text'][:2000]          
                    label = sample['label']
                    
                    if label == 1 and len(ai_texts) < max_per_class:
                        ai_texts.append(text)
                    elif label == 0 and len(human_texts) < max_per_class:
                        human_texts.append(text)
                except:
                    continue
        
        texts = ai_texts + human_texts
        labels = [1] * len(ai_texts) + [0] * len(human_texts)
        
        print(f"Data loading completed: AI={len(ai_texts)}, Human={len(human_texts)}")
        return texts, labels
    
    def predict_fast(self, text: str) -> Dict:
        if self.classifier is None:
            raise ValueError("Model not trained")
        
        start_time = time.time()
        
              
        features = self.feature_extractor.extract_features_batch([text])
        features_scaled = self.scaler.transform(features)
        
            
        prob = self.classifier.predict_proba(features_scaled)[0, 1]
        pred = self.classifier.predict(features_scaled)[0]
        
        prediction_time = time.time() - start_time
        
        return {
            'prediction': 'AI Generated' if pred == 1 else 'Human Written',
            'ai_probability': prob,
            'confidence': max(prob, 1-prob),
            'prediction_time': prediction_time * 1000      
        }
    
    def save_model(self, path: str):
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'vectorizer': self.feature_extractor.vectorizer
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved: {path}")
    
    def load_model(self, path: str):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.feature_extractor.vectorizer = model_data['vectorizer']
        print(f"Model loaded: {path}")

def benchmark_speed(dtd_model: UltraFastDTD, test_texts: List[str]) -> Dict:
    print("\n=== Performance benchmark ===")
    
             
    single_times = []
    for text in test_texts[:10]:
        result = dtd_model.predict_fast(text)
        single_times.append(result['prediction_time'])
    
    avg_single_time = np.mean(single_times)
    
            
    start_time = time.time()
    batch_results = [dtd_model.predict_fast(text) for text in test_texts[:50]]
    batch_time = (time.time() - start_time) * 1000      
    
    return {
        'avg_single_prediction_ms': avg_single_time,
        'batch_50_total_ms': batch_time,
        'throughput_per_second': 1000 / avg_single_time
    }

def main():
    print("=== Ultra-fast high-accuracy DTD system ===")
    
           
    dtd = UltraFastDTD()
    
          
    data_path = r"c:\Users\blc\Desktop\B\B2-DeepFake Text Detector (DTD)\M4GT-Bench\SubtaskA.jsonl"
    
          
    results = dtd.train_optimized(data_path, use_ensemble=True)
    
          
    model_path = r"c:\Users\blc\Desktop\B\B2-DeepFake Text Detector (DTD)\optimized_dtd_model.pkl"
    dtd.save_model(model_path)
    
          
    test_texts = [
        "This is a simple human-written sentence.",
        "The implementation of advanced machine learning algorithms necessitates comprehensive consideration of computational complexity factors and optimization strategies to ensure scalable deployment across distributed systems.",
        "I love spending time with my family on weekends.",
        "Furthermore, the systematic analysis of large-scale datasets requires sophisticated statistical methodologies and robust preprocessing pipelines to extract meaningful insights."
    ]
    
    performance_stats = benchmark_speed(dtd, test_texts)
    
    print(f"\n=== Performance statistics ===")
    print(f"Average prediction time: {performance_stats['avg_single_prediction_ms']:.2f}ms")
    print(f"Throughput: {performance_stats['throughput_per_second']:.0f} texts/sec")
    
          
    print("\n=== Prediction examples ===")
    for i, text in enumerate(test_texts):
        result = dtd.predict_fast(text)
        print(f"Text {i+1}: {result['prediction']} (Confidence: {result['confidence']:.3f}, time: {result['prediction_time']:.1f}ms)")
    
    return dtd, results, performance_stats

if __name__ == "__main__":
    system, training_results, perf_stats = main()
