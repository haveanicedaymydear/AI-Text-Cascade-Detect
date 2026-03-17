#!/usr/bin/env python3
"""
Comprehensive test suite for the DTD system.
Includes edge cases, exception handling, and performance stress tests.
"""

import os
import time
import json
import warnings
import numpy as np
from typing import Dict
from optimized_dtd_v2 import UltraFastDTD

warnings.filterwarnings("ignore")


class DTDTestSuite:
    """Comprehensive test suite for the DTD system."""

    def __init__(self):
        self.dtd_model = UltraFastDTD()
        self.test_results = {}
        self.load_model()

    def load_model(self):
        """Load the trained model."""
        model_path = r"c:\Users\blc\Desktop\B\B2-DeepFake Text Detector (DTD)\optimized_dtd_model.pkl"
        if os.path.exists(model_path):
            self.dtd_model.load_model(model_path)
            print("Model loaded successfully")
        else:
            print("Model file not found. Please train the model first.")
            return False
        return True

    def test_edge_cases(self) -> Dict:
        """Run edge case tests."""
        print("\n=== Edge Case Tests ===")

        edge_cases = {
            "minimum_length": "Short text.",
            "maximum_length": "a" * 9999,
            "only_punctuation": "!!! ??? ... ;;; ::: ,,, --- ___ *** ### @@@ %%%",
            "only_numbers": "123456789 987654321 1234567890 0987654321 5555555555",
            "mixed_languages": "Hello world 你好世界 Hola mundo Bonjour monde こんにちは世界",
            "special_characters": "[@#$%^&*()_+-={}[]|\\:;\"'<>?,./~`]",
            "repeated_words": "the the the the the the the the the the",
            "single_word": "supercalifragilisticexpialidocious",
            "scientific_notation": "1.23e-45 9.87E+123 6.022e23 3.14159265359",
            "urls_and_emails": "https://example.com user@domain.com www.test.org",
            "very_long_words": "pneumonoultramicroscopicsilicovolcanoconiosispneumonoultramicroscopicsilicovolcanoconiosis",
            "unicode_symbols": "α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω",
            "newlines_tabs": "Line 1\nLine 2\tTab separated\r\nCarriage return",
            "empty_sentences": ". . . . . . . . . .",
            "code_like": "def function(x): return x * 2 if x > 0 else 0",
        }

        results = {}
        for case_name, text in edge_cases.items():
            try:
                start_time = time.time()
                result = self.dtd_model.predict_fast(text)
                execution_time = (time.time() - start_time) * 1000

                results[case_name] = {
                    "success": True,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "ai_probability": result["ai_probability"],
                    "execution_time_ms": execution_time,
                    "text_length": len(text),
                }
                print(f"{case_name}: {result['prediction']} (confidence: {result['confidence']:.3f})")

            except Exception as e:
                results[case_name] = {
                    "success": False,
                    "error": str(e),
                    "text_length": len(text),
                }
                print(f"{case_name}: Error - {e}")

        self.test_results["edge_cases"] = results
        return results

    def test_performance_stress(self) -> Dict:
        """Run performance stress tests."""
        print("\n=== Performance Stress Tests ===")

        length_tests = [50, 100, 500, 1000, 2000, 5000]
        base_text = "This is a sample text for performance testing. " * 20

        performance_results = {}

        for target_length in length_tests:
            if target_length <= len(base_text):
                test_text = base_text[:target_length]
            else:
                repeat_count = (target_length // len(base_text)) + 1
                test_text = (base_text * repeat_count)[:target_length]

            times = []
            predictions = []

            for _ in range(5):
                try:
                    start_time = time.time()
                    result = self.dtd_model.predict_fast(test_text)
                    execution_time = (time.time() - start_time) * 1000

                    times.append(execution_time)
                    predictions.append(result["ai_probability"])

                except Exception as e:
                    print(f"Length {target_length} test failed: {e}")
                    continue

            if times:
                performance_results[f"length_{target_length}"] = {
                    "avg_time_ms": np.mean(times),
                    "min_time_ms": np.min(times),
                    "max_time_ms": np.max(times),
                    "std_time_ms": np.std(times),
                    "avg_ai_prob": np.mean(predictions),
                    "prediction_variance": np.var(predictions),
                }

                print(
                    f"Length {target_length}: average {np.mean(times):.2f}ms, "
                    f"std {np.std(times):.2f}ms"
                )

        batch_sizes = [1, 5, 10, 20]
        batch_test_text = "This is a batch processing test. " * 10

        batch_results = {}
        for batch_size in batch_sizes:
            texts = [batch_test_text] * batch_size

            start_time = time.time()
            batch_predictions = []

            try:
                for text in texts:
                    result = self.dtd_model.predict_fast(text)
                    batch_predictions.append(result)

                total_time = (time.time() - start_time) * 1000
                avg_time_per_text = total_time / batch_size

                batch_results[f"batch_{batch_size}"] = {
                    "total_time_ms": total_time,
                    "avg_time_per_text_ms": avg_time_per_text,
                    "throughput_per_second": 1000 / avg_time_per_text,
                    "success_count": len(batch_predictions),
                }

                print(
                    f"Batch {batch_size}: total {total_time:.2f}ms, "
                    f"average per text {avg_time_per_text:.2f}ms"
                )

            except Exception as e:
                batch_results[f"batch_{batch_size}"] = {
                    "error": str(e)
                }
                print(f"Batch {batch_size} test failed: {e}")

        performance_results.update(batch_results)
        self.test_results["performance_stress"] = performance_results
        return performance_results

    def test_robustness(self) -> Dict:
        """Run robustness tests."""
        print("\n=== Robustness Tests ===")

        robustness_texts = {
            "academic_paper": """
            The systematic analysis of large-scale neural network architectures
            demonstrates that transformer-based models exhibit superior performance
            across multiple natural language processing tasks, particularly in
            contexts requiring long-range dependency modeling and contextual understanding.
            """,

            "casual_conversation": """
            Hey! So like, I was thinking about what you said yesterday and honestly
            I'm not really sure if that's the best approach? Maybe we should try
            something different, you know? What do you think?
            """,

            "news_article": """
            Recent developments in artificial intelligence have sparked significant
            debate among researchers and policymakers. The technology's rapid advancement
            raises important questions about regulation, ethics, and societal impact.
            """,

            "technical_documentation": """
            To initialize the system, first ensure all dependencies are installed.
            Run the configuration script with appropriate parameters. Monitor the
            output logs for any error messages. If issues occur, consult the troubleshooting guide.
            """,

            "creative_writing": """
            The moonlight danced across the rippling water, casting silver shadows
            that seemed to whisper secrets of the night. In the distance, an owl's
            call echoed through the stillness, haunting and beautiful.
            """,

            "scientific_abstract": """
            We present a novel approach for optimizing distributed computing systems
            through adaptive resource allocation algorithms. Our methodology achieves
            significant improvements in throughput while reducing latency by 23% compared
            to existing baseline implementations.
            """,
        }

        robustness_results = {}
        for text_type, content in robustness_texts.items():
            try:
                original_result = self.dtd_model.predict_fast(content.strip())

                noisy_content = content.replace(".", "...").replace(",", " ,")
                noisy_result = self.dtd_model.predict_fast(noisy_content.strip())

                case_content = content.upper()
                case_result = self.dtd_model.predict_fast(case_content.strip())

                robustness_results[text_type] = {
                    "original": {
                        "prediction": original_result["prediction"],
                        "confidence": original_result["confidence"],
                        "ai_probability": original_result["ai_probability"],
                    },
                    "noisy": {
                        "prediction": noisy_result["prediction"],
                        "confidence": noisy_result["confidence"],
                        "ai_probability": noisy_result["ai_probability"],
                    },
                    "uppercase": {
                        "prediction": case_result["prediction"],
                        "confidence": case_result["confidence"],
                        "ai_probability": case_result["ai_probability"],
                    },
                    "consistency": {
                        "original_vs_noisy_diff": abs(
                            original_result["ai_probability"] - noisy_result["ai_probability"]
                        ),
                        "original_vs_case_diff": abs(
                            original_result["ai_probability"] - case_result["ai_probability"]
                        ),
                    },
                }

                print(
                    f"{text_type}: original {original_result['prediction']}, "
                    f"noisy {noisy_result['prediction']}, uppercase {case_result['prediction']}"
                )

            except Exception as e:
                robustness_results[text_type] = {"error": str(e)}
                print(f"{text_type} robustness test failed: {e}")

        self.test_results["robustness"] = robustness_results
        return robustness_results

    def run_all_tests(self) -> Dict:
        """Run the full test suite."""
        print("=== DTD Comprehensive Test Suite Started ===")
        start_time = time.time()

        edge_results = self.test_edge_cases()
        performance_results = self.test_performance_stress()
        robustness_results = self.test_robustness()

        total_time = time.time() - start_time

        report = {
            "test_summary": {
                "total_test_time_seconds": total_time,
                "edge_cases_tested": len(edge_results),
                "edge_cases_passed": sum(1 for r in edge_results.values() if r.get("success", False)),
                "performance_metrics_collected": len(performance_results),
                "robustness_scenarios_tested": len(robustness_results),
                "timestamp": time.time(),
            },
            "detailed_results": {
                "edge_cases": edge_results,
                "performance_stress": performance_results,
                "robustness": robustness_results,
            },
        }

        report_path = r"c:\Users\blc\Desktop\B\B2-DeepFake Text Detector (DTD)\test_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n=== Testing Completed ===")
        print(f"Total test time: {total_time:.2f} seconds")
        print(
            f"Edge case pass rate: "
            f"{report['test_summary']['edge_cases_passed']}/"
            f"{report['test_summary']['edge_cases_tested']}"
        )
        print("Detailed report saved: test_report.json")

        return report


def main():
    """Main test entry point."""
    test_suite = DTDTestSuite()
    results = test_suite.run_all_tests()
    return results


if __name__ == "__main__":
    test_results = main()