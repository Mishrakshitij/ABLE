import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from able.evaluation import classification_metrics, evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def test_macro_f1_includes_absent_classes(self):
        metrics = classification_metrics([[2, 1, 0], [1, 2, 0], [0, 0, 0]])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 4 / 9)

    def test_prediction_id_validation_and_available_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps({"example_id": "1:2", "response": "Hello",
                                        "token_count": 1, "generated_nll": 0, "scored_tokens": 2}) + "\n")
            example = SimpleNamespace(example_id="1:2", conversation_id="1", turn_id=2)
            result = evaluate_predictions([example], path)
            self.assertEqual(result["token_perplexity"], 1)
            self.assertEqual(result["response_length_tokens"], 1)
            self.assertNotIn("classifier_agreement", result)
            with self.assertRaisesRegex(ValueError, "outside"):
                evaluate_predictions([], path)
            path.write_text(path.read_text() * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
                evaluate_predictions([example], path)

    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([[0, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
