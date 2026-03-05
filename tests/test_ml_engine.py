"""
Model XP — Machine Learning Engine Test Suite
Authored by Nicholas Michael Grossi

Tests all three classifiers (Decision Tree, Naive Bayes, k-NN)
and the unified MLEngine interface.
"""

import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_xp.ml_engine import (
    DecisionTreeClassifier,
    NaiveBayesClassifier,
    KNNClassifier,
    MLEngine,
    _euclidean_distance,
    _entropy,
    _information_gain
)

# ---------------------------------------------------------------------------
# Shared test dataset: Iris-like binary classification
# Features: [sepal_length, sepal_width, petal_length, petal_width]
# Labels: "class_a" or "class_b"
# ---------------------------------------------------------------------------
TRAIN_X = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8],
    [6.5, 3.0, 5.8, 2.2],
]
TRAIN_Y = [
    "class_a", "class_a", "class_a", "class_a", "class_a",
    "class_b", "class_b", "class_b", "class_b", "class_b"
]
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------

class TestUtilities(unittest.TestCase):

    def test_euclidean_distance_zero(self):
        self.assertAlmostEqual(_euclidean_distance([0, 0], [0, 0]), 0.0)

    def test_euclidean_distance_known(self):
        # Distance between (0,0) and (3,4) = 5.0
        self.assertAlmostEqual(_euclidean_distance([0, 0], [3, 4]), 5.0)

    def test_euclidean_distance_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _euclidean_distance([1, 2], [1, 2, 3])

    def test_entropy_uniform(self):
        # Two equal classes: entropy = 1.0 bit
        labels = ["a", "a", "b", "b"]
        self.assertAlmostEqual(_entropy(labels), 1.0)

    def test_entropy_pure(self):
        # All same class: entropy = 0.0
        labels = ["a", "a", "a"]
        self.assertAlmostEqual(_entropy(labels), 0.0)

    def test_entropy_empty(self):
        self.assertAlmostEqual(_entropy([]), 0.0)

    def test_information_gain_perfect_split(self):
        parent = ["a", "a", "b", "b"]
        children = [["a", "a"], ["b", "b"]]
        gain = _information_gain(parent, children)
        self.assertAlmostEqual(gain, 1.0)

    def test_information_gain_no_split(self):
        parent = ["a", "a", "b", "b"]
        children = [["a", "b"], ["a", "b"]]
        gain = _information_gain(parent, children)
        self.assertAlmostEqual(gain, 0.0)


# ---------------------------------------------------------------------------
# Decision Tree Tests
# ---------------------------------------------------------------------------

class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        self.model = DecisionTreeClassifier(max_depth=5)
        self.report = self.model.train(TRAIN_X, TRAIN_Y, feature_names=FEATURE_NAMES)

    def test_training_report_structure(self):
        self.assertEqual(self.report["status"], "trained")
        self.assertIn("algorithm", self.report)
        self.assertEqual(self.report["samples"], 10)
        self.assertEqual(self.report["features"], 4)

    def test_predict_class_a(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertEqual(result["prediction"], "class_a")

    def test_predict_class_b(self):
        result = self.model.predict([6.5, 3.0, 5.8, 2.2])
        self.assertEqual(result["prediction"], "class_b")

    def test_prediction_has_trace(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertIn("trace", result)
        self.assertIsInstance(result["trace"], list)
        self.assertGreater(len(result["trace"]), 0)

    def test_prediction_has_algorithm(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertIn("Decision Tree", result["algorithm"])

    def test_determinism(self):
        r1 = self.model.predict([5.0, 3.5, 1.4, 0.2])
        r2 = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertEqual(r1["prediction"], r2["prediction"])
        self.assertEqual(r1["trace"], r2["trace"])

    def test_untrained_raises(self):
        model = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            model.predict([1.0, 2.0, 3.0, 4.0])

    def test_empty_training_data_raises(self):
        model = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            model.train([], [])

    def test_is_trained(self):
        self.assertTrue(self.model.is_trained())

    def test_is_not_trained(self):
        model = DecisionTreeClassifier()
        self.assertFalse(model.is_trained())


# ---------------------------------------------------------------------------
# Naive Bayes Tests
# ---------------------------------------------------------------------------

class TestNaiveBayes(unittest.TestCase):

    def setUp(self):
        self.model = NaiveBayesClassifier()
        self.report = self.model.train(TRAIN_X, TRAIN_Y, feature_names=FEATURE_NAMES)

    def test_training_report_structure(self):
        self.assertEqual(self.report["status"], "trained")
        self.assertIn("algorithm", self.report)
        self.assertIn("class_priors", self.report)
        self.assertAlmostEqual(self.report["class_priors"]["class_a"], 0.5)
        self.assertAlmostEqual(self.report["class_priors"]["class_b"], 0.5)

    def test_predict_class_a(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertEqual(result["prediction"], "class_a")

    def test_predict_class_b(self):
        result = self.model.predict([6.5, 3.0, 5.8, 2.2])
        self.assertEqual(result["prediction"], "class_b")

    def test_prediction_has_log_probabilities(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertIn("log_probabilities", result)
        self.assertIn("class_a", result["log_probabilities"])
        self.assertIn("class_b", result["log_probabilities"])

    def test_prediction_has_trace(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertIn("trace", result)
        self.assertGreater(len(result["trace"]), 0)

    def test_determinism(self):
        r1 = self.model.predict([5.0, 3.5, 1.4, 0.2])
        r2 = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertEqual(r1["prediction"], r2["prediction"])
        self.assertEqual(r1["log_probabilities"], r2["log_probabilities"])

    def test_untrained_raises(self):
        model = NaiveBayesClassifier()
        with self.assertRaises(RuntimeError):
            model.predict([1.0, 2.0, 3.0, 4.0])

    def test_is_trained(self):
        self.assertTrue(self.model.is_trained())


# ---------------------------------------------------------------------------
# k-NN Tests
# ---------------------------------------------------------------------------

class TestKNN(unittest.TestCase):

    def setUp(self):
        self.model = KNNClassifier(k=3)
        self.report = self.model.train(TRAIN_X, TRAIN_Y, feature_names=FEATURE_NAMES)

    def test_training_report_structure(self):
        self.assertEqual(self.report["status"], "trained")
        self.assertEqual(self.report["k"], 3)
        self.assertEqual(self.report["samples"], 10)

    def test_predict_class_a(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertEqual(result["prediction"], "class_a")

    def test_predict_class_b(self):
        result = self.model.predict([6.5, 3.0, 5.8, 2.2])
        self.assertEqual(result["prediction"], "class_b")

    def test_prediction_has_neighbors(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertIn("k_nearest_neighbors", result)
        self.assertEqual(len(result["k_nearest_neighbors"]), 3)

    def test_prediction_has_vote_counts(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertIn("vote_counts", result)

    def test_prediction_has_trace(self):
        result = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertIn("trace", result)
        self.assertGreater(len(result["trace"]), 0)

    def test_determinism(self):
        r1 = self.model.predict([5.0, 3.5, 1.4, 0.2])
        r2 = self.model.predict([5.0, 3.5, 1.4, 0.2])
        self.assertEqual(r1["prediction"], r2["prediction"])
        self.assertEqual(r1["k_nearest_neighbors"], r2["k_nearest_neighbors"])

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)

    def test_untrained_raises(self):
        model = KNNClassifier(k=3)
        with self.assertRaises(RuntimeError):
            model.predict([1.0, 2.0, 3.0, 4.0])

    def test_is_trained(self):
        self.assertTrue(self.model.is_trained())


# ---------------------------------------------------------------------------
# MLEngine Unified Interface Tests
# ---------------------------------------------------------------------------

class TestMLEngine(unittest.TestCase):

    def setUp(self):
        self.engine = MLEngine()

    def test_train_decision_tree(self):
        report = self.engine.train("dt_model", "decision_tree", TRAIN_X, TRAIN_Y,
                                   feature_names=FEATURE_NAMES)
        self.assertEqual(report["status"], "trained")
        self.assertEqual(report["model_id"], "dt_model")

    def test_train_naive_bayes(self):
        report = self.engine.train("nb_model", "naive_bayes", TRAIN_X, TRAIN_Y)
        self.assertEqual(report["status"], "trained")
        self.assertEqual(report["model_id"], "nb_model")

    def test_train_knn(self):
        report = self.engine.train("knn_model", "knn", TRAIN_X, TRAIN_Y, k=3)
        self.assertEqual(report["status"], "trained")
        self.assertEqual(report["model_id"], "knn_model")

    def test_predict_decision_tree(self):
        self.engine.train("dt_model", "decision_tree", TRAIN_X, TRAIN_Y)
        result = self.engine.predict("dt_model", [5.0, 3.5, 1.4, 0.2])
        self.assertEqual(result["prediction"], "class_a")
        self.assertEqual(result["model_id"], "dt_model")

    def test_predict_naive_bayes(self):
        self.engine.train("nb_model", "naive_bayes", TRAIN_X, TRAIN_Y)
        result = self.engine.predict("nb_model", [5.0, 3.5, 1.4, 0.2])
        self.assertEqual(result["prediction"], "class_a")

    def test_predict_knn(self):
        self.engine.train("knn_model", "knn", TRAIN_X, TRAIN_Y, k=3)
        result = self.engine.predict("knn_model", [5.0, 3.5, 1.4, 0.2])
        self.assertEqual(result["prediction"], "class_a")

    def test_list_models(self):
        self.engine.train("dt_model", "decision_tree", TRAIN_X, TRAIN_Y)
        self.engine.train("nb_model", "naive_bayes", TRAIN_X, TRAIN_Y)
        models = self.engine.list_models()
        self.assertIn("dt_model", models)
        self.assertIn("nb_model", models)

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError):
            self.engine.predict("nonexistent_model", [1.0, 2.0, 3.0, 4.0])

    def test_unsupported_algorithm_raises(self):
        with self.assertRaises(ValueError):
            self.engine.train("bad_model", "neural_network", TRAIN_X, TRAIN_Y)

    def test_training_log_populated(self):
        self.engine.train("dt_model", "decision_tree", TRAIN_X, TRAIN_Y)
        log = self.engine.get_training_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["model_id"], "dt_model")

    def test_all_three_algorithms_agree_on_class_a(self):
        """All three algorithms must classify a clear class_a sample as class_a."""
        self.engine.train("dt", "decision_tree", TRAIN_X, TRAIN_Y)
        self.engine.train("nb", "naive_bayes", TRAIN_X, TRAIN_Y)
        self.engine.train("knn", "knn", TRAIN_X, TRAIN_Y, k=3)
        x = [5.0, 3.5, 1.4, 0.2]
        self.assertEqual(self.engine.predict("dt", x)["prediction"], "class_a")
        self.assertEqual(self.engine.predict("nb", x)["prediction"], "class_a")
        self.assertEqual(self.engine.predict("knn", x)["prediction"], "class_a")

    def test_all_three_algorithms_agree_on_class_b(self):
        """All three algorithms must classify a clear class_b sample as class_b."""
        self.engine.train("dt", "decision_tree", TRAIN_X, TRAIN_Y)
        self.engine.train("nb", "naive_bayes", TRAIN_X, TRAIN_Y)
        self.engine.train("knn", "knn", TRAIN_X, TRAIN_Y, k=3)
        x = [6.5, 3.0, 5.8, 2.2]
        self.assertEqual(self.engine.predict("dt", x)["prediction"], "class_b")
        self.assertEqual(self.engine.predict("nb", x)["prediction"], "class_b")
        self.assertEqual(self.engine.predict("knn", x)["prediction"], "class_b")

    def test_save_knn_model(self):
        import json
        import tempfile
        self.engine.train("knn_save", "knn", TRAIN_X, TRAIN_Y, k=3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        self.engine.save_model("knn_save", path)
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["algorithm"], "knn")
        self.assertEqual(data["k"], 3)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
