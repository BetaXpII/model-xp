"""
Model XP — Machine Learning Engine
Authored by Nicholas Michael Grossi

Provides deterministic, interpretable machine learning classifiers
that operate without external dependencies and without neural networks.

All algorithms are founded on mathematical principles:
  - Decision Tree: Information gain via entropy (Shannon, 1948)
  - Naive Bayes:   Conditional probability (Bayes, 1763)
  - k-NN:          Euclidean distance metric (Euclid, ~300 BC)

Every model produces an auditable decision trace.
No probabilistic sampling. No stochastic gradient descent.
No GPU required. Runs on any hardware.
"""

import math
import json
import os
import datetime
from collections import Counter, defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def _euclidean_distance(a: list, b: list) -> float:
    """Compute the Euclidean distance between two numeric vectors."""
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}."
        )
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _entropy(labels: list) -> float:
    """Compute the Shannon entropy of a label distribution."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum(
        (c / n) * math.log2(c / n)
        for c in counts.values()
        if c > 0
    )


def _information_gain(parent_labels: list, child_groups: list) -> float:
    """Compute information gain from splitting parent labels into child groups."""
    n = len(parent_labels)
    if n == 0:
        return 0.0
    parent_entropy = _entropy(parent_labels)
    weighted_child_entropy = sum(
        (len(group) / n) * _entropy(group)
        for group in child_groups
    )
    return parent_entropy - weighted_child_entropy


# ---------------------------------------------------------------------------
# Decision Tree Classifier
# ---------------------------------------------------------------------------

class DecisionTreeNode:
    """A single node in a Decision Tree."""

    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        label: Optional[str] = None,
        left=None,
        right=None,
        gain: float = 0.0
    ):
        self.feature_index = feature_index   # Feature used for splitting
        self.threshold = threshold            # Split threshold value
        self.label = label                    # Leaf node class label
        self.left = left                      # Left subtree (feature <= threshold)
        self.right = right                    # Right subtree (feature > threshold)
        self.gain = gain                      # Information gain at this split


class DecisionTreeClassifier:
    """
    A deterministic Decision Tree Classifier.

    Builds a binary decision tree using information gain (entropy reduction)
    as the split criterion. All splits are deterministic given the same data.

    Parameters:
        max_depth (int): Maximum tree depth. Prevents overfitting.
        min_samples_split (int): Minimum samples required to split a node.
    """

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[DecisionTreeNode] = None
        self._feature_names: list = []
        self._trained: bool = False
        self._training_samples: int = 0
        self._class_labels: list = []

    def train(self, X: list, y: list, feature_names: Optional[list] = None) -> dict:
        """
        Train the decision tree on labeled data.

        Parameters:
            X: List of feature vectors (list of lists of numbers).
            y: List of class labels (strings or integers).
            feature_names: Optional list of feature names for interpretability.

        Returns:
            A training report dict.
        """
        if not X or not y:
            raise ValueError("Training data must not be empty.")
        if len(X) != len(y):
            raise ValueError("Feature matrix and label vector must have equal length.")

        self._feature_names = feature_names or [f"feature_{i}" for i in range(len(X[0]))]
        self._class_labels = sorted(set(y))
        self._training_samples = len(X)
        self.root = self._build(X, y, depth=0)
        self._trained = True

        return {
            "status": "trained",
            "algorithm": "Decision Tree (Information Gain / Shannon Entropy)",
            "samples": len(X),
            "features": len(X[0]),
            "classes": self._class_labels,
            "max_depth": self.max_depth,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }

    def predict(self, x: list) -> dict:
        """
        Predict the class label for a single feature vector.

        Returns a dict with the predicted label and a full decision trace.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        trace = []
        label = self._traverse(self.root, x, trace)
        return {
            "prediction": label,
            "algorithm": "Decision Tree",
            "trace": trace
        }

    def _build(self, X: list, y: list, depth: int) -> DecisionTreeNode:
        """Recursively build the decision tree."""
        # Stopping conditions
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(set(y)) == 1:
            majority = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(label=majority)

        best_gain, best_feature, best_threshold = 0.0, 0, 0.0
        n_features = len(X[0])

        for fi in range(n_features):
            values = sorted(set(row[fi] for row in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2.0
                left_y = [y[j] for j in range(len(X)) if X[j][fi] <= threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][fi] > threshold]
                if not left_y or not right_y:
                    continue
                gain = _information_gain(y, [left_y, right_y])
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, fi, threshold

        if best_gain == 0.0:
            majority = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(label=majority)

        left_mask = [X[j][best_feature] <= best_threshold for j in range(len(X))]
        left_X = [X[j] for j in range(len(X)) if left_mask[j]]
        left_y = [y[j] for j in range(len(X)) if left_mask[j]]
        right_X = [X[j] for j in range(len(X)) if not left_mask[j]]
        right_y = [y[j] for j in range(len(X)) if not left_mask[j]]

        left_node = self._build(left_X, left_y, depth + 1)
        right_node = self._build(right_X, right_y, depth + 1)

        return DecisionTreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
            gain=best_gain
        )

    def _traverse(self, node: DecisionTreeNode, x: list, trace: list) -> str:
        """Traverse the tree to produce a prediction and decision trace."""
        if node.label is not None:
            trace.append(f"Leaf node reached. Predicted class: '{node.label}'.")
            return node.label

        fname = (
            self._feature_names[node.feature_index]
            if node.feature_index < len(self._feature_names)
            else f"feature_{node.feature_index}"
        )
        value = x[node.feature_index]

        if value <= node.threshold:
            trace.append(
                f"Split: {fname} = {value} <= {node.threshold:.4f} "
                f"(gain={node.gain:.4f}) → LEFT"
            )
            return self._traverse(node.left, x, trace)
        else:
            trace.append(
                f"Split: {fname} = {value} > {node.threshold:.4f} "
                f"(gain={node.gain:.4f}) → RIGHT"
            )
            return self._traverse(node.right, x, trace)

    def is_trained(self) -> bool:
        return self._trained


# ---------------------------------------------------------------------------
# Naive Bayes Classifier
# ---------------------------------------------------------------------------

class NaiveBayesClassifier:
    """
    A deterministic Gaussian Naive Bayes Classifier.

    Computes class-conditional probability distributions from training data
    and applies Bayes' theorem for classification. All computations are
    deterministic given the same training data.

    Suitable for continuous numeric features.
    """

    def __init__(self):
        self._class_priors: dict = {}
        self._class_stats: dict = {}   # {class: [{mean, variance}, ...]}
        self._trained: bool = False
        self._class_labels: list = []
        self._training_samples: int = 0

    def train(self, X: list, y: list, feature_names: Optional[list] = None) -> dict:
        """
        Train the Naive Bayes classifier on labeled data.

        Parameters:
            X: List of feature vectors.
            y: List of class labels.

        Returns:
            A training report dict.
        """
        if not X or not y:
            raise ValueError("Training data must not be empty.")

        self._feature_names = feature_names or [f"feature_{i}" for i in range(len(X[0]))]
        self._class_labels = sorted(set(y))
        self._training_samples = len(X)
        n = len(y)

        # Separate samples by class
        class_samples = defaultdict(list)
        for xi, yi in zip(X, y):
            class_samples[yi].append(xi)

        # Compute priors and per-feature statistics
        for cls, samples in class_samples.items():
            self._class_priors[cls] = len(samples) / n
            n_features = len(samples[0])
            stats = []
            for fi in range(n_features):
                values = [s[fi] for s in samples]
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                variance = max(variance, 1e-9)  # Prevent division by zero
                stats.append({"mean": mean, "variance": variance})
            self._class_stats[cls] = stats

        self._trained = True

        return {
            "status": "trained",
            "algorithm": "Naive Bayes (Gaussian, Bayes Theorem)",
            "samples": n,
            "features": len(X[0]),
            "classes": self._class_labels,
            "class_priors": self._class_priors,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }

    def predict(self, x: list) -> dict:
        """
        Predict the class label for a single feature vector.

        Returns a dict with the predicted label, class log-probabilities, and proof.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        log_probs = {}
        for cls in self._class_labels:
            log_prob = math.log(self._class_priors[cls])
            for fi, val in enumerate(x):
                stat = self._class_stats[cls][fi]
                mean, var = stat["mean"], stat["variance"]
                # Gaussian log-likelihood
                log_prob += (
                    -0.5 * math.log(2 * math.pi * var)
                    - ((val - mean) ** 2) / (2 * var)
                )
            log_probs[cls] = log_prob

        predicted = max(log_probs, key=log_probs.get)

        return {
            "prediction": predicted,
            "algorithm": "Naive Bayes (Gaussian)",
            "log_probabilities": {k: round(v, 6) for k, v in log_probs.items()},
            "trace": [
                f"Prior probabilities: {self._class_priors}",
                f"Log-posterior computed for each class using Gaussian likelihood.",
                f"Predicted class: '{predicted}' (highest log-posterior = {log_probs[predicted]:.6f})."
            ]
        }

    def is_trained(self) -> bool:
        return self._trained


# ---------------------------------------------------------------------------
# k-Nearest Neighbors Classifier
# ---------------------------------------------------------------------------

class KNNClassifier:
    """
    A deterministic k-Nearest Neighbors Classifier.

    Classifies a query vector by finding the k nearest training samples
    using Euclidean distance and returning the majority class label.
    No training phase required — classification is performed at query time.

    Parameters:
        k (int): Number of nearest neighbors to consider.
    """

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self._X_train: list = []
        self._y_train: list = []
        self._feature_names: list = []
        self._trained: bool = False
        self._class_labels: list = []

    def train(self, X: list, y: list, feature_names: Optional[list] = None) -> dict:
        """
        Store the training data for use during classification.

        Parameters:
            X: List of feature vectors.
            y: List of class labels.

        Returns:
            A training report dict.
        """
        if not X or not y:
            raise ValueError("Training data must not be empty.")

        self._X_train = X
        self._y_train = y
        self._feature_names = feature_names or [f"feature_{i}" for i in range(len(X[0]))]
        self._class_labels = sorted(set(y))
        self._trained = True

        return {
            "status": "trained",
            "algorithm": f"k-Nearest Neighbors (k={self.k}, Euclidean Distance)",
            "samples": len(X),
            "features": len(X[0]),
            "classes": self._class_labels,
            "k": self.k,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }

    def predict(self, x: list) -> dict:
        """
        Predict the class label for a single feature vector.

        Returns a dict with the predicted label, k nearest neighbors, and proof.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        # Compute distances to all training samples
        distances = []
        for i, xi in enumerate(self._X_train):
            dist = _euclidean_distance(x, xi)
            distances.append((dist, self._y_train[i], i))

        # Sort by distance (ascending) and select k nearest
        distances.sort(key=lambda t: t[0])
        k_nearest = distances[:self.k]

        # Majority vote
        votes = Counter(label for _, label, _ in k_nearest)
        predicted = votes.most_common(1)[0][0]

        neighbors_info = [
            {
                "index": idx,
                "distance": round(dist, 6),
                "label": label
            }
            for dist, label, idx in k_nearest
        ]

        return {
            "prediction": predicted,
            "algorithm": f"k-Nearest Neighbors (k={self.k})",
            "k_nearest_neighbors": neighbors_info,
            "vote_counts": dict(votes),
            "trace": [
                f"Computed Euclidean distance to {len(self._X_train)} training samples.",
                f"Selected {self.k} nearest neighbors.",
                f"Vote counts: {dict(votes)}.",
                f"Predicted class by majority vote: '{predicted}'."
            ]
        }

    def is_trained(self) -> bool:
        return self._trained


# ---------------------------------------------------------------------------
# ML Engine — Unified Interface
# ---------------------------------------------------------------------------

class MLEngine:
    """
    The Model XP Machine Learning Engine.

    Provides a unified interface for training and querying deterministic
    machine learning classifiers. All models are interpretable and produce
    a full decision trace with every prediction.

    Supported algorithms:
        - "decision_tree": Decision Tree (Information Gain)
        - "naive_bayes":   Naive Bayes (Gaussian)
        - "knn":           k-Nearest Neighbors (Euclidean Distance)
    """

    SUPPORTED_ALGORITHMS = ["decision_tree", "naive_bayes", "knn"]

    def __init__(self):
        self._models: dict = {}
        self._training_log: list = []

    def train(
        self,
        model_id: str,
        algorithm: str,
        X: list,
        y: list,
        feature_names: Optional[list] = None,
        **kwargs
    ) -> dict:
        """
        Train a named model using the specified algorithm.

        Parameters:
            model_id:      A unique identifier for this model.
            algorithm:     One of 'decision_tree', 'naive_bayes', 'knn'.
            X:             Feature matrix (list of lists of numbers).
            y:             Label vector (list of strings or integers).
            feature_names: Optional list of feature names.
            **kwargs:      Algorithm-specific parameters (e.g., max_depth, k).

        Returns:
            A training report dict.
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: '{algorithm}'. "
                f"Supported: {self.SUPPORTED_ALGORITHMS}."
            )

        if algorithm == "decision_tree":
            model = DecisionTreeClassifier(
                max_depth=kwargs.get("max_depth", 10),
                min_samples_split=kwargs.get("min_samples_split", 2)
            )
        elif algorithm == "naive_bayes":
            model = NaiveBayesClassifier()
        elif algorithm == "knn":
            model = KNNClassifier(k=kwargs.get("k", 3))

        report = model.train(X, y, feature_names=feature_names)
        report["model_id"] = model_id
        self._models[model_id] = model
        self._training_log.append(report)
        return report

    def predict(self, model_id: str, x: list) -> dict:
        """
        Predict the class label for a feature vector using a trained model.

        Parameters:
            model_id: The identifier of the trained model to use.
            x:        A single feature vector (list of numbers).

        Returns:
            A prediction result dict with label, algorithm, and trace.
        """
        if model_id not in self._models:
            raise KeyError(
                f"Model '{model_id}' not found. "
                f"Available models: {list(self._models.keys())}."
            )
        model = self._models[model_id]
        result = model.predict(x)
        result["model_id"] = model_id
        result["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
        return result

    def list_models(self) -> list:
        """Return a list of trained model IDs."""
        return list(self._models.keys())

    def save_model(self, model_id: str, path: str) -> None:
        """
        Serialize a trained k-NN or Naive Bayes model to a JSON file.
        Decision trees are serialized via their training log.
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found.")
        model = self._models[model_id]
        data = {"model_id": model_id}

        if isinstance(model, KNNClassifier):
            data["algorithm"] = "knn"
            data["k"] = model.k
            data["X_train"] = model._X_train
            data["y_train"] = model._y_train
            data["feature_names"] = model._feature_names
        elif isinstance(model, NaiveBayesClassifier):
            data["algorithm"] = "naive_bayes"
            data["class_priors"] = model._class_priors
            data["class_stats"] = model._class_stats
            data["class_labels"] = model._class_labels
            data["feature_names"] = model._feature_names
        else:
            data["algorithm"] = "decision_tree"
            data["note"] = "Decision tree serialization requires re-training from stored data."

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_training_log(self) -> list:
        """Return the full training log for all models."""
        return list(self._training_log)
