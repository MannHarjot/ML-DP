from dataclasses import dataclass

import numpy as np


@dataclass
class DPLogisticRegression:
    epsilon: float = 2.0
    delta: float = 1e-5
    learning_rate: float = 0.05
    epochs: int = 30
    batch_size: int = 128
    l2_reg: float = 1e-4
    max_grad_norm: float = 1.0
    random_state: int = 42

    def _sigma(self) -> float:
        # Practical approximation for Gaussian mechanism noise scale.
        return np.sqrt(2 * np.log(1.25 / self.delta)) / max(self.epsilon, 1e-9)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DPLogisticRegression":
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("DPLogisticRegression currently supports binary labels {0, 1} only")

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        self.w_ = np.zeros(n_features, dtype=float)
        self.b_ = 0.0

        sigma = self._sigma()

        for _ in range(self.epochs):
            idx = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                xb = X[batch_idx]
                yb = y[batch_idx]

                logits = xb @ self.w_ + self.b_
                probs = self._sigmoid(logits)
                errors = probs - yb

                grads_w = errors[:, None] * xb + self.l2_reg * self.w_
                grads_b = errors

                norms = np.sqrt((grads_w**2).sum(axis=1) + grads_b**2)
                clip_factors = np.minimum(1.0, self.max_grad_norm / (norms + 1e-12))

                grads_w = grads_w * clip_factors[:, None]
                grads_b = grads_b * clip_factors

                grad_w = grads_w.mean(axis=0)
                grad_b = grads_b.mean()

                noise_std = sigma * self.max_grad_norm / max(len(batch_idx), 1)
                grad_w += rng.normal(0.0, noise_std, size=grad_w.shape)
                grad_b += float(rng.normal(0.0, noise_std))

                self.w_ -= self.learning_rate * grad_w
                self.b_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs_1 = self._sigmoid(X @ self.w_ + self.b_)
        probs_0 = 1.0 - probs_1
        return np.column_stack([probs_0, probs_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
