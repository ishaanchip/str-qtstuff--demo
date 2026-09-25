"""Logistic regression classifier trained on randomly generated sample data.

Run: python3 model.py
"""
import numpy as np

SEED = 42
N_SAMPLES = 1000
N_FEATURES = 5


def make_data(n_samples, n_features, rng):
    """Generate a binary classification dataset from a hidden linear rule plus noise."""
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=n_features)
    true_b = 0.5
    logits = X @ true_w + true_b + rng.normal(scale=0.5, size=n_samples)
    y = (logits > 0).astype(float)
    return X, y, true_w, true_b


def train_test_split(X, y, test_frac, rng):
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]

#mod

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for epoch in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            self.w -= self.lr * (X.T @ (p - y)) / n
            self.b -= self.lr * np.mean(p - y)
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
                print(f"epoch {epoch:4d}  loss {loss:.4f}")
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(float)


def main():
    rng = np.random.default_rng(SEED)
    X, y, true_w, true_b = make_data(N_SAMPLES, N_FEATURES, rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2, rng)

    model = LogisticRegression().fit(X_train, y_train)

    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)
    print(f"\ntrain accuracy: {train_acc:.3f}")
    print(f"test accuracy:  {test_acc:.3f}")
    print(f"\ntrue weights:    {np.round(true_w, 2)}  bias {true_b:.2f}")
    print(f"learned weights: {np.round(model.w, 2)}  bias {model.b:.2f}")


if __name__ == "__main__":
    main()
