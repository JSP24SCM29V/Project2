import numpy as np
from scipy import sparse

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None, None

        best_loss = float('inf')
        best_feature = None
        best_threshold = None
        current_loss = self._mse(y)

        for feature in range(n):
            values = X[:, feature]
            order = np.argsort(values)
            sorted_values = values[order]
            sorted_y = y[order]
            
            for i in range(1, m):
                if sorted_values[i] == sorted_values[i-1]:
                    continue
                threshold = (sorted_values[i] + sorted_values[i-1]) / 2
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx

                if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                    continue

                left_loss = self._mse(y[left_idx])
                right_loss = self._mse(y[right_idx])
                weighted_loss = (np.sum(left_idx) * left_loss + np.sum(right_idx) * right_loss) / m

                if weighted_loss < best_loss:
                    best_loss = weighted_loss
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, current_loss - best_loss if best_feature is not None else 0

    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'leaf': True, 'value': np.mean(y)}

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain == 0:
            return {'leaf': True, 'value': np.mean(y)}

        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        left_subtree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])

class GradientBoostingClassifier:
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=5, min_samples_split=2, subsample=0.8, max_features=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.max_features = max_features
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)  # Convert to {-1, 1}
        pos_mean = np.mean(y == 1)
        neg_mean = np.mean(y == -1)
        pos_mean = np.clip(pos_mean, 1e-10, 1 - 1e-10)
        neg_mean = np.clip(neg_mean, 1e-10, 1 - 1e-10)
        self.initial_pred = np.log(pos_mean / neg_mean)
        predictions = np.full(len(y), self.initial_pred)
        n_samples, n_features = X.shape

        # Validation set for early stopping
        val_idx = np.random.choice(n_samples, int(0.2 * n_samples), replace=False)
        train_idx = np.setdiff1d(np.arange(n_samples), val_idx)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for _ in range(self.n_estimators):
            # Subsample data
            sample_idx = np.random.choice(len(X_train), int(self.subsample * len(X_train)), replace=False)
            X_sample, y_sample = X_train[sample_idx], y_train[sample_idx]

            # Subsample features
            if self.max_features is not None:
                n_features_sample = int(self.max_features * n_features) if self.max_features <= 1.0 else self.max_features
                feature_idx = np.random.choice(n_features, n_features_sample, replace=False)
                X_sample = X_sample[:, feature_idx]
            else:
                feature_idx = np.arange(n_features)

            probabilities = 1 / (1 + np.exp(-predictions[train_idx][sample_idx]))
            gradients = y_sample * (1 - probabilities)
            gradients = np.clip(gradients, -1, 1)  # Gradient clipping

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, gradients)

            # Update predictions with feature subset
            tree_pred = tree.predict(X[:, feature_idx])
            predictions += self.learning_rate * tree_pred

            # Validation loss for early stopping
            val_proba = 1 / (1 + np.exp(-predictions[val_idx]))
            val_loss = -np.mean(y_val * np.log(np.clip(val_proba, 1e-10, 1 - 1e-10)) + (1 - y_val) * np.log(np.clip(1 - val_proba, 1e-10, 1 - 1e-10)))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            self.trees.append((tree, feature_idx))

        return GradientBoostingResults(self)

class GradientBoostingResults:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        predictions = np.full(len(X), self.model.initial_pred)
        for tree, feature_idx in self.model.trees:
            predictions += self.model.learning_rate * tree.predict(X[:, feature_idx])
        probabilities = 1 / (1 + np.exp(-predictions))
        
        # Simple Platt scaling for calibration
        from sklearn.linear_model import LogisticRegression
        proba = probabilities.reshape(-1, 1)
        if hasattr(self.model, 'y_train_'):
            lr = LogisticRegression().fit(proba, self.model.y_train_)
            probabilities = lr.predict_proba(proba)[:, 1]
        
        return np.vstack((1 - probabilities, probabilities)).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        accuracy = np.mean(y_pred == y)
        true_positives = np.sum((y_pred == 1) & (y == 1))
        predicted_positives = np.sum(y_pred == 1)
        actual_positives = np.sum(y == 1)

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        def roc_auc_score(y_true, y_score):
            order = np.argsort(y_score)[::-1]
            y_true = y_true[order]
            y_score = y_score[order]
            tpr = [0]
            fpr = [0]
            tp = 0
            fp = 0
            pos = np.sum(y_true == 1)
            neg = np.sum(y_true == 0)
            for i in range(len(y_true)):
                if y_true[i] == 1:
                    tp += 1
                else:
                    fp += 1
                tpr.append(tp / pos if pos > 0 else 0)
                fpr.append(fp / neg if neg > 0 else 0)
            # return np.trapezoid(tpr, fpr)
            return np.trapz(tpr, fpr)

        roc_auc = roc_auc_score(y, y_proba)

        def log_loss(y_true, y_pred_proba):
            y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
            return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))

        logloss = log_loss(y, y_proba)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'log_loss': logloss
        }

