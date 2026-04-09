"""
Perceptron Branch Predictor — ML Extension (beyond paper).

A genuine machine-learning-based branch predictor using a table of
perceptron weight vectors. Each perceptron computes a weighted sum
of the global branch history to produce a prediction.

  y = w_0 + sum(w_i * h_i)   where h_i in {-1, +1}
  prediction = Taken if y >= 0, else Not-Taken

Training rule (online learning):
  If misprediction OR |y| <= theta:
    w_0 += t
    w_i += t * h_i           where t = +1 (taken) or -1 (not-taken)

The threshold theta = floor(1.93 * history_length + 14) determines
when to train even on correct predictions (confidence-based training).

This predictor captures complex, long-range correlations that 2-bit
counters cannot represent — making it a meaningful ML contribution
beyond the paper's BHT/GAs methods.

References:
  - Jimenez & Lin, "Dynamic Branch Prediction with Perceptrons,"
    HPCA, 2001.
"""

from predictors.base import BasePredictor
from core.energy import ENERGY


class PerceptronPredictor(BasePredictor):
    """Perceptron-based branch predictor with online learning."""

    def __init__(self, table_size=64, history_length=16):
        """
        Args:
            table_size:     number of perceptron weight vectors
            history_length: number of past branch outcomes used as features
        """
        self.table_size = table_size
        self.history_length = history_length

        # Weight table: each entry has (history_length + 1) weights
        # w[0] = bias, w[1..N] = correlation weights
        self.weights = [[0] * (history_length + 1) for _ in range(table_size)]

        # Global history as {-1, +1} values
        self.history = [-1] * history_length

        # Training threshold (from Jimenez & Lin)
        self.theta = int(1.93 * history_length + 14)

    def _index(self, pc):
        return pc % self.table_size

    def _output(self, pc):
        """Compute perceptron output y."""
        w = self.weights[self._index(pc)]
        y = w[0]
        for i in range(self.history_length):
            y += w[i + 1] * self.history[i]
        return y

    def predict(self, pc):
        return 1 if self._output(pc) >= 0 else 0

    def update(self, pc, actual):
        y = self._output(pc)
        pred = 1 if y >= 0 else 0
        t = 1 if actual == 1 else -1

        # Train on misprediction or low-confidence correct prediction
        if pred != actual or abs(y) <= self.theta:
            idx = self._index(pc)
            w = self.weights[idx]
            w[0] += t  # bias
            for i in range(self.history_length):
                w[i + 1] += t * self.history[i]
                w[i + 1] = max(-128, min(127, w[i + 1]))  # clamp

        # Update global history
        self.history = [t] + self.history[:-1]

    def read_energy(self):
        return ENERGY['perceptron_inference']

    def write_energy(self):
        return ENERGY['perceptron_update']
