# Pluggable branch predictor implementations
from branch_prediction_sim.predictors.base import BasePredictor
from branch_prediction_sim.predictors.bht import BHTPredictor
from branch_prediction_sim.predictors.gas import GAsPredictor
from branch_prediction_sim.predictors.btb import BTBPredictor
from branch_prediction_sim.predictors.perceptron import PerceptronPredictor

# Static predictor (Lx default)
class StaticPredictor(BasePredictor):
    """Always predicts not-taken. This is the Lx reference configuration."""
    def predict(self, pc): return 0
    def update(self, pc, actual): pass
    def read_energy(self): return 0.0
    def write_energy(self): return 0.0

__all__ = [
    'BasePredictor', 'StaticPredictor',
    'BHTPredictor', 'GAsPredictor', 'BTBPredictor', 'PerceptronPredictor',
]
