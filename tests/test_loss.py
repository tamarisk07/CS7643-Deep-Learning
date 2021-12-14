import unittest
import numpy as np
from models.softmax_regression import SoftmaxRegression


class TestActivation(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        self.model = SoftmaxRegression()

    def test_ce_loss(self):
        x = np.array([[0.2, 0.5, 0.3], [0.5, 0.1, 0.4], [0.3, 0.3, 0.4]])
        y = np.array([1, 2, 0])
        expected_loss = 0.937803
        loss = self.model.cross_entropy_loss(x, y)
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_accuracy(self):
        x = np.array([[0.2, 0.5, 0.3], [0.5, 0.1, 0.4], [0.3, 0.3, 0.4]])
        y = np.array([1, 2, 0])
        expected_acc = 0.3333
        acc = self.model.compute_accuracy(x, y)
        self.assertAlmostEqual(acc, expected_acc, places=4)
