import unittest
import numpy as np
from models.softmax_regression import SoftmaxRegression


class TestActivation(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        self.model = SoftmaxRegression()

    def test_sigmoid(self):
        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.1841628, 0.4218198],
                      [0.42978908, 0.26740977],
                      [0.66023782, 0.77794766],
                      [0.16133995, 0.71140804]])
        outs = self.model.sigmoid(x)
        diff = np.sum(np.abs((outs - y)))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_sigmoid_dev(self):
        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.15024686, 0.24388786],
                      [0.24507043, 0.19590178],
                      [0.22432384, 0.1727451 ],
                      [0.13530937, 0.20530664]])

        outs = self.model.sigmoid_dev(x)
        diff = np.sum(np.abs((outs - y)))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_relu(self):
        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.66435418, 1.2537461],
                      [0.0, 0.90223236]])
        out = self.model.ReLU(x)
        diff = np.sum(np.abs((y - out)))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_relu_dev(self):
        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [1., 1.],
                      [0.0, 1.]])
        out = self.model.ReLU_dev(x)
        diff = np.sum(np.abs((y - out)))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_softmax(self):
        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.23629739, 0.76370261],
                      [0.67372745, 0.32627255],
                      [0.35677439, 0.64322561],
                      [0.07239128, 0.92760872]])

        out = self.model.softmax(x)

        diff = np.sum(np.abs((y - out)))
        self.assertAlmostEqual(diff, 0, places=7)
