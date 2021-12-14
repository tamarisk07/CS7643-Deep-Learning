import unittest
import numpy as np
from models.softmax_regression import SoftmaxRegression
from models.two_layer_nn import TwoLayerNet


class TestNetwork(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        self.test_batch = np.load('tests/softmax_grad_check/test_batch.npy')
        self.test_label = np.load('tests/softmax_grad_check/test_label.npy')

    def test_one_layer_softmax_relu(self):
        model = SoftmaxRegression()
        expected_loss = 2.3029
        expected_grad = np.load('tests/softmax_grad_check/softmax_relu_grad.npy')
        loss, _ = model.forward(self.test_batch, self.test_label, mode='train')
        w_grad = model.gradients['W1']
        self.assertAlmostEqual(expected_loss, loss, places=5)
        diff = np.sum(np.abs(expected_grad - w_grad))
        self.assertAlmostEqual(diff, 0)

    def test_two_layer_net(self):
        model = TwoLayerNet(hidden_size=128)
        expected_loss = 2.30285
        w1_grad_expected = np.load('tests/twolayer_grad_check/w1.npy')
        b1_grad_expected = np.load('tests/twolayer_grad_check/b1.npy')
        w2_grad_expected = np.load('tests/twolayer_grad_check/w2.npy')
        b2_grad_expected = np.load('tests/twolayer_grad_check/b2.npy')

        loss, _ = model.forward(self.test_batch, self.test_label, mode='train')

        self.assertAlmostEqual(expected_loss, loss, places=5)


        self.assertAlmostEqual(np.sum(np.abs(w1_grad_expected - model.gradients['W1'])), 0)
        self.assertAlmostEqual(np.sum(np.abs(b1_grad_expected - model.gradients['b1'])), 0)
        self.assertAlmostEqual(np.sum(np.abs(w2_grad_expected - model.gradients['W2'])), 0)
        self.assertAlmostEqual(np.sum(np.abs(b2_grad_expected - model.gradients['b2'])), 0)