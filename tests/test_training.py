import unittest

import numpy as np

from models.softmax_regression import SoftmaxRegression
from models.two_layer_nn import TwoLayerNet

from optimizer.sgd import SGD
from utils import load_mnist_trainval, load_mnist_test, generate_batched_data
from utils import train, evaluate


class TestTraining(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def test_regularization(self):
        optimizer = SGD(learning_rate=1e-4, reg=100)
        model = SoftmaxRegression()

        optimizer.apply_regularization(model)
        w_grad_reg = model.gradients['W1']
        reg_diff = w_grad_reg
        expected_diff = model.weights['W1'] * optimizer.reg

        diff = np.mean(np.abs(reg_diff - expected_diff))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_sgd(self):
        optimizer = SGD(learning_rate=1e-4, reg=1e-3)
        model = SoftmaxRegression()
        np.random.seed(467)
        fake_gradients = np.random.randn(784, 10)
        model.gradients['W1'] = fake_gradients
        optimizer.update(model)
        expected_weights = np.load('tests/sgd/sgd_updated_weights.npy')
        diff = np.abs(expected_weights - model.weights['W1'])
        diff = np.sum(diff)

        self.assertAlmostEqual(diff, 0)

    def test_one_layer_train(self):
        model = SoftmaxRegression()
        optimizer = SGD(learning_rate=0.1, reg=1e-3)
        train_data, train_label, _, _ = load_mnist_trainval()
        test_data, test_label = load_mnist_test()

        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=128, shuffle=True, seed=1024)
        _, train_acc = train(1, batched_train_data, batched_train_label, model, optimizer, debug=False)

        batched_test_data, batched_test_label = generate_batched_data(test_data, test_label, batch_size=128)
        _, test_acc = evaluate(batched_test_data, batched_test_label, model, debug=False)
        self.assertGreater(train_acc, 0.3)
        self.assertGreater(test_acc, 0.3)

    def test_two_layer_train(self):
        model = TwoLayerNet(hidden_size=128)
        optimizer = SGD(learning_rate=0.1, reg=1e-3)
        train_data, train_label, _, _ = load_mnist_trainval()
        test_data, test_label = load_mnist_test()

        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=32, shuffle=True, seed=1024)
        _, train_acc = train(1, batched_train_data, batched_train_label, model, optimizer, debug=False)

        batched_test_data, batched_test_label = generate_batched_data(test_data, test_label, batch_size=32)
        _, test_acc = evaluate(batched_test_data, batched_test_label, model, debug=False)
        self.assertGreater(train_acc, 0.3)
        self.assertGreater(test_acc, 0.3)