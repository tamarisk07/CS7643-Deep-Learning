import unittest
import numpy as np
from modules import Linear
from .utils import *

class TestLinear(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def _affine_forward(self, x, w, b):
        layer = Linear(w.shape[0], w.shape[1])
        layer.weight = w
        layer.bias = b
        return layer.forward(x)

    def _affine_backward(self, x, w, b, dout):
        layer = Linear(w.shape[0], w.shape[1])
        layer.weight = w
        layer.bias = b
        tmp = layer.forward(x)
        layer.backward(dout)
        return layer.dx, layer.dw, layer.db

    def test_forward(self):
        num_inputs = 2
        input_shape = (4, 5, 6)
        output_dim = 3

        input_size = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)

        x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
        b = np.linspace(-0.3, 0.1, num=output_dim)

        out = self._affine_forward(x, w, b)
        correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                                [3.25553199, 3.5141327, 3.77273342]])
        self.assertAlmostEquals(rel_error(out, correct_out), 0, places=8)

    def test_backward(self):
        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        dx_num = eval_numerical_gradient_array(lambda x: self._affine_forward(x, w, b), x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: self._affine_forward(x, w, b), w, dout)
        db_num = eval_numerical_gradient_array(lambda b: self._affine_forward(x, w, b), b, dout)

        dx, dw, db = self._affine_backward(x, w, b, dout)

        self.assertAlmostEquals(rel_error(dx, dx_num), 0, places=8)
        self.assertAlmostEquals(rel_error(dw, dw_num), 0, places=8)
        self.assertAlmostEquals(rel_error(db, db_num), 0, places=8)

