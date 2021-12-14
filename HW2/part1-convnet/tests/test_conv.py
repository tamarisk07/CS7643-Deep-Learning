import unittest
import numpy as np
from modules import Conv2D
from .utils import *

class TestConv(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def _conv_forward(self, x, w, b, in_channels, out_channels, kernel_size, stride, padding):
        conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        conv.weight = w
        conv.bias = b
        return conv.forward(x)

    def _conv_backward(self, x, w, b, dout, in_channels, out_channels, kernel_size, stride, padding):
        conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        conv.weight = w
        conv.bias = b
        tmp = conv.forward(x)
        conv.backward(dout)
        return conv.dx, conv.dw, conv.db

    def test_forward(self):
        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)
        out = self._conv_forward(x, w, b, 3, 3, 4, 2, 1)
        correct_out = np.array([[[[[-0.08759809, -0.10987781],
                                   [-0.18387192, -0.2109216]],
                                  [[0.21027089, 0.21661097],
                                   [0.22847626, 0.23004637]],
                                  [[0.50813986, 0.54309974],
                                   [0.64082444, 0.67101435]]],
                                 [[[-0.98053589, -1.03143541],
                                   [-1.19128892, -1.24695841]],
                                  [[0.69108355, 0.66880383],
                                   [0.59480972, 0.56776003]],
                                  [[2.36270298, 2.36904306],
                                   [2.38090835, 2.38247847]]]]])
        diff = rel_error(out, correct_out)
        self.assertAlmostEquals(diff, 0, places=7)

    def test_backward(self):
        x = np.random.randn(4, 3, 5, 5)
        w = np.random.randn(2, 3, 3, 3)
        b = np.random.randn(2, )
        dout = np.random.randn(4, 2, 5, 5)

        dx_num = eval_numerical_gradient_array(lambda x: self._conv_forward(x, w, b, 3, 2, 3, 1, 1), x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: self._conv_forward(x, w, b, 3, 2, 3, 1, 1), w, dout)
        db_num = eval_numerical_gradient_array(lambda b: self._conv_forward(x, w, b, 3, 2, 3, 1, 1), b, dout)

        dx, dw, db = self._conv_backward(x, w, b, dout, 3, 2, 3, 1, 1)

        self.assertAlmostEquals(rel_error(dx, dx_num), 0, places=6)
        self.assertAlmostEquals(rel_error(dw, dw_num), 0, places=6)
        self.assertAlmostEquals(rel_error(db, db_num), 0, places=6)




