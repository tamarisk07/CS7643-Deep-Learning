import unittest
import numpy as np
from modules import MaxPooling
from .utils import *

class TestConv(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def _pool_forward(self, x):
        pool = MaxPooling(kernel_size=2, stride=2)
        return pool.forward(x)

    def test_forward(self):
        x_shape = (2, 3, 4, 4)
        x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)


        out = self._pool_forward(x)

        correct_out = np.array([[[[-0.26315789, -0.24842105],
                                  [-0.20421053, -0.18947368]],
                                 [[-0.14526316, -0.13052632],
                                  [-0.08631579, -0.07157895]],
                                 [[-0.02736842, -0.01263158],
                                  [0.03157895, 0.04631579]]],
                                [[[0.09052632, 0.10526316],
                                  [0.14947368, 0.16421053]],
                                 [[0.20842105, 0.22315789],
                                  [0.26736842, 0.28210526]],
                                 [[0.32631579, 0.34105263],
                                  [0.38526316, 0.4]]]])

        diff = rel_error(out, correct_out)
        self.assertAlmostEquals(diff, 0, places=7)

    def test_backward(self):
        x = np.random.randn(3, 2, 8, 8)
        dout = np.random.randn(3, 2, 4, 4)



        dx_num = eval_numerical_gradient_array(lambda x: self._pool_forward(x), x, dout)
        pool = MaxPooling(kernel_size=2, stride=2)
        out = pool.forward(x)
        pool.backward(dout)
        dx = pool.dx

        self.assertAlmostEquals(rel_error(dx, dx_num), 0, places=8)
