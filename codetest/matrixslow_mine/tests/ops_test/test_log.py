import sys
from unicodedata import decimal
sys.path.append("../..")
from core.node import Tensor
import numpy as np
import math
decimal=5
def test_simple_log():
    x = Tensor(10, requires_grad=True)
    z = x.log()

    np.testing.assert_array_almost_equal(z.data, math.log(10),decimal=decimal)

    z.backward()

    np.testing.assert_array_almost_equal(x.grad.data.tolist(), 0.1,decimal=decimal)


def test_array_log():
    x = Tensor([1, 2, 3], requires_grad=True)
    z = x.log()

    np.testing.assert_array_almost_equal(z.data, np.log([1, 2, 3]),decimal=decimal)

    z.backward([1, 1, 1])

    np.testing.assert_array_almost_equal(x.grad.data.tolist(), [1, 0.5, 1 / 3],decimal=decimal)

test_simple_log()
test_array_log()