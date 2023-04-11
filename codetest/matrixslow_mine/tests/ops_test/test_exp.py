import sys
sys.path.append("../..")
from core.node import Tensor
import numpy as np

def test_simple_exp():
    x = Tensor(2, requires_grad=True)
    z = x.exp()  # e^2

    np.testing.assert_array_almost_equal(z.data, np.exp(2))

    z.backward()

    np.testing.assert_array_almost_equal(x.grad.data, np.exp(2))


def test_array_exp():
    x = Tensor([1, 2, 3], requires_grad=True)
    z = x.exp()

    np.testing.assert_array_almost_equal(z.data, np.exp([1, 2, 3]))

    z.backward([1, 1, 1])

    np.testing.assert_array_almost_equal(x.grad.data, np.exp([1, 2, 3]))


def test_simple_pow():
    x = Tensor(2, requires_grad=True)
    y = 2
    z = x ** y

    assert z.data == 4

    z.backward()

    assert x.grad.data == 4


def test_array_pow():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = 3
    z = x ** y

    assert z.data.tolist() == [1, 8, 27]

    z.backward([1, 1, 1])

    assert x.grad.data.tolist() == [3, 12, 27]


def test_simple_neg():
    x = Tensor(2, requires_grad=True)
    z = -x  # -2

    assert z.data == -2

    z.backward()

    assert x.grad.data == -1

def test_array_neg():
    x = Tensor([1, 2, 3], requires_grad=True)

    z = -x

    np.testing.assert_array_equal(z.data, [-1, -2, -3])

    z.backward([1, 1, 1])

    np.testing.assert_array_equal(x.grad.data, [-1, -1, -1])

    

test_simple_exp()
test_array_exp()

test_simple_pow()
test_array_pow()

test_simple_neg()
test_array_neg()