# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp

from warp.tests.test_base import *

import unittest

wp.init()

@wp.kernel
def kernel_svd(
        A: wp.array(dtype=wp.mat33),
        U: wp.array(dtype=wp.mat33),
        s: wp.array(dtype=wp.vec3),
        V: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    Ui, si, Vi = wp.svd3(A[i])
    U[i] = Ui
    s[i] = si
    V[i] = Vi

def test_svd(test, device):
    dim = 10

    np.random.seed(0)
    A_ = np.random.normal(size=(dim, 3, 3))

    A = wp.array(A_, dtype=wp.mat33, device=device)
    U = wp.empty_like(A)
    s = wp.empty(dim, dtype=wp.vec3, device=device)
    V = wp.empty_like(A)

    with CheckOutput(test):
        wp.launch(kernel_svd, dim=dim, inputs=[A, U, s, V], device=device)

    U_ = U.numpy()
    s_ = s.numpy()
    V_ = V.numpy()
    for Ai, Ui, si, Vi in zip(A_, U_, s_, V_):
        assert_np_equal(Ui@np.diag(si)@Vi.T, Ai, 0.1)
        assert_np_equal(Ui@Ui.T, np.eye(3), 1e-4)
        assert_np_equal(Vi@Vi.T, np.eye(3), 1e-4)


def register(parent):

    devices = wp.get_devices()

    class TestSVD(parent):
        pass

    add_function_test(TestSVD, "test_svd", test_svd, devices=devices)

    return TestSVD

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
