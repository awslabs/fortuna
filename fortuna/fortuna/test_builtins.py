import unittest

import jax.numpy as jnp
import numpy as np
from fortuna.utils.builtins import HashableMixin


class ChildA(HashableMixin):
    def __init__(self, a, b):
        self.a = a
        self.b = b


class ChildB(HashableMixin):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self._prvt = None


class ChildC(HashableMixin):
    def __init__(self, a, b):
        self.child = ChildA(a, b)


class TestHashableMixin(unittest.TestCase):
    def test_same_hash_and_equal(self):
        c1 = ChildA(1, 2)
        c2 = ChildA(1, 2)
        self.assertEqual(c1, c2)
        self.assertEqual(c2, c1)
        self.assertEqual(hash(c1), hash(c2))

        c1 = ChildC(1, 2)
        c2 = ChildC(1, 2)
        self.assertEqual(c1, c2)
        self.assertEqual(c2, c1)
        self.assertEqual(hash(c1), hash(c2))

    def test_same_hash_and_equal_with_private_attributes(self):
        c1 = ChildA(1, 2)
        c2 = ChildA(1, 2)
        c2._private_attr = 100
        self.assertEqual(c1, c2)
        self.assertEqual(c2, c1)
        self.assertEqual(hash(c1), hash(c2))

        c1 = ChildC(1, 2)
        c2 = ChildC(1, 2)
        c2.child._prvt = None
        self.assertEqual(c1, c2)
        self.assertEqual(c2, c1)
        self.assertEqual(hash(c1), hash(c2))

        c2 = ChildC(1, 2)
        c2._prvt = None
        self.assertEqual(c1, c2)
        self.assertEqual(c2, c1)
        self.assertEqual(hash(c1), hash(c2))

    def test_same_hash_but_not_equal(self):
        c1 = ChildA(1, 2)
        c2 = ChildB(1, 2)
        self.assertNotEqual(c1, c2)
        self.assertNotEqual(c2, c1)
        self.assertEqual(hash(c1), hash(c2))

    def test_different_hash_and_not_equal(self):
        c1 = ChildA(1, 2)
        c2 = ChildA(10, 20)
        self.assertNotEqual(c1, c2)
        self.assertNotEqual(c2, c1)
        self.assertNotEqual(hash(c1), hash(c2))

        c2 = ChildA(1, 2)
        c2.d = 100
        self.assertNotEqual(hash(c1), hash(c2))
        self.assertNotEqual(c1, c2)
        self.assertNotEqual(c2, c1)

        c2 = ChildC(1, 2)
        self.assertNotEqual(c1, c2)
        self.assertNotEqual(c2, c1)
        self.assertNotEqual(hash(c1), hash(c2))

        c1 = ChildC(1, 1)
        self.assertNotEqual(c1, c2)
        self.assertNotEqual(c2, c1)
        self.assertNotEqual(hash(c1), hash(c2))

    def test_assert_arrays_are_not_supported(self):
        c1 = ChildA(1, jnp.arange(3))
        with self.assertRaises(TypeError):
            hash(c1)

        c1 = ChildA(1, np.arange(3))
        with self.assertRaises(TypeError):
            hash(c1)
