import numpy as np

from mani_skill.utils.geometry.geometry import angle_between_vec, rotation_between_vec


def test_parallel_vectors_have_zero_angle():
    vector = np.array([1.0, 1.0, 1.0])
    assert angle_between_vec(vector, vector) == 0.0


def test_opposite_vectors_have_a_half_turn():
    vector = np.array([1.0, 1.0, 1.0])
    assert np.isclose(angle_between_vec(vector, -vector), np.pi)


def test_right_angle_is_unchanged():
    assert np.isclose(
        angle_between_vec(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        np.pi / 2,
    )


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def test_rotation_of_a_vector_onto_itself():
    vector = _unit([1.0, 1.0, 1.0])
    got = rotation_between_vec(vector, vector).apply(vector)
    assert np.allclose(got, vector)


def test_rotation_onto_the_opposite_vector():
    vector = _unit([1.0, 1.0, 1.0])
    got = rotation_between_vec(vector, -vector).apply(vector)
    assert np.allclose(got, -vector)


def test_quarter_turn_rotation_is_unchanged():
    source = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])
    got = rotation_between_vec(source, target).apply(source)
    assert np.allclose(got, target)
