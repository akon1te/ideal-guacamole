import numpy as np

from src.dynsys.data.dataset import normalise_train_validation


def test_validation_uses_training_statistics_only():
    train = np.array([[0.0, 10.0], [2.0, 14.0]], dtype=np.float32)
    validation = np.array([[100.0, 200.0]], dtype=np.float32)

    train_norm, validation_norm, mean, std = normalise_train_validation(train, validation)

    assert np.allclose(mean, [1.0, 12.0])
    assert np.allclose(std, [1.0, 2.0])
    assert np.allclose(train_norm.mean(axis=0), [0.0, 0.0])
    assert np.allclose(validation_norm, [[99.0, 94.0]])
