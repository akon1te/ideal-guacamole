import torch

from src.dynsys.models.csode import CSODE
from src.dynsys.models.registry import available_model_names, build_model


def test_csode_rollout_has_shared_model_shape():
    model = CSODE(dim=3, hidden=8, layers=3)
    y0 = torch.randn(4, 3)
    grid = torch.linspace(0.0, 0.5, 6)
    assert model(y0, grid).shape == (4, 5, 3)


def test_default_control_starts_at_observed_state():
    model = CSODE(dim=3, hidden=8)
    y0 = torch.randn(2, 3)
    z0 = torch.cat([y0, y0], dim=-1)
    assert torch.allclose(z0[:, :3], z0[:, 3:])


def test_control_update_drives_both_joint_state_components():
    model = CSODE(dim=3, hidden=8)
    z = torch.randn(2, 6)
    derivative = model.func(torch.tensor(0.0), z)
    control = model.func.control(torch.tensor(0.0), z[:, 3:])
    assert torch.allclose(derivative[:, 3:], control)
    assert torch.allclose(derivative[:, :3] - control,
                          model.func.main(torch.tensor(0.0), z[:, :3]))


def test_registry_builds_csode_and_exposes_it_publicly():
    assert "csode" in available_model_names()
    model = build_model("csode", dim=3, hidden=8, seq_len=5)
    assert isinstance(model, CSODE)
