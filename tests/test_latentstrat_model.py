import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


def test_model_shapes_attention_and_active_l2():
    import torch

    from latentstrat.config import default_options
    from latentstrat.model import init_model
    from latentstrat.training import model_loss

    torch.set_default_dtype(torch.float64)
    opts = default_options().model_copy(update={"l2_embedding": 0.5, "l2_heads": 0, "l2_set": 0})
    model = init_model(8, 4, 4, 1, opts)
    red = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    blue = torch.tensor([[3, 4, 5], [4, 5, 6]], dtype=torch.long)
    pred = model(red, blue)

    assert pred.cont_z.shape == (2, 4)
    assert pred.bin_logits.shape == (2, 1)
    np.testing.assert_allclose(pred.red_pma_weights.detach().numpy().sum(axis=2), np.ones((2, 1)))

    cont = torch.zeros((2, 4))
    binary = torch.tensor([[1.0], [0.0]])
    loss, _, _ = model_loss(model, red, blue, cont, binary, opts)
    loss.backward()
    grad = model.team_embedding.weight.grad
    assert torch.linalg.norm(grad[:7]) > 0
    assert torch.allclose(grad[7], torch.zeros_like(grad[7]))
