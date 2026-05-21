import numpy as np
import torch

from latentstrat.config import default_options
from latentstrat.model import init_model, optimizer_parameter_groups
from latentstrat.training import create_optimizer, model_loss


def test_model_shapes_attention_and_heads():
    opts = default_options().model_copy(update={"latent_dim": 8, "attention_heads": 2})
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    red = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    blue = torch.tensor([[3, 4, 5], [4, 5, 6]], dtype=torch.long)

    pred = model(red, blue)

    assert pred.cont_z.shape == (2, 4)
    assert pred.bin_logits.shape == (2, 1)
    assert pred.z_match.shape == (2, 5 * opts.latent_dim)
    assert pred.red_pma_weights.shape == (2, 1, 3)
    np.testing.assert_allclose(
        pred.red_pma_weights.detach().numpy().sum(axis=2),
        np.ones((2, 1)),
        rtol=1e-6,
    )


def test_model_is_tolerant_to_alliance_slot_permutation():
    opts = default_options().model_copy(update={"attention_dropout": 0.0, "ffn_dropout": 0.0})
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    model.eval()
    red = torch.tensor([[0, 1, 2]], dtype=torch.long)
    blue = torch.tensor([[3, 4, 5]], dtype=torch.long)

    with torch.inference_mode():
        baseline = model(red, blue).cont_z
        permuted = model(red[:, [2, 0, 1]], blue[:, [1, 2, 0]]).cont_z

    assert torch.allclose(baseline, permuted, atol=1e-5)


def test_team_set_zero_slot_creates_true_zero_vector():
    opts = default_options()
    model = init_model(4, opts.latent_dim, 4, 1, opts)
    with torch.no_grad():
        model.team_embedding.weight[0].fill_(1.0)
    team_idx = torch.tensor([[0, 1, 2]], dtype=torch.long)

    zeroed = model.team_set(team_idx, zero_slot=1)
    unzeroed = model.team_set(team_idx, zero_slot=0)

    assert torch.allclose(zeroed[:, 0, :], torch.zeros_like(zeroed[:, 0, :]))
    assert not torch.allclose(unzeroed[:, 0, :], torch.zeros_like(unzeroed[:, 0, :]))


def test_optimizer_groups_do_not_decay_embeddings_biases_or_norms():
    opts = default_options()
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    groups = optimizer_parameter_groups(model, opts)

    decayed = {
        id(parameter) for group in groups if group["weight_decay"] for parameter in group["params"]
    }
    not_decayed = {
        id(parameter)
        for group in groups
        if not group["weight_decay"]
        for parameter in group["params"]
    }

    assert id(model.team_embedding.weight) in not_decayed
    assert id(model.cont_head.bias) in not_decayed
    assert id(model.sab.norm1.weight) in not_decayed
    assert id(model.cont_head.weight) in decayed


def test_inactive_embedding_row_is_not_changed_by_optimizer_step():
    opts = default_options().model_copy(
        update={"l2_embedding": 0.5, "l2_heads": 0.1, "l2_set": 0.1}
    )
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    optimizer = create_optimizer(model, opts)
    red = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    blue = torch.tensor([[3, 4, 5], [4, 5, 6]], dtype=torch.long)
    cont = torch.zeros((2, 4))
    binary = torch.tensor([[1.0], [0.0]])
    before = model.team_embedding.weight[7].detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = model_loss(model, red, blue, cont, binary, opts, torch.tensor([1.0]))
    loss.backward()
    optimizer.step()

    assert torch.allclose(model.team_embedding.weight[7], before)
