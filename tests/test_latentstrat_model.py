import numpy as np
import torch

from latentstrat.config import default_options
from latentstrat.model import init_model, optimizer_parameter_groups
from latentstrat.training import active_embedding_l2, create_optimizer, model_loss


def test_model_shapes_attention_and_heads():
    opts = default_options().model_copy(update={"latent_dim": 8, "attention_heads": 2})
    model = init_model(9, opts.latent_dim, 4, 1, opts)
    red = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    blue = torch.tensor([[4, 5, 6], [5, 6, 7]], dtype=torch.long)

    pred = model(red, blue)

    assert pred.cont_z.shape == (2, 4)
    assert pred.bin_logits.shape == (2, 1)
    assert pred.atomic_z.shape == (2, 2 * len(opts.atomic_count_targets))
    assert pred.foul_z.shape == (2, 2 * len(opts.foul_targets))
    assert pred.bonus_logits.shape == (2, 2 * len(opts.bonus_binary_targets))
    assert pred.special_logits.shape == (2, 2 * len(opts.special_binary_targets))
    assert pred.z_match.shape == (2, 5 * opts.latent_dim)
    assert pred.red_pma_weights.shape == (2, 1, 3)
    assert pred.endgame_logits.shape == (2, 6, 3)
    assert pred.award_logits.shape == (2, 6, len(opts.award_targets))
    np.testing.assert_allclose(
        pred.red_pma_weights.detach().numpy().sum(axis=2),
        np.ones((2, 1)),
        rtol=1e-6,
    )


def test_model_is_tolerant_to_alliance_slot_permutation():
    opts = default_options().model_copy(update={"attention_dropout": 0.0, "ffn_dropout": 0.0})
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    model.eval()
    red = torch.tensor([[1, 2, 3]], dtype=torch.long)
    blue = torch.tensor([[4, 5, 6]], dtype=torch.long)

    with torch.inference_mode():
        baseline = model(red, blue).cont_z
        permuted = model(red[:, [2, 0, 1]], blue[:, [1, 2, 0]]).cont_z

    assert torch.allclose(baseline, permuted, atol=1e-5)


def test_team_set_zero_slot_routes_to_learned_ghost_row():
    opts = default_options()
    model = init_model(4, opts.latent_dim, 4, 1, opts)
    model.eval()
    with torch.no_grad():
        model.team_embedding.weight[0].fill_(7.0)
        model.team_embedding.weight[1].fill_(1.0)
    team_idx = torch.tensor([[1, 2, 3]], dtype=torch.long)

    zeroed, effective_missing, _ = model.team_set(team_idx, zero_slot=1)
    unzeroed, _, _ = model.team_set(team_idx, zero_slot=0)

    assert bool(effective_missing[0, 0])
    assert torch.allclose(zeroed[:, 0, :], torch.full_like(zeroed[:, 0, :], 7.0))
    assert not torch.allclose(unzeroed[:, 0, :], torch.zeros_like(unzeroed[:, 0, :]))


def test_missing_slots_use_ghost_row_without_attention_masking():
    opts = default_options().model_copy(update={"team_dropout_rate": 0.0})
    model = init_model(6, opts.latent_dim, 4, 1, opts)
    model.eval()
    with torch.no_grad():
        model.team_embedding.weight[0].fill_(4.0)
    red = torch.tensor([[0, 1, 2]], dtype=torch.long)
    blue = torch.tensor([[3, 4, 5]], dtype=torch.long)
    missing = torch.tensor([[True, False, False]])

    slot_vectors, effective_missing, dropout = model.team_set(red, missing_mask=missing)
    pred = model(red, blue, red_missing_mask=missing, pma_mode="uniform")

    assert torch.allclose(slot_vectors[:, 0, :], torch.full_like(slot_vectors[:, 0, :], 4.0))
    assert bool(effective_missing[0, 0])
    assert not dropout.any()
    assert pred.red_missing_mask.tolist() == [[True, False, False]]
    np.testing.assert_allclose(
        pred.red_pma_weights.detach().numpy(),
        np.full((1, 1, 3), 1.0 / 3.0),
        rtol=1e-6,
    )


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

    assert id(model.Z_base.weight) in not_decayed
    assert id(model.Z_event.weight) in not_decayed
    assert id(model.cont_head.bias) in not_decayed
    assert id(model.sab.norm1.weight) in not_decayed
    assert id(model.cont_head.weight) in decayed
    assert id(model.atomic_head.weight) in decayed
    assert id(model.team_value_head.linear.weight) in decayed
    assert {group["weight_decay"] for group in groups if group["weight_decay"]} == {1e-4}


def test_optimizer_groups_skip_frozen_embeddings():
    opts = default_options()
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    model.Z_base.weight.requires_grad = False
    model.Z_event.weight.requires_grad = False

    groups = optimizer_parameter_groups(model, opts)
    grouped = {id(parameter) for group in groups for parameter in group["params"]}

    assert id(model.Z_base.weight) not in grouped
    assert id(model.Z_event.weight) not in grouped
    assert id(model.cont_head.weight) in grouped
    assert id(model.sab.attention.in_proj_weight) in grouped


def test_active_embedding_l2_is_zero_for_frozen_embeddings():
    opts = default_options()
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    with torch.no_grad():
        model.Z_base.weight.fill_(2.0)
    red = torch.tensor([[1, 2, 3]], dtype=torch.long)
    blue = torch.tensor([[4, 5, 6]], dtype=torch.long)

    unfrozen = active_embedding_l2(model, red, blue, coefficient=0.5)
    model.Z_base.weight.requires_grad = False
    frozen = active_embedding_l2(model, red, blue, coefficient=0.5)

    assert float(unfrozen.detach()) > 0
    assert float(frozen.detach()) == 0.0


def test_value_heads_return_rank_scalars():
    opts = default_options().model_copy(update={"team_dropout_rate": 0.0})
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    teams = torch.tensor([1, 2, 3], dtype=torch.long)
    alliance = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    team_values = model.team_value(teams)
    alliance_values = model.alliance_value(alliance)

    assert team_values.shape == (3,)
    assert alliance_values.shape == (2,)


def test_inactive_embedding_row_is_not_changed_by_optimizer_step():
    opts = default_options().model_copy(
        update={"l2_embedding": 0.5, "l2_heads": 0.1, "l2_set": 0.1}
    )
    model = init_model(9, opts.latent_dim, 4, 1, opts)
    optimizer = create_optimizer(model, opts)
    red = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    blue = torch.tensor([[4, 5, 6], [5, 6, 7]], dtype=torch.long)
    cont = torch.zeros((2, 4))
    binary = torch.tensor([[1.0], [0.0]])
    before = model.team_embedding.weight[8].detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = model_loss(model, red, blue, cont, binary, opts, torch.tensor([1.0]))
    loss.backward()
    optimizer.step()

    assert torch.allclose(model.team_embedding.weight[8], before)


def test_random_team_dropout_is_training_only_and_keeps_one_slot():
    opts = default_options().model_copy(update={"team_dropout_rate": 1.0})
    model = init_model(5, opts.latent_dim, 4, 1, opts)
    teams = torch.tensor([[1, 2, 3]], dtype=torch.long)

    model.train()
    with torch.no_grad():
        model.team_embedding.weight[0].fill_(5.0)
    slot_vectors, effective_missing, dropout = model.team_set(teams)

    assert dropout.sum().item() == 2
    assert effective_missing.sum().item() == 2
    assert torch.allclose(
        slot_vectors[effective_missing],
        torch.full_like(slot_vectors[effective_missing], 5.0),
    )

    model.eval()
    _, eval_missing, eval_dropout = model.team_set(teams)

    assert not eval_dropout.any()
    assert not eval_missing.any()


def test_siamese_win_head_is_antisymmetric():
    opts = default_options().model_copy(update={"team_dropout_rate": 0.0})
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    model.eval()
    red = torch.tensor([[1, 2, 3]], dtype=torch.long)
    blue = torch.tensor([[4, 5, 6]], dtype=torch.long)

    with torch.inference_mode():
        forward = model(red, blue).bin_logits
        swapped = model(blue, red).bin_logits

    assert torch.allclose(forward, -swapped, atol=1e-6)


def test_delta_integration_gate_uses_delta_weeks():
    opts = default_options()
    model = init_model(5, opts.latent_dim, 4, 1, opts)
    z_base = torch.ones((1, opts.latent_dim))
    z_event = torch.full((1, opts.latent_dim), 0.5)

    one_week = model.delta_integration_gate(z_base, z_event, torch.tensor([1.0]))
    five_weeks = model.delta_integration_gate(z_base, z_event, torch.tensor([5.0]))

    assert not torch.allclose(one_week, five_weeks)
