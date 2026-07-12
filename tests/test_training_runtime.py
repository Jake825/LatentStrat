import random

import numpy as np
import pytest
import torch

from latentstrat.training_runtime import (
    OptimizationController,
    TrainingRuntimeConfig,
    build_lr_scheduler,
    config_fingerprint,
    load_resume_checkpoint,
    load_training_state,
    resume_payload,
    save_resume_checkpoint,
    seed_everything,
    seeded_generator,
    stable_seed,
    validate_resume_checkpoint,
)


def _model_optimizer():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    return model, optimizer


def test_stable_seed_is_repeatable_and_stream_specific():
    assert stable_seed(7, "season", 1) == stable_seed(7, "season", 1)
    assert stable_seed(7, "season", 1) != stable_seed(7, "season", 2)


def test_seed_everything_seeds_python_numpy_and_torch():
    seed_everything(123)
    first = (random.random(), np.random.random(), torch.rand(()))
    seed_everything(123)
    second = (random.random(), np.random.random(), torch.rand(()))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_controller_accumulates_and_flushes_partial_window():
    model, optimizer = _model_optimizer()
    config = TrainingRuntimeConfig(
        gradient_accumulation_steps=2, max_grad_norm=100.0, scheduler="none"
    )
    controller = OptimizationController(model, optimizer, config)
    x = torch.ones((1, 1))
    for index in range(3):
        loss = model(x).square().mean() + 1.0
        controller.backward(loss, microbatch_index=index, microbatch_count=3)
    assert controller.optimizer_step == 2
    assert len(controller.step_results) == 2


def test_controller_clips_and_rejects_nonfinite_loss():
    model, optimizer = _model_optimizer()
    with torch.no_grad():
        model.weight.fill_(10.0)
    controller = OptimizationController(
        model,
        optimizer,
        TrainingRuntimeConfig(max_grad_norm=0.01, scheduler="none"),
    )
    loss = model(torch.ones((1, 1))).square().mean()
    result = controller.backward(loss, microbatch_index=0, microbatch_count=1)
    assert result is not None and result.clipped
    with pytest.raises(FloatingPointError):
        controller.backward(
            torch.tensor(float("nan"), requires_grad=True),
            microbatch_index=0,
            microbatch_count=1,
        )


def test_scheduler_steps_only_with_optimizer_steps():
    model, optimizer = _model_optimizer()
    config = TrainingRuntimeConfig(
        gradient_accumulation_steps=2,
        max_grad_norm=100.0,
        scheduler="cosine",
        lr_eta_min=0.001,
    )
    scheduler = build_lr_scheduler(optimizer, config, epochs=1, microbatches_per_epoch=4)
    controller = OptimizationController(model, optimizer, config, scheduler=scheduler)
    for index in range(4):
        controller.backward(
            model(torch.ones((1, 1))).square().mean(),
            microbatch_index=index,
            microbatch_count=4,
        )
    assert controller.optimizer_step == 2
    assert scheduler is not None and scheduler.last_epoch == 2


def test_epoch_resume_restores_identical_next_step(tmp_path):
    seed_everything(99)
    model, optimizer = _model_optimizer()
    config = TrainingRuntimeConfig(max_grad_norm=10.0, scheduler="cosine")
    scheduler = build_lr_scheduler(optimizer, config, epochs=2, microbatches_per_epoch=1)
    generator = seeded_generator(99, "test")
    controller = OptimizationController(model, optimizer, config, scheduler=scheduler)
    controller.backward(
        model(torch.ones((1, 1))).square().mean(),
        microbatch_index=0,
        microbatch_count=1,
    )
    resolved = {"epochs": 2, **config.__dict__}
    payload = resume_payload(
        trainer="test",
        phase="fit",
        completed_epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_step=controller.optimizer_step,
        history=[{"epoch": 1}],
        config=resolved,
        source_fingerprint="fixture",
        loader_generator=generator,
    )
    path = save_resume_checkpoint(tmp_path / "latest.ckpt", payload)
    loaded = load_resume_checkpoint(path)
    validate_resume_checkpoint(
        loaded,
        trainer="test",
        phase="fit",
        config=resolved,
        source_fingerprint="fixture",
    )

    resumed_model, resumed_optimizer = _model_optimizer()
    resumed_scheduler = build_lr_scheduler(
        resumed_optimizer, config, epochs=2, microbatches_per_epoch=1
    )
    resumed_generator = seeded_generator(0, "placeholder")
    load_training_state(
        loaded,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        loader_generator=resumed_generator,
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.parameters(), resumed_model.parameters(), strict=True)
    )
    assert optimizer.state_dict() == resumed_optimizer.state_dict()
    assert scheduler is not None and resumed_scheduler is not None
    assert scheduler.state_dict() == resumed_scheduler.state_dict()
    assert torch.equal(generator.get_state(), resumed_generator.get_state())


def test_resume_validation_reports_configuration_mismatch():
    payload = {
        "trainer": "test",
        "phase": "fit",
        "config_fingerprint": config_fingerprint({"epochs": 1}),
        "source_fingerprint": "fixture",
    }
    with pytest.raises(ValueError, match="config_fingerprint"):
        validate_resume_checkpoint(
            payload,
            trainer="test",
            phase="fit",
            config={"epochs": 2},
            source_fingerprint="fixture",
        )
