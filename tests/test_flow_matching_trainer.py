import inspect
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

deepspeed_stub = types.ModuleType("deepspeed")
deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
sys.modules.setdefault("deepspeed", deepspeed_stub)

flow_match_spec = importlib.util.spec_from_file_location(
    "flow_match_for_trainer_tests", ROOT / "diffsynth" / "diffusion" / "flow_match.py"
)
flow_match_module = importlib.util.module_from_spec(flow_match_spec)
flow_match_spec.loader.exec_module(flow_match_module)
FlowMatchScheduler = flow_match_module.FlowMatchScheduler
from keyframegen.train import agibot_non_ar_train as trainer


def make_wan_pipe(num_steps=1000, shift=5.0):
    pipe = SimpleNamespace(scheduler=FlowMatchScheduler("Wan"))
    cfg = trainer.FlowMatchingConfig(
        mode="diffsynth_wan",
        num_train_timesteps=num_steps,
        sigma_shift=shift,
        min_timestep_boundary=0.0,
        max_timestep_boundary=1.0,
        use_training_weight=True,
    )
    trainer.initialize_flow_matching_scheduler(pipe, cfg)
    return pipe, cfg


def test_diffsynth_wan_scheduler_initializes_1000_shift5():
    pipe, cfg = make_wan_pipe()
    assert len(pipe.scheduler.timesteps) == 1000
    assert len(pipe.scheduler.sigmas) == 1000
    assert hasattr(pipe.scheduler, "linear_timesteps_weights")
    assert torch.allclose(
        pipe.scheduler.timesteps.float(),
        pipe.scheduler.sigmas.float() * 1000.0,
        atol=1e-3,
        rtol=1e-4,
    )
    assert cfg.sigma_shift == 5.0


def test_shared_helper_matches_scheduler_direct_calls():
    pipe, cfg = make_wan_pipe()
    target = torch.randn(1, 2, 4, 3, 3)
    noise = torch.randn_like(target)
    timestep_index = 123

    fm = trainer.build_flow_matching_batch(
        pipe=pipe,
        target_latents=target,
        first_frame_latents=None,
        fuse_vae_embedding_in_latents=False,
        flow_cfg=cfg,
        deterministic_noise=noise,
        deterministic_timestep_index=timestep_index,
    )
    timestep = pipe.scheduler.timesteps[timestep_index].to(dtype=target.dtype).reshape(1)

    assert torch.allclose(
        fm.noisy_latents,
        pipe.scheduler.add_noise(target, noise, timestep),
    )
    assert torch.allclose(
        fm.training_target,
        pipe.scheduler.training_target(target, noise, timestep),
    )
    assert torch.allclose(
        fm.training_weight.cpu(),
        pipe.scheduler.training_weight(timestep).reshape(()).float(),
    )


def test_uniform_linear_reproduces_old_equations():
    pipe = SimpleNamespace(scheduler=None)
    cfg = trainer.FlowMatchingConfig(
        mode="uniform_linear",
        min_timestep=250.0,
        max_timestep=250.0,
        use_training_weight=False,
    )
    target = torch.randn(1, 2, 3, 2, 2)
    noise = torch.randn_like(target)
    timestep = torch.tensor([250.0])

    fm = trainer.build_flow_matching_batch(
        pipe=pipe,
        target_latents=target,
        first_frame_latents=None,
        fuse_vae_embedding_in_latents=False,
        flow_cfg=cfg,
        deterministic_noise=noise,
        deterministic_timestep=timestep,
    )
    sigma = timestep.view(1, 1, 1, 1, 1) / 1000.0

    assert torch.allclose(fm.noisy_latents, (1.0 - sigma) * target + sigma * noise)
    assert torch.allclose(fm.training_target, noise - target)
    assert fm.timestep_index is None
    assert fm.training_weight.item() == 1.0


def test_first_frame_slot_excluded_from_loss_denominator():
    pipe, cfg = make_wan_pipe(num_steps=16)
    target = torch.zeros(1, 1, 4, 2, 2)
    first = torch.ones(1, 1, 1, 2, 2)
    noise = torch.ones_like(target)

    fm = trainer.build_flow_matching_batch(
        pipe=pipe,
        target_latents=target,
        first_frame_latents=first,
        fuse_vae_embedding_in_latents=True,
        flow_cfg=cfg,
        deterministic_noise=noise,
        deterministic_timestep_index=3,
    )
    error = torch.ones_like(fm.training_target)
    assert fm.loss_mask[:, :, 0].sum().item() == 0
    assert fm.loss_mask.sum().item() == 12
    assert trainer.masked_mse_loss(error, fm.loss_mask).item() == 1.0


def test_validation_seed_and_sample_index_are_deterministic():
    pipe, cfg = make_wan_pipe()
    shape = torch.Size([1, 2, 3, 2, 2])

    left = trainer.fixed_validation_noise_and_timestep(
        shape, 7, 20260715, cfg, "cpu", torch.float32, pipe.scheduler
    )
    right = trainer.fixed_validation_noise_and_timestep(
        shape, 7, 20260715, cfg, "cpu", torch.float32, pipe.scheduler
    )

    assert torch.equal(left[0], right[0])
    assert left[1] == right[1]


def test_different_validation_sample_indices_normally_differ():
    pipe, cfg = make_wan_pipe()
    shape = torch.Size([1, 2, 3, 2, 2])

    left = trainer.fixed_validation_noise_and_timestep(
        shape, 1, 20260715, cfg, "cpu", torch.float32, pipe.scheduler
    )
    right = trainer.fixed_validation_noise_and_timestep(
        shape, 2, 20260715, cfg, "cpu", torch.float32, pipe.scheduler
    )

    assert not torch.equal(left[0], right[0]) or left[1] != right[1]


def test_training_and_validation_call_same_shared_helper():
    train_source = inspect.getsource(trainer.train)
    validation_source = inspect.getsource(trainer.run_validation)

    assert "build_flow_matching_batch(" in train_source
    assert "build_flow_matching_batch(" in validation_source
    assert "(1.0 - sigma)" not in train_source
    assert "(1.0 - sigma)" not in validation_source


def test_boundary_conversion_is_never_empty():
    for num_timesteps in (2, 16, 1000):
        for min_boundary, max_boundary in (
            (0.0, 1.0),
            (0.0, 0.001),
            (0.999, 1.0),
            (0.2, 0.2001),
        ):
            left, right = trainer.scheduler_index_bounds(
                num_timesteps, min_boundary, max_boundary
            )
            assert 0 <= left < right <= num_timesteps


def test_weighted_loss_is_unweighted_times_scheduler_weight():
    pipe, cfg = make_wan_pipe(num_steps=16)
    target = torch.zeros(1, 1, 2, 2, 2)
    noise = torch.ones_like(target)
    fm = trainer.build_flow_matching_batch(
        pipe=pipe,
        target_latents=target,
        first_frame_latents=None,
        fuse_vae_embedding_in_latents=False,
        flow_cfg=cfg,
        deterministic_noise=noise,
        deterministic_timestep_index=4,
    )
    error = torch.full_like(target, 2.0)
    unweighted = trainer.masked_mse_loss(error, fm.loss_mask)
    weighted = unweighted * fm.training_weight.to(unweighted)

    assert torch.allclose(weighted, unweighted * fm.training_weight.to(unweighted))


def test_full_resume_flow_matching_mismatch_is_detected():
    current = trainer.FlowMatchingConfig(
        mode="diffsynth_wan",
        num_train_timesteps=1000,
        sigma_shift=5.0,
    )
    checkpoint = {
        "mode": "diffsynth_wan",
        "num_train_timesteps": 1000,
        "sigma_shift": 3.0,
    }

    try:
        trainer.assert_resume_flow_matching_compatible(checkpoint, current)
    except ValueError as exc:
        assert "sigma_shift" in str(exc)
    else:
        raise AssertionError("Expected resume mismatch to raise ValueError")


class FakePipe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(3, 3, bias=False)

    def model_fn(self, dit, latents=None, context=None, **kwargs):
        return dit(latents) + context


def test_training_wrapper_forward_matches_direct_model_fn():
    pipe = FakePipe()
    wrapper = trainer.KeyframeWanTrainingModule(pipe)
    latents = torch.randn(2, 3)
    context = torch.randn(2, 3)

    direct = pipe.model_fn(dit=pipe.dit, latents=latents, context=context)
    wrapped = wrapper(latents=latents, context=context)

    assert torch.allclose(direct, wrapped)


def test_trainable_parameter_ids_do_not_change_after_engine_wrap():
    pipe = FakePipe()
    for parameter in pipe.dit.parameters():
        parameter.requires_grad_(True)
    wrapper = trainer.KeyframeWanTrainingModule(pipe)
    expected = trainer.trainable_parameter_ids(pipe.dit)
    engine = SimpleNamespace(module=wrapper)

    trainer.assert_training_module_ownership(
        training_module=wrapper,
        engine=engine,
        expected_trainable_ids=expected,
        rank=0,
        local_rank=0,
        enforce_cuda_device=False,
    )


def test_training_and_validation_forward_through_engine_not_pipe_model_fn():
    train_source = inspect.getsource(trainer.train)
    validation_source = inspect.getsource(trainer.run_validation)

    assert "pred = engine(**model_kwargs)" in train_source
    assert "pred = engine(**model_kwargs)" in validation_source
    assert "pipe.model_fn(**model_kwargs)" not in train_source
    assert "pipe.model_fn(**model_kwargs)" not in validation_source


def test_no_loop_dit_load_models_to_device_calls_remain():
    train_source = inspect.getsource(trainer.train)
    validation_source = inspect.getsource(trainer.run_validation)

    assert 'pipe.load_models_to_device(["dit"])' not in validation_source
    assert train_source.count('pipe.load_models_to_device(["dit"])') == 1


def make_optimizer_for_scheduler_tests():
    p1 = torch.nn.Parameter(torch.ones(1))
    p2 = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD(
        [
            {"params": [p1], "lr": 0.1, "name": "a"},
            {"params": [p2], "lr": 0.01, "name": "b"},
        ]
    )
    return optimizer


def test_constant_scheduler_keeps_base_lr():
    optimizer = make_optimizer_for_scheduler_tests()
    cfg = trainer.LRSchedulerConfig(type="constant", warmup_steps=0)
    scheduler = trainer.build_lr_scheduler(optimizer, cfg)

    for _ in range(3):
        optimizer.step()
        scheduler.step()

    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.01]


def test_warmup_preserves_relative_lr_ratios():
    optimizer = make_optimizer_for_scheduler_tests()
    cfg = trainer.LRSchedulerConfig(type="constant_with_warmup", warmup_steps=4)
    scheduler = trainer.build_lr_scheduler(optimizer, cfg)

    optimizer.step()
    scheduler.step()

    lrs = [group["lr"] for group in optimizer.param_groups]
    assert abs((lrs[0] / lrs[1]) - 10.0) < 1e-6


def test_warmup_reaches_base_lr_at_expected_update():
    optimizer = make_optimizer_for_scheduler_tests()
    cfg = trainer.LRSchedulerConfig(type="constant_with_warmup", warmup_steps=3)
    scheduler = trainer.build_lr_scheduler(optimizer, cfg)

    for _ in range(3):
        optimizer.step()
        scheduler.step()

    lrs = [group["lr"] for group in optimizer.param_groups]
    assert abs(lrs[0] - 0.1) < 1e-8
    assert abs(lrs[1] - 0.01) < 1e-8


def test_scheduler_advances_only_on_optimizer_boundaries():
    optimizer = make_optimizer_for_scheduler_tests()
    cfg = trainer.LRSchedulerConfig(type="constant_with_warmup", warmup_steps=2)
    scheduler = trainer.build_lr_scheduler(optimizer, cfg)
    initial_epoch = scheduler.last_epoch

    for is_boundary in [False, True, False, True]:
        if is_boundary:
            optimizer.step()
            scheduler.step()

    assert scheduler.last_epoch == initial_epoch + 2


def test_scheduler_state_restores_from_checkpoint_state():
    optimizer = make_optimizer_for_scheduler_tests()
    cfg = trainer.LRSchedulerConfig(type="constant_with_warmup", warmup_steps=3)
    scheduler = trainer.build_lr_scheduler(optimizer, cfg)
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()

    optimizer2 = make_optimizer_for_scheduler_tests()
    scheduler2 = trainer.build_lr_scheduler(optimizer2, cfg)
    scheduler2.load_state_dict(state)

    assert scheduler2.last_epoch == scheduler.last_epoch
    assert scheduler2.state_dict() == scheduler.state_dict()


def test_single_process_gradient_norm_matches_torch_reference():
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(3))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = torch.tensor([0.0, 12.0, 0.0])
    groups = {
        "self_attn_lora": [("p1", p1)],
        "cross_attn_lora": [("p2", p2)],
        "cross_attn_full": [],
        "full_extra": [],
    }

    stats = trainer.gradient_group_stats(
        groups, distributed=False, aggregate_across_ranks=False, device="cpu"
    )
    global_norm = (stats["self_attn_lora"]["sum_sq"] + stats["cross_attn_lora"]["sum_sq"]) ** 0.5
    reference = torch.cat([p1.grad.flatten(), p2.grad.flatten()]).norm().item()

    assert abs(global_norm - reference) < 1e-6


def test_nonfinite_gradients_are_detected():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([float("inf"), 1.0])
    groups = {
        "self_attn_lora": [("p", p)],
        "cross_attn_lora": [],
        "cross_attn_full": [],
        "full_extra": [],
    }

    stats = trainer.gradient_group_stats(
        groups, distributed=False, aggregate_across_ranks=False, device="cpu"
    )

    assert stats["self_attn_lora"]["nonfinite_count"] == 1


def test_empty_optimizer_groups_do_not_emit_metrics():
    groups = {
        "self_attn_lora": [],
        "cross_attn_lora": [],
        "cross_attn_full": [],
        "full_extra": [],
    }

    stats = trainer.gradient_group_stats(
        groups, distributed=False, aggregate_across_ranks=False, device="cpu"
    )

    assert stats == {}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
