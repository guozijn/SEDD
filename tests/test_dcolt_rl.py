import torch

from sedd_mini.official_posttrain_rl import (
    DCoLTRollout,
    OfficialTrace,
    build_parser,
    dcolt_loss,
    filtered_log_probs,
    group_normalized_advantages,
)


def test_group_normalized_advantages_zero_mean_and_zero_variance():
    advantages = group_normalized_advantages([0.0, 1.0, 0.0, 1.0])
    assert len(advantages) == 4
    assert abs(sum(advantages)) < 1.0e-5
    assert advantages[1] > 0
    assert advantages[0] < 0

    assert group_normalized_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]


def test_dcolt_loss_backpropagates_through_logprob():
    logprob = torch.tensor(-2.0, requires_grad=True)
    trace = OfficialTrace(
        ids=torch.zeros((1, 4), dtype=torch.long),
        response_ids=torch.zeros(2, dtype=torch.long),
        logprob_sum=logprob,
        ref_logprob_sum=torch.tensor(-2.5),
        entropy_sum=torch.tensor(1.0),
        action_count=2,
    )
    rollout = DCoLTRollout(trace=trace, reward=1.0, prediction="A", text="Answer: A", gold="A")

    loss, metrics = dcolt_loss([rollout], [1.0], clip_eps=0.2, beta=0.02, entropy_coef=0.0)
    loss.backward()

    assert logprob.grad is not None
    assert metrics["policy_loss"] < 0
    assert "kl_term" in metrics


def test_filtered_log_probs_keeps_entropy_finite_with_masked_logits():
    logits = torch.tensor([[10.0, 1.0, -1.0, -2.0], [float("-inf"), 0.0, -3.0, -4.0]])
    log_probs = filtered_log_probs(logits, top_k=1, top_p=0.9)
    probs = log_probs.exp()
    entropy = -(probs * torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))).sum(dim=-1)

    assert torch.isfinite(entropy).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1.0e-5)


def test_dcolt_cli_defaults_match_notebook_artifacts():
    args = build_parser().parse_args([])

    assert args.model_path == "runs/arc_models/arc_lora_sft/checkpoint_last.pt"
    assert args.reference_model_path == "runs/arc_models/base/checkpoint_base.pt"
    assert args.records_path == "data/processed/arc_challenge_rl_train.jsonl"
    assert args.out_dir == "runs/arc_models/arc_dcolt_rl"
    assert args.updates == 100
    assert args.batch_size == 1
    assert args.num_generations == 4
    assert args.repeat_times == 1
    assert args.sample_steps == 4
    assert args.max_new_tokens == 12
    assert args.clip_eps == 0.2
    assert args.beta == 0.02
    assert args.save_every == 0
    assert args.log_every == 5
