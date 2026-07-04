import torch

from sedd_mini.official_posttrain_rl import (
    DCoLTRollout,
    OfficialTrace,
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
