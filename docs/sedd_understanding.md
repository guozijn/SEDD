# SEDD Understanding Notes

## One-Sentence Summary

SEDD replaces autoregressive next-token prediction with a continuous-time discrete
diffusion process over tokens, and trains a bidirectional network to estimate
probability ratios between neighboring discrete states using score entropy.

## Problem Setting

Continuous diffusion models learn a score, usually interpreted as a gradient of
log-density. Text is discrete, so ordinary gradients do not exist. Lou, Meng, and
Ermon reframe the score as a set of concrete ratios:

```text
s_theta(x, t)_y ~= p_t(y) / p_t(x)
```

where `x` is the current discrete state and `y` is a neighboring state reachable
under a CTMC transition graph. This ratio is enough to construct the reverse
process because the reverse CTMC rate is the forward transpose rate multiplied
by the ratio estimate.

## Forward Process

The original paper supports general graph choices. This repo uses the practical
absorbing graph:

- Data tokens stay unchanged with probability `exp(-sigma(t))`.
- Otherwise they jump to a special absorbing `<mask>` token.
- Prompt tokens can be protected by setting their `loss_mask` to false.

The log-linear noise schedule follows the official SEDD implementation:

```text
sigma(t) = -log(1 - (1 - eps) t)
dsigma/dt = (1 - eps) / (1 - (1 - eps) t)
```

This schedule makes the mask probability approach one as `t -> 1`.

## Score Entropy Loss

For absorbing diffusion, only positions currently at `<mask>` contribute. Let
`x0` be the clean token and `x_t` be the noised token. For a masked position,
the target ratio for the clean token is:

```text
r(t) = 1 / expm1(sigma(t))
```

The loss implemented in `src/sedd_mini/diffusion.py` is:

```text
sum_y exp(log_score_y)
- r(t) * log_score_{x0}
+ r(t) * (log r(t) - 1)
```

then multiplied by `dsigma/dt` and averaged over active masked positions.

Interpretation:

- `exp(log_score_y)` keeps the predicted ratios positive.
- The negative term pulls the clean token ratio toward the analytic target.
- The constant term makes the divergence zero at the optimum.
- This is a discrete analogue of score matching, but it avoids forcing a
  Euclidean gradient onto categorical data.

## Why Bidirectional Attention Matters

Autoregressive LMs factor text left to right. SEDD predicts denoising ratios from
the whole corrupted sequence, so a token can use left and right context. That is
why infilling is natural: keep known tokens fixed, mask unknown spans, and run
the reverse process.

## Pretraining Pipeline

`sedd-prepare --mode pretrain` concatenates byte-tokenized text into fixed
length blocks. All non-pad tokens are eligible for corruption and loss.

`sedd-train --config configs/tiny_pretrain.yaml` trains with score entropy:

1. Sample `t ~ Uniform(eps, 1 - eps)`.
2. Convert `t` to `sigma` and `dsigma`.
3. Mask tokens using the absorbing transition.
4. Predict log-ratios with the Transformer.
5. Apply score entropy on masked, loss-enabled positions.

## Supervised Fine-Tuning

SFT uses the same score entropy loss but changes the loss mask:

```text
User: <prompt>
Assistant: <response>
```

Prompt tokens remain visible and have zero loss. Response tokens are corruptible
and trainable. This adapts the model to instruction following without changing
the diffusion objective.

For official checkpoints, `src/sedd_mini/official_finetune.py` applies the same
idea to `louaaron/sedd-small` or `louaaron/sedd-medium`:

- Data is tokenized with GPT-2 BPE.
- Prompt tokens are clamped and uncorrupted.
- Response tokens are noised through the upstream absorbing graph.
- The upstream model, graph, and log-linear noise schedule are reused.
- Checkpoints save the fine-tuned state dict plus the base HF model path, so the
  official backend can serve the fine-tuned model.

## Evaluation

The repo reports three metrics:

- `score_entropy`: direct validation objective.
- `denoise_ce`: cross-entropy when a random subset is masked.
- `pseudo_ppl`: cross-entropy/perplexity when all loss positions are masked.

The pseudo-perplexity is not the exact generative likelihood bound from the
paper. It is a cheap interview-scale proxy for denoising quality.

## Sampling

The official SEDD sampler can use Euler or analytic CTMC predictors. This repo
supports two sampling backends:

- `mini`: a practical masked-denoising sampler:

1. Encode the prompt.
2. Append `<mask>` tokens for the response budget.
3. At each step, predict distributions for masked response positions.
4. Fill a scheduled subset of remaining masks.
5. Decode the response span.

This is not a full likelihood sampler, but it demonstrates the central
bidirectional denoising behavior and is robust for small checkpoints.

- `official`: a lazy wrapper around `louaaron/Score-Entropy-Discrete-Diffusion`
  that loads `louaaron/sedd-small` or `louaaron/sedd-medium` and calls the
  upstream analytic sampler. This path requires CUDA/flash-attn, so it is meant
  for the remote GPU rather than local CPU/MPS smoke tests.

## RL Post-Training Route

Autoregressive RLHF optimizes log-probabilities of sampled tokens. For discrete
diffusion, the actions are denoising choices over a trajectory. This repo
implements a SEPO-inspired baseline:

```text
loss = - advantage * sum_t log pi_theta(action_t | x_t, t)
       + beta_kl * sum_t (log pi_theta - log pi_ref)
       - beta_entropy * entropy
```

The reward is computed on the final decoded sample. The included reward functions
are intentionally transparent heuristics, but the same interface can call a
preference model, a unit-test checker, a math verifier, or a human-feedback
model.

For official checkpoints, `src/sedd_mini/official_posttrain_rl.py` implements the
same trajectory view using GPT-2 token IDs and the official score model. It
samples response tokens from mask positions, accumulates denoising log-probs,
uses a frozen official reference model for KL control, and saves a loadable
official checkpoint.

Why this is a reasonable adaptation:

- The denoising process is an MDP: state is the partially denoised sequence,
  action is the token filled at selected positions, terminal reward scores the
  final sequence.
- Score-network probabilities define the policy for sampled denoising actions.
- A frozen reference checkpoint prevents reward hacking and preserves the SFT
  distribution.

Limitations:

- The sampler is a masked-denoising approximation, not exact CTMC likelihood
  training.
- REINFORCE has high variance; stronger versions should add group baselines,
  reward normalization, PPO-style clipping over denoising actions, and a more
  faithful SEPO objective.
- A serious RL run needs a real reward model or task verifier.

## What I Would Improve With More Time

1. Swap byte tokenization for GPT-2 BPE and load `louaaron/sedd-small` weights.
2. Add LoRA adapters around the official SEDD DDiT blocks for 16 GB fine-tuning.
3. Add exact likelihood-bound evaluation from the paper.
4. Implement group-relative SEPO/GRPO over multiple samples per prompt.
5. Distill the official checkpoint into the mini backend for cheaper demos.

## Sources

- Lou, Meng, Ermon, "Discrete Diffusion Modeling by Estimating the Ratios of the
  Data Distribution": https://arxiv.org/abs/2310.16834
- Official implementation: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- Hugging Face model cards: https://huggingface.co/louaaron/sedd-small and
  https://huggingface.co/louaaron/sedd-medium
- Zekri and Boullé, "Fine-Tuning Discrete Diffusion Models with Policy Gradient
  Methods": https://arxiv.org/abs/2502.01384
