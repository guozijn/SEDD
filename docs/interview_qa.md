# Interview Q&A Bank

## Theory

**Q: What is the main idea of SEDD?**  
A: Learn concrete probability ratios `p_t(y) / p_t(x)` on a discrete CTMC,
instead of trying to define a continuous score on categorical tokens.

**Q: Why is score entropy better suited than L2 score matching for discrete data?**  
A: The target is a positive ratio. Score entropy uses a divergence based on
`-log` and predicts log-ratios, which naturally enforces positivity through
exponentiation. Plain L2 does not encode that structure.

**Q: What does the score network output?**  
A: For each position and candidate token, it outputs a log-ratio. During
training the loss uses these log-ratios; during sampling they are exponentiated
to recover positive ratio estimates.

**Q: How does the absorbing graph work?**  
A: Each clean token either stays unchanged or jumps to the absorbing mask token.
Once masked under the forward process, it remains masked. The reverse process
uses the score ratios to move from mask back to data tokens.

**Q: Why does SEDD enable infilling?**  
A: The model is bidirectional. Known tokens can be clamped anywhere in the
sequence, while unknown tokens are masked and denoised using both left and right
context.

**Q: How is SEDD different from BERT MLM?**  
A: BERT learns a fixed-noise denoising classifier. SEDD learns time-dependent
ratio estimates that define a full reverse diffusion process.

**Q: Is SEDD autoregressive?**  
A: No. It does not factor `p(x)` left-to-right. It generates by reversing a
discrete noising process, often allowing parallel or semi-parallel token updates.

**Q: What is the reverse CTMC rate?**  
A: Conceptually, the reverse rate from state `x` to neighbor `y` is the forward
transpose rate multiplied by `p_t(y) / p_t(x)`, which the score network estimates.

## Implementation

**Q: What did you implement faithfully from the paper?**  
A: The absorbing forward process, log-linear noise schedule, score entropy loss,
bidirectional score model, SFT through response-only loss masking, and the
trajectory-policy view for RL.

**Q: What did you simplify?**  
A: The model is byte-level and small. Sampling uses iterative masked denoising
instead of the full official CTMC sampler. Evaluation uses proxy denoising and
pseudo-perplexity metrics rather than the full likelihood estimator.

**Q: Do you support the official pretrained SEDD checkpoints?**  
A: Yes. The default user-facing backend is `official`; it loads
`louaaron/sedd-small` or `louaaron/sedd-medium` through the upstream SEDD repo
and uses its analytic sampler on CUDA. The `mini` backend remains available for
compact pretrain/SFT/RL experiments.

**Q: Can the official model be fine-tuned in this project?**  
A: Yes. `sedd-official-prepare` builds GPT-2-tokenized SFT data,
`sedd-official-sft` applies response-only score-entropy fine-tuning to
`louaaron/sedd-small` or `sedd-medium`, and `sedd-official-rl` performs
trajectory policy-gradient post-training with a frozen official reference model.

**Q: Why not use `sedd-medium` by default?**  
A: It is much larger and less convenient for a 16 GB take-home GPU. I would use
`sedd-small` first for backend validation and LoRA-style fine-tuning, then move
to `sedd-medium` only if memory and time allow.

**Q: Why byte tokenization?**  
A: It removes tokenizer dependencies, handles any UTF-8 text, and keeps the
vocabulary small enough for quick experiments. For production-scale work I would
move to GPT-2 BPE or the official SEDD tokenizer setup.

**Q: How does supervised fine-tuning work here?**  
A: I format examples as `User: ... Assistant: ...`, keep prompt tokens visible,
and apply diffusion corruption/loss only to assistant response tokens.

**Q: Why keep the same score entropy objective for SFT?**  
A: It preserves the model semantics. SFT changes the data distribution and the
loss mask, not the meaning of the score network.

**Q: How would you evaluate a real SEDD model?**  
A: Validation score entropy, generative perplexity under a strong external LM,
conditional generation metrics, infilling quality, sample diversity, latency
versus number of function evaluations, and likelihood-bound estimates if needed.

**Q: Why include pseudo-perplexity?**  
A: It is cheap and useful for debugging denoising quality, but it is not a claim
of exact likelihood.

## RL

**Q: Why is RL harder for diffusion LMs than autoregressive LMs?**  
A: The final text is produced through many denoising transitions, not a simple
left-to-right chain. Credit assignment and log-probability accounting must happen
over the reverse diffusion trajectory.

**Q: What is the policy in your RL implementation?**  
A: The policy is the model-induced distribution over token choices at masked
positions during denoising.

**Q: What is the action?**  
A: Filling a selected masked position with a sampled token at a denoising step.

**Q: What is the state?**  
A: The partially denoised sequence plus the current noise/time level.

**Q: How do you prevent reward hacking?**  
A: Add a frozen reference-model KL term and keep rewards simple during smoke
tests. A serious run should use a calibrated reward model or verifier plus
stronger KL scheduling.

**Q: How is this related to SEPO?**  
A: SEPO frames policy gradients for discrete diffusion. This repo implements a
small SEPO-inspired trajectory policy-gradient baseline suitable for a runnable
take-home demo.

**Q: What would you do for a stronger RL version?**  
A: Sample multiple completions per prompt, normalize rewards group-wise, add
PPO-style clipping on denoising log-ratios, tune KL by target divergence, and
use task verifiers or preference rewards.

## Research Judgment

**Q: What is the strongest argument for SEDD?**  
A: It attacks a structural weakness of autoregressive modeling: generation need
not be strictly left-to-right, and bidirectional denoising naturally supports
infilling and controllable generation.

**Q: What is the strongest concern?**  
A: Conditional generation from short prompts and exact likelihood evaluation are
less straightforward than in autoregressive models, and high-quality sampling
depends on reverse-process design.

**Q: Why might SEDD be important for AGI-oriented model architecture research?**  
A: It explores a different factorization of sequence modeling. If bidirectional
generation can scale, it may unlock more flexible planning, editing, and
iterative reasoning behaviors than strict token-by-token decoding.

**Q: What would you present live?**  
A: Show the score entropy equation, run the API demo, display a tiny train/SFT/RL
log, explain which parts are faithful versus simplified, then outline the path to
official-checkpoint fine-tuning.
