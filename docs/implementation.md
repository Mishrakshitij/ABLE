# Model and optimization details

ABLE uses a causal language model conditioned on dialogue history and the user's
gender, age group, and OCEAN persona. Supervised fine-tuning produces the PDSS
checkpoint. PPO then optimizes responses using four task classifiers, a fluency
score, and contextual similarity. The reference is the [EMNLP 2024 paper,
Sections 4–5 and Appendix A.2](https://aclanthology.org/2024.emnlp-main.1252.pdf).
This repository implements that method from the published description; it does
not contain the authors' original training checkpoints or claim reproduced
benchmark scores.

## Supervised model and classifiers

The default generator is `microsoft/phi-2` with LoRA. Each Doctor response is one
training target, with all preceding dialogue available as context. The prompt
includes gender, age, and the five OCEAN trait levels. Future turns and the target
response are excluded from its context. Context and padding labels are `-100`;
cross entropy shifts the target once and averages only response tokens, including
EOS. Long inputs retain the profile prefix and the most recent context.

Four independent `roberta-large` classifiers operate on response text. Their
output sizes are 19 persona classes, 6 gender–age classes, 3 politeness classes,
and 3 empathy classes. Metadata is the target for persona and gender–age
classification, never part of the classifier input. All reward classifiers are
frozen during PPO. Saved classifier task metadata detects accidentally swapped
checkpoints.

LoRA rank 16, alpha 16, dropout 0.05, and the `q_proj`, `k_proj`, `v_proj`, and
`dense` target modules are implementation defaults. The paper does not specify
these LoRA settings. The classifier and SFT configurations likewise expose the
optimizer and context limits so experiments can record their choices.

## Reward functions

For the reference response `y`, generated response `g`, target class `k`, and
classifier `C_j`, the `paper` mode implements the printed equations:

```text
R_j = C_j(y)[k] - alpha * C_j(g)[k],  j in {persona, gender_age, politeness, empathy}
R_5 = tanh(NLL_reference(g | context, profile))
R_6 = beta * BERTScore_F1(g, y) + gamma * BERTScore_F1(g, context)
R   = sum_j weight_j * R_j
```

`alpha` must be between 1 and 2. Coherence coefficients are nonnegative and sum
to one. The six nonnegative reward weights sum to one; all-zero weights enable
the ABLE-R ablation. The default coefficients are alpha 1, equal coherence
weights, and equal weights across all six rewards. The paper provides ranges
and normalization constraints, but does not report the chosen values.

There is a sign ambiguity in the printed method: maximizing its first four
rewards lowers the probability of the target class, and maximizing its fifth
reward increases language-model loss. `configs/ppo.json` preserves those printed
signs. `configs/ppo-aligned.json` explicitly selects `aligned`, which negates
R1–R5 and leaves R6 unchanged. The two modes are recorded in checkpoint configs
and must be compared as separate experiments.

The paper's `Loss(y, g)` in the naturalness term does not define token alignment
or a scoring procedure for unequal-length texts. Here it is the mean NLL of the
sampled response tokens under the frozen SFT model, conditioned on the same
prompt. This gives a fixed fluency evaluator throughout optimization. EOS is
scored when sampled; no synthetic EOS is appended to a truncated rollout.

BERTScore uses contextual F1 with `roberta-large`, without baseline rescaling.
It compares the generated response separately with its reference and preceding
dialogue; profile metadata is excluded from the latter. There is no lexical or
heuristic substitute when BERTScore or classifier checkpoints are unavailable.

For a local BERTScore checkpoint directory or a custom model name, set
`rewards.bertscore_model` to that path/name and set
`rewards.bertscore_num_layers` to the representation layer to use. The value
must fit the model's available layers. The standard `roberta-large` setting
uses bert-score's published default layer selection when this option is omitted.
The BERTScore encoder is frozen, and its scoring batch size follows
`rewards.batch_size`. An offline integration test exercises the actual
bert-score package with a one-layer local RoBERTa model.

## PPO and the value function

Each sampled token is an action. A linear value head predicts a scalar from the
hidden state immediately preceding that action. The response score is placed
on the last response token. By default, discounted Monte Carlo returns with
discount 0.95 give advantages `return - old_value`, following the paper's
baseline formulation. GAE is optional through `gae_lambda`; `null` disables it.

The actor minimizes the negative clipped PPO surrogate:

```text
ratio       = exp(current_log_probability - behavior_log_probability)
policy_loss = -mean(min(ratio * advantage,
                       clip(ratio, 1-epsilon, 1+epsilon) * advantage))
```

The critic minimizes half the larger of unclipped and clipped squared return
errors. Optional entropy regularization and a sampled KL penalty against the
frozen SFT checkpoint are available. The `paper` configuration sets both
coefficients to zero, disables advantage normalization, and uses no GAE.
Critic clipping, value coefficient 0.5, and four PPO epochs per rollout are
implementation choices because the paper does not define a complete critic
training algorithm.

Response and EOS masks exclude all prompt tokens and padding from policy loss,
critic loss, entropy, KL, and reward propagation. Behavior log probabilities,
returns, advantages, and reward models are detached. Generation samples from
the full categorical distribution at temperature 1, with no top-k or top-p
filtering, so optimization uses the same action distribution as collection.
Dropout is disabled during both rollout and PPO updates; gradients remain
enabled for the trainable actor parameters and critic. Left-padded inputs use
explicit position IDs when recomputing token probabilities.

Optional PPO `training.gradient_checkpointing` supports the Phi, GPT-2,
Llama, and Mistral architecture families. Only the modules that implement
checkpoint gates enter training mode; their attention, MLP, and dropout
children remain in evaluation mode. This activates gradient checkpointing
without changing functional attention dropout or LoRA dropout during PPO.
Other architectures reject this optional setting rather than silently change
the sampled policy's probabilities. Checkpointing is disabled by default.

Rollouts are collected in `training.batch_size` chunks, stored on CPU, and
optimized in minibatches. `ppo.rollout_batch_size` is the number of responses
per update. The paper's 640 steps per update is interpreted as 640 collected
responses. `training.max_steps` counts rollout updates, and `training.epochs`
limits complete dataset passes. These units are explicit because the appendix
does not distinguish token steps, response steps, and optimizer steps.

The published batch size 8, random seed 10, response cap 50, clip range 0.2,
discount 0.95, AdamW learning rate 1e-5, 32,000-step limit, and 20-epoch limit
are represented in the configuration. Appendix A.2 lists `human_reward=10`
without defining where it enters the objective; this implementation does not
invent a seventh reward or add an unexplained constant.

## Checkpoints and reproducibility

PPO starts from an existing SFT checkpoint and retains a separately frozen copy
of that checkpoint. It saves the policy or LoRA adapter, tokenizer, critic head,
optimizer state, RNG state, epoch, rollout position, and full configuration.
Set `ppo.resume_from` to a saved directory to resume at a rollout boundary.
Resumption requires the same dataset, reward and PPO configuration, training
seed, and optimizer/minibatch settings. The output directory, checkpoint
frequency, epoch limit, and step limit may change. Keep the SFT and classifier
checkpoint contents fixed for the whole experiment.

## Automatic evaluation

The paper's PCA, GAA, PA, and EA compare the classifier's predicted class for
the reference with its predicted class for the generation. These are reported
as `classifier_agreement`. Accuracy against the dataset target is separately
reported as `gold_label_accuracy`.

`mean_response_perplexity` is the mean of individual response perplexities,
following Eq. 17. `token_perplexity` is the exponentiated mean NLL weighted by
token count. Generated responses are scored by the requested generator
checkpoint, using the exact generated token IDs and the exact prompt supplied
to generation. Scoring includes sampled EOS and excludes subsequent padding;
length-capped responses do not receive an invented EOS. Text is never decoded
and re-encoded to compute likelihoods. `response_length_tokens` is mean
non-special generated token count; despite the paper's “ratio” name, Eq. 18
contains no length denominator.

`nrep_bertscore` averages similarity to the preceding two generated Doctor
responses within each conversation, following Eq. 19. Lower values indicate
less repetition. The first two responses in each conversation are excluded
because both predecessors are unavailable. Human evaluation scores are not
automatically inferred or supplied as model results.

## Validation

The test suite checks supervised masking, both signs of clipped PPO, critic
clipping, terminal rewards, discounted returns, first-EOS masking, both reward
modes, and real actor/critic updates. A small randomly initialized GPT-2 model
and a locally constructed tokenizer exercise actual rollout generation and
verify that recomputed action probabilities match the behavior policy. These
tests require no pretrained checkpoint downloads and validate implementation
behavior, not the published benchmark results.

The checkpoint integration tests additionally train a tiny LoRA SFT model and
all four tiny reward classifiers, then execute actual PPO training. Only the
external BERTScore dependency is replaced in that test. Two uninterrupted
rollouts and an interrupted/resumed run must produce exactly equal policy,
critic, optimizer, RNG state, and final metrics. These tests also check that
reference/classifier weights remain frozen and that resuming with changed
dataset, reward, seed, or rollout settings fails explicitly.
