# Validation

The release checks exercise the data readers, numerical objectives, and complete
training paths independently of pretrained model downloads.

- Every distributed CSV is checked for its hash, schema, row coverage, label
  range, profile mapping, conversation order, and split membership. Complete
  duplicate dialogues cannot appear in different splits.
- Response masking tests exclude prompts and padding while retaining an actual
  EOS. Numerical tests cover clipped policy and value losses, discounted
  returns, both reward conventions, and invalid coefficients.
- Offline integration tests train a tiny LoRA causal model and four tiny
  RoBERTa classifiers, save and reload them, generate predictions, and evaluate
  classifier agreement and likelihoods.
- PPO integration tests perform real actor and critic updates with trained tiny
  classifiers. Only the external BERTScore model is replaced in these tests.
  Separate integration tests exercise real BERTScore with a local encoder.
  PPO tests verify frozen reference models, unchanged behavior statistics, and exact
  equivalence between resumed and uninterrupted runs, including optimizer and
  RNG states.
- Gradient-checkpointing tests execute real Phi and GPT-2 backward passes while
  checking that attention and LoRA dropout remain disabled during PPO updates.

A separate smoke run also used a tiny Phi-2 architecture with the configured
`q_proj`, `k_proj`, `v_proj`, and `dense` LoRA targets. Supervised training,
checkpoint reload, generation, and PPO completed with finite losses; all LoRA
tensors and the critic updated while the reference remained unchanged.

The local integration environment uses Python 3.11, PyTorch 2.13.0,
Transformers 5.16.1, and PEFT 0.20.0. CPU checks do not establish full-scale
Phi-2 or RoBERTa training quality. Published scores are identified separately in
[`paper-results.md`](paper-results.md); no benchmark reproduction is claimed.

Run:

```bash
python -m pip install -e ".[train,eval,test]"
python -m pytest tests -q
python scripts/verify_dataset.py
```

The package also builds as a wheel with `python -m pip wheel --no-deps .`.
GitHub Actions defines separate full-dataset and offline-model test jobs.
