# PERPDSCD

PERPDSCD contains conversations about physical disability support between a Patient and a Doctor, with gender, age, and OCEAN personality profiles. It covers 13 support topics and 19 persona classes.

The CSV files are in [`perpdscd/`](perpdscd/). Read all shards for a split in filename order; each conversation stays within one shard. Files are UTF-8 with a header row and standard CSV quoting.

| Split | Conversations | Utterances | Doctor responses | Min. turns | Mean turns | Max. turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 14,420 | 322,427 | 160,666 | 10 | 22.36 | 32 |
| Validation | 1,803 | 40,350 | 20,103 | 10 | 22.38 | 30 |
| Test | 1,803 | 40,308 | 20,073 | 10 | 22.36 | 30 |
| Total | 18,026 | 403,085 | 200,842 | 10 | 22.36 | 32 |

The release uses a deterministic 80/10/10 conversation split with seed 10. Conversations with identical normalized complete dialogue content stay in the same split. [`splits.json`](perpdscd/splits.json) lists the conversation IDs in each split. [`manifest.json`](perpdscd/manifest.json) records file hashes and split statistics.

| Column | Meaning |
| --- | --- |
| `Convo_id` | Conversation identifier |
| `Turn_id` | Original turn identifier |
| `Speaker` | `Patient` or `Doctor` |
| `Utterance` | Utterance text |
| `Gender` | `Male` or `Female` |
| `Age` | `Younger`, `Middle Aged`, or `Older` |
| `Persona` | Five OCEAN trait intensities |
| `Issue` | Support topic |
| `Physical disability` | Disability or support condition |
| `row_id` | Unique zero-based utterance identifier |
| `turn_index` | Consecutive, one-based position within the conversation |
| `split` | `train`, `validation`, or `test` |
| `persona_label` | Class `0`–`18` in Appendix A.1.2 order |
| `gender_age_label` | Class `0`–`5`: male younger/middle/older, then female younger/middle/older |
| `politeness_label` | `0` impolite, `1` neutral, `2` polite |
| `empathy_label` | `0` non-empathetic, `1` neutral, `2` empathetic |

[`labels.json`](perpdscd/labels.json) contains the complete class mappings. Use `Convo_id` with `turn_index` to address a turn uniquely. The training reader creates one example per Doctor response, using only earlier conversation turns as context.

Style-label counts across all utterances:

| Attribute | Class | Train | Validation | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Politeness | Impolite | 1 | 1 | 0 | 2 |
| Politeness | Neutral | 128,121 | 16,098 | 15,950 | 160,169 |
| Politeness | Polite | 194,305 | 24,251 | 24,358 | 242,914 |
| Empathy | Non-empathetic | 1 | 0 | 0 | 1 |
| Empathy | Neutral | 234,706 | 29,396 | 29,323 | 293,425 |
| Empathy | Empathetic | 87,720 | 10,954 | 10,985 | 109,659 |

Style-label counts for Doctor responses, which are the generation training targets:

| Attribute | Class | Train | Validation | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Politeness | Impolite | 1 | 0 | 0 | 1 |
| Politeness | Neutral | 52,788 | 6,595 | 6,581 | 65,964 |
| Politeness | Polite | 107,877 | 13,508 | 13,492 | 134,877 |
| Empathy | Non-empathetic | 1 | 0 | 0 | 1 |
| Empathy | Neutral | 72,946 | 9,149 | 9,089 | 91,184 |
| Empathy | Empathetic | 87,719 | 10,954 | 10,984 | 109,657 |

From the repository root, verify the files with:

```bash
python scripts/verify_dataset.py --dataset-dir datasets/perpdscd
```
