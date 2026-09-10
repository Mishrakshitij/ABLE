# Published results

The values in [`paper-results.csv`](paper-results.csv) are transcribed from
Table 2 of the [EMNLP 2024 paper](https://aclanthology.org/2024.emnlp-main.1252.pdf),
page 22452. They are manuscript results, not measurements from the maintained
implementation. Run the generation and evaluation commands in the repository
README to obtain measurements for your trained checkpoints.

PCA, GAA, PA, and EA are percentages for persona, gender–age, politeness, and
empathy agreement. PPL is perplexity. Rlen is the average generated response
length in tokens, despite the word “ratio” in its name. Nrep measures similarity
to previous generated responses; lower is better. See Eqs. 13–19.

| Model | PCA ↑ | GAA ↑ | PA ↑ | EA ↑ | PPL ↓ | Rlen | Nrep ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT2-large | 50.3 | 60.1 | 72.8 | 70.2 | 14.93 | 11.19 | 0.39 |
| ARDM | 55.2 | 67.9 | 77.6 | 75.6 | 11.14 | 13.49 | 0.31 |
| Llama2-7B | 54.7 | 67.2 | 78.6 | 77.1 | 7.01 | 16.94 | 0.22 |
| Mistral-7B | 55.4 | 68.3 | 79.2 | 78.4 | 6.85 | 17.10 | 0.21 |
| Zephyr-7B | 56.3 | 69.6 | 80.7 | 78.9 | 6.59 | 17.23 | 0.21 |
| Phi-1.5 | 56.8 | 70.1 | 80.5 | 78.7 | 6.67 | 17.15 | 0.20 |
| PDSS | 58.0 | 71.0 | 83.7 | 81.2 | 5.01 | 18.31 | 0.15 |
| ABLE-R | 57.9 | 71.3 | 83.5 | 81.6 | 5.08 | 18.12 | 0.14 |
| ABLE-TR | 58.4 | 71.9 | 85.4 | 83.0 | 4.94 | 18.28 | 0.11 |
| ABLE-GR | 60.7 | 73.1 | 86.7 | 84.2 | 4.86 | 18.35 | 0.10 |
| **ABLE** | **61.5** | **74.0** | **87.6** | **85.8** | **4.30** | **19.95** | **0.07** |

PDSS is the supervised Phi-2 model. ABLE-R removes all six task rewards;
ABLE-TR uses naturalness and coherence; ABLE-GR uses the four classifier rewards.
The [implementation notes](implementation.md) explain how to configure these
ablations and interpret the reward equations.
