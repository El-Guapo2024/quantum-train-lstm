# Efficient Neural Architecture Search for Quantum-Train

Reinforcement-learning-based Neural Architecture Search (NAS) over the Mapping Model used in the Quantum-Train framework, applied to fine-tuning GPT-2 on WikiText-2.

The best NAS-discovered RNN architecture **reduced testing perplexity from 98.81 → 17.62** — an 83% reduction (5.6× improvement) over the baseline MLP mapping. At a similar parameter budget, perplexity dropped to **31.73** (3.1× improvement).

This work was completed during a summer 2024 research stay at Foxconn (Hon Hai) in Taipei, in collaboration with MIT, under the mentorship of Chen-Yu Liu.

---

## Background

The Quantum-Train method [(Liu et al., *IEEE QCE 2024*)](#references) trains classical neural networks by mapping the probability distribution of a Quantum Neural Network (QNN) into the weights of a classical model via a learned **Mapping Model**. The choice of Mapping Model architecture directly impacts both the number of trainable parameters and the final downstream performance.

The original Quantum-Train Mapping Model was a fixed MLP. This project asks: **can we automatically discover a better Mapping Model architecture?**

## Approach

1. **Downstream task.** Fine-tune the LM head of GPT-2 (38M params out of 127M total) on WikiText-2. Evaluation metric: perplexity, derived from cross-entropy.
2. **Mapping Model search space:**
   - ≤ 2 hidden layers
   - Layer sizes ∈ `{64, 128, 256}`
   - Activations ∈ `{Tanh, ELU, ReLU, Identity}`
   - Architecture families: MLP and RNN variants
3. **NAS via Reinforcement Learning:**
   - **Controller:** LSTM that samples architectures from the search space.
   - **Policy:** log probability of the sampled architecture.
   - **Reward:** normalized inverse perplexity of the sampled (and trained) Mapping Model.
4. **Parameter sharing** (inspired by Pham et al., *ICML 2018*):
   - If a previously-sampled architecture (or one differing only by activation) is re-sampled, restore its pre-trained weights and fine-tune on 1/5 of an epoch instead of training from scratch.
   - Significantly reduces wall-clock cost of architecture search.

## Results

| Mapping Model              | Architecture                              | Perplexity |
| -------------------------- | ----------------------------------------- | ---------- |
| Baseline (initial MLP)     | `MLP [32, 64, 32]`                        | 98.811     |
| Best NAS-found             | `RNN [(64, identity), (256, tanh)]`       | **17.623** |
| Best at similar param count| `RNN [(256, identity), (64, relu)]`       | 31.734     |

- Best NAS architecture **reduced perplexity to ~17% of the baseline** (5.6× improvement).
- **At a similar parameter budget**, perplexity dropped to ~32% of baseline (3.1× improvement).
- Empirically, **RNN-based mapping models consistently outperformed MLPs** of comparable size on this task.

## Files

- `notebook.ipynb` — full implementation: LSTM controller, NAS loop with parameter sharing, evaluation against the Quantum-Train pipeline applied to GPT-2 / WikiText-2.

## Stack

PyTorch, [TorchQuantum](https://github.com/mit-han-lab/torchquantum), [TorchMPS](https://github.com/jemisjoky/TorchMPS), GPT-2 (HuggingFace `transformers`), HuggingFace `datasets` (WikiText-2).

## Future work

Originally identified follow-ups:

- Try Transformer-based Mapping Models.
- Expand the architecture search space.
- Experiment with alternative reward formulations.
- Apply Quantum-Train recursively to the architecture search itself, to further reduce trainable parameters.

## References

- **Quantum-Train framework:** *Training Classical Neural Networks by Quantum Machine Learning.* IEEE QCE, 2024.
- **Parameter-sharing NAS:** Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., & Dean, J. *Efficient Neural Architecture Search via Parameter Sharing.* ICML, 2018.

## Acknowledgments

Completed during a summer 2024 research stay at Foxconn (Hon Hai Precision Industry Co., Ltd.), Taipei, in collaboration with MIT. Mentor: Chen-Yu Liu.
