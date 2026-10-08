# kernel-error-propagation

Measuring and visualizing the propagation of numerical kernel errors through the layers of a Transformer model.

Seminar paper: *Kernel Error Propagation Through the Layers of a Transformer Model: Visualization and Analysis of Numerical Deviations*

---

## About

In practice, large language models are executed through optimized kernels that change the numerics: weight quantization, lower precision, operation fusion. Each such deviation is small on its own, but the result passes through dozens of layers connected by residual connections.

This project measures **what happens to that error along the way**: whether it dies out, persists, or is amplified, and whether the attention and FFN subsystems behave differently.

### Research questions

| | Question |
|---|---|
| **RQ1** | How does the relative error of the residual stream spread across layers when only attention is perturbed, versus when only the FFN is perturbed? |
| **RQ2** | Does an error introduced at layer *k* grow, shrink, or stabilize in subsequent layers, and does this depend on *k*? |
| **RQ3** | How much of the error in the hidden states translates into error at the model's output? |

Central hypothesis: attention is the only place where information is mixed **between** tokens, while the FFN works on each token independently. An error in attention can therefore corrupt the context in a way that an error in the FFN cannot.

---

## Method in three sentences

The model is run twice on the same input, once as a reference and once with an injected perturbation. Forward hooks record three tensors at every layer: the residual stream, the attention block output, and the FFN block output. The difference between the two passes is measured per layer, per token, and per dimension.

### Measurement points

```
```

### Perturbation types

| Type | Description | Status |
|---|---|---|
| **A** | Weight quantization (int8 / int4, group size 128, symmetric) | planned |
| **B** | Reduced precision (fp16 / bf16) on a chosen subsystem | planned |
| **C** | Deliberate bugs (wrong scaling, noise in the K projection, corrupted mask) | planned |

> **Note on terminology:** this project **does not modify real CUDA/Triton kernels.** The numerical deviations that arise in such kernels are modeled at the level of PyTorch operations. This reproduces the *numerical* consequence of a modified kernel, but not its performance characteristics. A real kernel is planned as an extension once a GPU is available.

### Metrics

---

## Installation

```bash
```

The CPU version of PyTorch is sufficient. To install `torch` on CPU without CUDA packages:

```bash
```

---

## Running

---

## Experiments

| Exp. | Perturbation | Scope | RQ |
|---|---|---|---|
| E1 | attention, int8 / int4 | all layers | RQ1, RQ3 |
| E2 | FFN, int8 / int4 | all layers | RQ1, RQ3 |
| E3 | attention, int4 | layer *k* onward | RQ2 |
| E4 | FFN, int4 | layer *k* onward | RQ2 |
| E5 | fp16 / bf16 | all layers | RQ1 |
| E6 | bug: scaling of the attention output | one layer | RQ2 |
| E7 | bug: noise in the K projection / corrupted mask | one layer | RQ1 |

E3 and E4 yield an *injection layer × measurement layer* matrix, which is the central result of the project.

---

## Repository structure

```
.
├── src/
├── docs/
├── requirements.txt
├── LICENSE              # GPL-3.0 full text
└── README.md
```

---

## Reproducibility

---

## Limitations

- Fake quantization simulates the numerics of a real int kernel, not its speed. There are no performance measurements.
- A small number of short sequences due to CPU time constraints.
- A single model from a single family, with no generalization to other architectures.
- If the reference is `bf16`, it carries its own rounding error, which suppresses the effect of finer perturbations (int8, fp16). Use `fp32` for a clean reference.
- The weights of the linear projections are perturbed, not the attention/norm operations themselves (except in E6/E7).

---

## Status

| Component | State |

---

## References

> Every reference is checked against its source before citing: authors, year, title, and venue.

---

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) (GPL-3.0). See the [LICENSE](LICENSE) file for the full text.
