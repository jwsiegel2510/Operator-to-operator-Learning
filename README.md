# Operator-to-operator Learning

Contains experiments testing and demonstrating the operator-to-operator learning methodology. Currently under construction.

## Generate Darcy training data

From the repository root, run the following on a CUDA host with a compatible
[CuPy installation](https://docs.cupy.dev/en/stable/install.html). For a local
CPU run, omit `--device cuda`.

```bash
python -m examples.data_gen.data_generation_darcy \
  --no-of-operators 2 --no-of-examples 40 --a-grid 65 --f-grid 32 \
  --device cuda --pair-batch-size 4 --chunk-size 2 --output-dir darcy_run
```

Sampling uses NumPy. Each coefficient is prepared once, and `--pair-batch-size`
controls how many forcings are solved together in float64. Reduce it if GPU
memory is limited. Each `.pt` file stores up to `--chunk-size` whole operators
with their matched `a`, `f`, and `u` tensors in float32.

Run `python -m unittest discover -s tests -v` to check the solver. The CUDA
test exercises the solver and chunk writer when CuPy and a CUDA device are available.

