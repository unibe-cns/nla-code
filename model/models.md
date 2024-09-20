# Model implementations

We implemented the model in several forms and frameworks. To be understood by a larger audience, we publish them all in this repository.

## For easier understanding of rtDeep and rtDeel
All model implementations below do not implement batching. For batching, the algebraic operations get significatly harder to read and hinder understanding.
- lagrange_model_rtdeep_np.py (rtDeep implementation in numpy, no batching)
- lagrange_model_rtdeel_np.py (rtDeel implementation in numpy, no batching)
- lagrange_model_rtdeep_torch.py (rtDeep implementation in torch, no batching)

## Used by the experiments
- lagrange_model_tf.py (rtDeep & rtDeel implementation in tensorflow 1.0, no batching)
- lagrange_model_torch (rtDeep & rtDeel implementation in torch, no batching)
- lagrange_model_torch_batched (rtDeep & rtDeel implementation in torch, with batching)
- lagrange_model.py (Wrapper across models to switch between above described implementations and backends)