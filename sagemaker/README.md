## SageMaker Ray Examples

This directory contains 4 examples of runnin Ray on SageMaker
- [hello_ray](/hello_ray) runs a basic hello world example where after a Ray cluster is created, a remote function is executed to return the IP address of each worker node.
- [distributed_xgboost](/distributed_xgboost) demonstrates how Ray can accelerate training of XGBoost models on large scale data through distributed CPU or GPU training
- [pytorch_lightning](/pytorch_lightning) leverages the RayLightning plugin to accelerate training and tuning of Deep Learning models on PyTorch Lightning
- [jax_alpa_language_model](/jax_alpa_language_model) shows how Ray with [alpa](https://github.com/alpa-projects/alpa) can accelerate training of large scale language models