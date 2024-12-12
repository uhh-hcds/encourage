# 📝 Mlflow

## Use Mlflow to track experiments and manage models

Mlflow is a platform for managing the end-to-end machine learning lifecycle. It provides tools to track experiments, log parameters, and metrics. Mlflow is designed to ensure observability of the model performance and tracing.
Therefore the encourage package provides a simple way to integrate Mlflow into your project.

### Enable Mlflow

To enable Mlflow you have to call the `enable_mlflow_tracing` function. This function will enable the Mlflow integration and start tracking the experiments.

```python
from encourage import enable_mlflow_tracing
enable_mlflow_tracing()
```

The `enable_mlflow_tracing` function will automatically log the parameters and metrics of the experiment and uri you have set with the following functions:

```python
import mlflow

mlflow.set_tracking_uri(<URI>)
mlflow.set_experiment(experiment_name=<EXPERIMENT_NAME>)
```