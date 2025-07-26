###  `quantization` Branch: Model Conversion & Optimization

The `quantization` branch focuses on model optimization. The `quantize.py` script performs the following:
* Loads the trained `scikit-learn` model.
* Extracts its coefficients and intercept, saving them as `unquant_params.joblib`.
* Manually performs 8-bit unsigned quantization on these parameters, calculating a unified scale and zero-point.
* Saves the quantized parameters and quantization details to `quant_params.joblib`.
* Creates a PyTorch linear regression model.
* Infers with the *de-quantized* parameters in the PyTorch model to evaluate the impact of quantization.

**Model Quantization Analysis:**

| Metric        | Original Sklearn Model | Quantized Model    |
| :------------ | :--------------------- | :----------------- |
| $R^2$ Score   | 0.5758                 | 0.5699853014070698 |
| Model Size    | 0.40 KB                | 0.39 KB            |

*Observation: Manual 8-bit quantization resulted in a reduction of the model size from 0.40 KB to 0.39 KB, with a minor reduction in the $R^2$ score from 0.5758 to approximately 0.5700, demonstrating a successful trade-off between size and performance.*# MLOps_Assignment_3
End-to-End MLOps Pipeline Assignment
