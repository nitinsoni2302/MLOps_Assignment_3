# MLOps Assignment 3: End-to-End MLOps Pipeline

## 1. Overview & Learning Objectives

This project demonstrates the creation of a complete, automated MLOps pipeline. It covers:
* Model development with Scikit-learn and PyTorch.
* Containerization of the model using Docker.
* Building a robust CI/CD workflow with GitHub Actions to automate training, testing, and deployment (pushing to Docker Hub).
* Model optimization through manual 8-bit unsigned quantization.

The goal is to build an end-to-end pipeline that ensures consistent model performance, automated deployments, and explores model optimization techniques for efficiency.

## 2. Public Links

* **GitHub Repository:** [https://github.com/nitinsoni2302/MLOps_Assignment_3](https://github.com/nitinsoni2302/MLOps_Assignment_3)
* **Docker Hub Repository:** [https://hub.docker.com/repository/docker/nitinsoni2302/mlops_housing_model]

## 3. Dataset & Model

* **Dataset:** California Housing dataset from `sklearn.datasets`.
* **Model:**
    * Initially, a Linear Regression model is trained using `scikit-learn`.
    * Later, a single-layer PyTorch neural network is created, and its weights are manually set using the parameters from the trained scikit-learn model for comparison and quantization.

## 4. Pipeline Branches & Implementation Details

### 4.1. `dev` Branch: Model Training

The `main` branch serves as the foundation for the project. It contains the initial `train.py` script responsible for:
* Loading the California Housing dataset.
* Splitting data into training and testing sets.
* Training a Scikit-learn Linear Regression model.
* Evaluating the model using the $R^2$ score.
* Saving the trained model to `linear_regression_model.joblib`.

**Original Sklearn Model Performance:**
* **Test R^2 score:** 0.5758

### 4.2. `dev` Branch: Feature Development

The `dev` branch is used for active development and new feature implementation. Changes are developed here and then merged into `main` after review. For this assignment, `dev` was used to integrate the `requirements.txt` changes for PyTorch libraries.

### 4.3. `docker_ci` Branch: CI/CD Pipeline with Docker & GitHub Actions

The `docker_ci` branch implements the continuous integration and continuous deployment pipeline.
* **`Dockerfile`**: Defines the environment for containerizing the model, ensuring consistent execution.
* **`.github/workflows/ci.yml`**: Configures a GitHub Actions workflow that:
    * Automates building the Docker image.
    * Runs tests to verify the image (by executing `predict.py` within the container).
    * Pushes the built image to Docker Hub upon successful completion.

### 4.4. `quantization` Branch: Model Conversion & Optimization

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

*Observation: Manual 8-bit quantization resulted in a reduction of the model size from 0.40 KB to 0.39 KB, with a minor reduction in the $R^2$ score from 0.5758 to approximately 0.5700, demonstrating a successful trade-off between size and performance.*

---
