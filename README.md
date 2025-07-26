### `dev` Branch: Model Training

The `main` branch serves as the foundation for the project. It contains the initial `train.py` script responsible for:
* Loading the California Housing dataset.
* Splitting data into training and testing sets.
* Training a Scikit-learn Linear Regression model.
* Evaluating the model using the $R^2$ score.
* Saving the trained model to `linear_regression_model.joblib`.

**Original Sklearn Model Performance:**
* **Test R^2 score:** 0.5758

### `dev` Branch: Feature Development

The `dev` branch is used for active development and new feature implementation. Changes are developed here and then merged into `main` after review. For this assignment, `dev` was used to integrate the `requirements.txt` changes for PyTorch libraries.
