###  `docker_ci` Branch: CI/CD Pipeline with Docker & GitHub Actions

The `docker_ci` branch implements the continuous integration and continuous deployment pipeline.
* **`Dockerfile`**: Defines the environment for containerizing the model, ensuring consistent execution.
* **`.github/workflows/ci.yml`**: Configures a GitHub Actions workflow that:
    * Automates building the Docker image.
    * Runs tests to verify the image (by executing `predict.py` within the container).
    * Pushes the built image to Docker Hub upon successful completion.# MLOps_Assignment_3
End-to-End MLOps Pipeline Assignment
