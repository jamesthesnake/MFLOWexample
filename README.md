# README

This is a demonstration of the main features provided by [MLflow](https://MLflow.org/). 

MLflow is an open source library to manage machine learning lifecycles. 

It has currently three components:

* MLflow Tracking: Tracks different aspects of ML experiments like code, data, configuration, and results
* MLflow Models: A packaging format for models that integrates with various deployment tools
* MLflow Projects: A packaging format for reproducible execution of code

All examples in this repo work with TensorFlow, but MLflow supports many other ML frameworks.

## Setup

MLflow is a Python library and can be installed with `pip`.

To install the dependencies for this repository:

    pip install -r requirements.txt


## MLflow Tracking

[MLflow Tracking](https://www.MLflow.org/docs/latest/tracking.html) is an API for logging parameters, code versions, metrics, and output files from
machine learning experiments. It also provides a user interface for visualizing the results. 

Start the MLflow tracking server:

    MLflow server --backend-store-uri /tmp/mlruns --default-artifact-root /tmp/artifacts

`--backend-store-uri` defines the location where experiment data and metrics are stored. This can be a local filesystem or a SQLAlchemy compatible 
database like SQLite. `--default-artifact-root` defines the location where artifacts like model files are stored. This can be the local filesystem or 
a remote storage like S3.

By default the tracking server binds to port 5000 on localhost.

Train a set of models with different parameters:

	python train_estimator.py --learning-rate 0.01 --tracking-url http://127.0.0.1:5000
	python train_estimator.py --learning-rate 0.001 --tracking-url http://127.0.0.1:5000
	python train_estimator.py --learning-rate 0.0001 --tracking-url http://127.0.0.1:5000

An experiment sends parameters, metrics and training results to the tracking service. The last checkpoint is exported as an MLflow Model and pushed 
to the tracking service.

Open a browser at `http://127.0.0.1:5000` and select `MNIST_TF_Estimator` to see the results of the three runs:

![Tracker1](images/tracker1.png?raw=true "Tracker1")

Click one of the runs to show the artifacts that have been recorded:

![Tracker2](images/tracker2.png?raw=true "Tracker2")

There are the usual files created by the TensorFlow Estimator:

 * Checkpoints
 * GraphDef definition
 * Event files for training and evaluation

In folder `exported_model` is also the MLflow Model that has been created by `train_estimator.py`.


## MLflow Model

[MLflow Model](https://www.MLflow.org/docs/latest/models.html) is a standardized packaging format for machine learning models.

Each MLflow Model is a directory containing arbitrary files, together with an *MLmodel* file in the root of the directory that stores meta-data.

An MLflow Model can define multiple [flavors](https://www.mlflow.org/docs/latest/models.html#built-in-model-flavors) that the model can be used with. 
A flavour acts as an adapter between the model and a specific framework or tool.

MLflow has built-in flavors are:

 * Spark
 * PyTorch
 * Keras
 * TensorFlow
 * ONNX
 * etc.

The script `inference_MLflow_model.py` shows an example how to use an MLflow model in TensorFlow.

First we need to download the MLflow model from the Tracking Service:

    MLflow_TRACKING_URI=http://127.0.0.1:5000 mlflow artifacts download --run-id 2eddaed00e264f73b5bd94b057054d7c --artifact-path exported_model

Note that the `run-id` value is a unique ID and must replaced by the actual value from the Tracking Server UI.

The `mlflow artifacts download ` command copies the model to a local directory and returns the path, e.g.:

    /tmp/artifacts/1/c69616b474964e7fa9f6f6919965d7e5/artifacts/exported_model

Now we can use the model for inference:

    python inference_MLflow_model.py file:/tmp/artifacts/1/2eddaed00e264f73b5bd94b057054d7c/artifacts/exported_model


## Using Tensorflow Serving

A popular option to deploy TensorFlow models is to use [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving). 

MLflow Model does not support Tensorflow Serving currently. As a workaround, we can download the SavedModel from the Tracking Service and deploy 
it into Tensorflow Serving. This works because the MLflow model is just the SavedModel with some meta-data.

We use the same `mlflow artifacts download` command again but specify a different artifact path `exported_model/tfmodel`:

    MLflow_TRACKING_URI=http://127.0.0.1:5000 mlflow artifacts download --run-id 2eddaed00e264f73b5bd94b057054d7c --artifact-path exported_model/tfmodel

This downloads the SavedModel to a local directory.

Check out the [Tensorflow Serving documentation](https://www.tensorflow.org/tfx/guide/serving) how to deploy the SavedModel with TensorFlow Serving.


## MLflow Project

[MLflow Project](https://www.MLflow.org/docs/latest/projects.html) defines a format to organize and describe code. It also provides an 
API and command-line tools for running these projects.  

Each MLflow Project has a *MLproject* YAML file that specifies the following properties:

* Project name
* Entry points: Commands that can be run within the project and specifications about there parameters.
* Environment: The software environment that should be used to execute project entry points. This includes all library dependencies.

An MLflow Project can be located on the local filesystem or on Github. The environment can be defined as a [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
or a [Docker container](https://www.docker.com/).

For an example check out the MLflow Project in `sample_project`. It defines two endpoints `main` and `validate` and uses a Conda environment.

Run the `main` endpoint:

    touch /tmp/train_dataset.tgz
    mlflow run sample_project -P data_path=/tmp/train_dataset.tgz

Run the `validate` endpoint:

    touch /tmp/test_dataset.tgz
    mlflow run sample_project -e validate -P data_path=/tmp/train_dataset.tgz

At first glance, this might look like an overcomplicated way to run a script but can actually be quite useful. By packaging some code in an 
MLflow Project other people can run it using a single command without having to worry about setting up environments or library dependencies.  

The `MLflow run` command can reference projects that are hosted on Github. This can be combined with a scheduler like [Apache Airflow](https://airflow.apache.org/) 
to create recurring workflows.
