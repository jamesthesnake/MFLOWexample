"""A MNIST classifier using the TensorFlow Estimator API. Parameters, evaluation 
results and model files are logged using MLFLow."""

import os
import argparse
import datetime
import tempfile
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.saved_model import tag_constants
from tensorflow.examples.tutorials.mnist import input_data
import dataset

tf.logging.set_verbosity(tf.logging.INFO)

def current_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def build_model(x, hidden_size, keep_prob):
    """Build the model."""

    with tf.variable_scope("model"):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='images')
        conv1 = layers.convolution2d(x_image, num_outputs=32, kernel_size=5,
                                     stride=1, padding='SAME', activation_fn=tf.nn.relu,
                                     scope='conv1')
        pool1 = layers.max_pool2d(inputs=conv1, kernel_size=2, stride=2,
                                  padding='SAME', scope='pool1')
        conv2 = layers.convolution2d(pool1, num_outputs=64, kernel_size=5, stride=1,
                                     padding='SAME', activation_fn=tf.nn.relu, scope='conv2')
        pool2 = layers.max_pool2d(inputs=conv2, kernel_size=2, stride=2,
                                  padding='SAME', scope='pool2')
        flattened = layers.flatten(pool2)
        fc1 = layers.fully_connected(flattened, hidden_size, activation_fn=tf.nn.relu, 
                                     scope='fc1')
        drop1 = layers.dropout(fc1, keep_prob=keep_prob, scope='drop1')
        logits = layers.fully_connected(drop1, 10, activation_fn=None,  scope='logits')
        return logits

def model_fn(features, labels, mode, params, config):
    """Model function that returns EstimatorSpecs."""

    if mode == tf.estimator.ModeKeys.PREDICT:
        images = features['images']
        logits = build_model(images, params['hidden_size'], 1.0)
        predictions = { 
            "class": tf.argmax(logits, axis=1, output_type=tf.int32),
            'probabilities': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = build_model(features, params['hidden_size'], params['keep_rate'])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]) 
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step()) 
        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = build_model(features, params['hidden_size'], 1.0)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
        eval_metric_ops = { "accuracy": (acc, acc_op) }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=eval_metric_ops)


def main(learning_rate, batch_size, checkpoint_base_path, data_path, tracking_url):

    checkpoint_path = os.path.join(checkpoint_base_path, current_timestamp())
    os.makedirs(checkpoint_path, exist_ok=True)

    params = {
        'hidden_size': 512, 
        'keep_rate': 0.8, 
        'learning_rate': learning_rate, 
        'nb_epochs': 1, 
        'batch_size': batch_size,
        'checkpoint_path': checkpoint_path
    }


    # Configure the location where tracking data will be written to. In real-life
    # this would be a remote MLFlow Tracking Servinc (using HTTP) or something like
    # S3, HDFS etc.
    mlflow.set_tracking_uri(tracking_url)

    # Set name of experiment
    mlflow.set_experiment('MNIST_TF_Estimator')

    with mlflow.start_run() as run:

        # Log parameters in MLFlow
        for name, value in params.items():
            mlflow.log_param(name, value)

        def train_input_fn():
            ds = dataset.train(data_path)
            ds = ds.shuffle(buffer_size=50000)
            ds = ds.take(5000) # just to speed up training
            ds = ds.batch(params['batch_size'])
            ds = ds.repeat(params['nb_epochs'])
            return ds      

        def eval_input_fn():
            ds = dataset.test(data_path)
            ds = ds.batch(params['batch_size'])
            return ds
            
        run_config = tf.estimator.RunConfig(log_step_count_steps=50)
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=checkpoint_path, 
                                           params=params, config=run_config)

        estimator.train(input_fn=train_input_fn)
        eval_results = estimator.evaluate(input_fn=eval_input_fn)
        tf.logging.info('Eval loss: %s' % eval_results['loss'])
        tf.logging.info('Eval accuracy: %s' % eval_results['accuracy'])

        # Log results in MLFlow
        mlflow.log_metric("eval_loss", eval_results['loss'])
        mlflow.log_metric("eval_acc", eval_results['accuracy'])

        # Send checkpoint and event files to MLFlow
        mlflow.log_artifacts(checkpoint_path)

        # Export the latest checkpoint as SavedModel
        feat_spec = {"images": tf.placeholder("float", name="images", shape=[None, 784])}
        receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_spec)
        export_dir_base = os.path.join(checkpoint_path, 'saved_models')
        saved_estimator_path = estimator.export_saved_model(export_dir_base, receiver_fn).decode("utf-8")

        tf.logging.info('SavedModel has been exported to %s' % saved_estimator_path)

        # Log the SavedModel as MLFlow model
        mlflow.tensorflow.log_model(tf_saved_model_dir=saved_estimator_path,
                                    tf_meta_graph_tags=[tag_constants.SERVING],
                                    tf_signature_def_key="serving_default",
                                    artifact_path="exported_model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', help='Learning rate', type=float, default=0.0001)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=128)
    parser.add_argument('--data-path', help='Data directory', default='/tmp/mnist')
    parser.add_argument('--checkpoint-path', help='Checkpoint directory', default='/tmp/models')
    parser.add_argument('--tracking-url', help='MLFlow tracking URL', default='file:/tmp/mlruns')

    args = parser.parse_args()
    main(learning_rate=args.learning_rate, batch_size=args.batch_size, 
         data_path=args.data_path, checkpoint_base_path=args.checkpoint_path,
         tracking_url=args.tracking_url)
