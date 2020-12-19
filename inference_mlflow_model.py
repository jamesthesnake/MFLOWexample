import argparse
import numpy as np
import mlflow.tensorflow
import tensorflow as tf
import dataset

def main(model_uri, data_path):

    tf_graph = tf.Graph()
    with tf.Session(graph=tf_graph) as sess:
        with tf_graph.as_default():

            # Use MNIST Dataset we have used to training
            ds = dataset.train(data_path)
            next_op = tf.data.make_one_shot_iterator(ds).get_next()

            # Load the MLflow model
            signature_def = mlflow.tensorflow.load_model(model_uri=model_uri, tf_sess=sess)
            input_tensors = {input_signature.name: tf_graph.get_tensor_by_name(input_signature.name) 
                             for _, input_signature in signature_def.inputs.items()}
            output_tensors = {output_signature.name: tf_graph.get_tensor_by_name(output_signature.name)
                              for _, output_signature in signature_def.outputs.items()}

            for _ in range(10):
                # This uses a 2-step process:
                #  1. Run the `next_op` to fetch the next image from the dataset
                #  2. Use a feed dictionary to run the prediction
                # This is for purpose of demonstration only and should never be 
                # used in a real system because this is very inefficient.
                image, label = sess.run(next_op)
                feed_dict = { input_tensors['images:0']: np.expand_dims(image, axis=0) }
                pred = sess.run(output_tensors['ArgMax:0'], feed_dict=feed_dict)[0]
                correct = 'SAME' if label==pred else 'NOT_SAME'
                print(label, pred, correct)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mlflow_model_uri', help='MLflow model URI')
    parser.add_argument('--data-path', help='Data directory', default='/tmp/mnist')
    args = parser.parse_args()
    print(args)
    main(model_uri=args.mlflow_model_uri, data_path=args.data_path)
