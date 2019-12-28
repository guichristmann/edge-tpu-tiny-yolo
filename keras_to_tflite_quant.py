"""Keras to TF-Lite converter

This script is used to convert the Keras model of Tiny YOLO-V3 to fully
quantized Tensorflow Lite model. This is necessary, since the Edge TPU USB
accelerator only supports integer operations.

Note: The Keras model contains a `RESIZE_NEAREST_NEIGHBOR` operation. The
quantization of this operation is only supported on Tensorflow 2.0 nightly
packages.

Command Line Args:
    keras-model: Path to Keras model to be converted and quantized.
    output-filename: Output filename of the TF-Lite quantized model.

"""

import tensorflow as tf
import numpy as np
import sys

def representative_dataset_gen():
    for _ in range(250):
        yield [np.random.uniform(0.0, 1.0, size=(1, 416, 416, 3)).astype(np.float32)]

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <keras-model> <output-filename>")
    sys.exit(1)

model_fn = sys.argv[1]
out_fn = sys.argv[2]

# Convert and apply full integer quantization
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_fn)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set inputs and outputs of network to 8-bit unsigned integer
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()

open(sys.argv[2], "wb").write(tflite_model)
