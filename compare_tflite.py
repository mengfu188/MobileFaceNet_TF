import numpy as np
import tensorflow as tf
import cv2
import argparse
import fire


def load_interpreter(lite_file):
    interpreter = tf.lite.Interpreter(model_path=lite_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print(output_details)

    return interpreter


def invoke(interpreter, data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(data.shape) == 3:
        data = np.expand_dims(data, 0)
    data = (data - 127.5) / 128
    data = data.astype(np.float32)
    data = data.copy()
    vec1 = interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def _compare(vec1, vec2):
    return np.sum(np.square(vec1 - vec2))


def compare(lite_file, img1, img2):
    interpreter = load_interpreter(lite_file)

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    img1 = cv2.resize(img1, (112, 112))
    img2 = cv2.resize(img2, (112, 112))

    vec1 = invoke(interpreter, img1)
    vec2 = invoke(interpreter, img2)
    r = _compare(vec1, vec2)

    print(r)


if __name__ == '__main__':
    fire.Fire()
