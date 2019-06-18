# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import re
import os
from PIL import Image
from pathlib import Path


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Load the model
            load_model('arch/pretrained_model/MobileFaceNet_9925_9680.pb')

            # Get input and output tensors, ignore phase_train_placeholder for it have default value.
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            embedding_size = embeddings.get_shape()[1]

            base_dir = Path('~/datasets/my_align_112').expanduser()

            image_files = list(base_dir.glob('*/*'))

            nrof_images = len(image_files)

            imgs = np.zeros(shape=(nrof_images, 112, 112, 3))

            for i in range(nrof_images):
                imgs[i, ...] = np.asarray(Image.open(image_files[i]))

            # 归一化 * 0.0078125相当于/128
            imgs = (imgs - 127.5) * 0.0078125

            emb = sess.run(embeddings, feed_dict={inputs_placeholder: imgs})

            # print('距离为：',np.sqrt(np.sum((emb[0]-emb[1])**2)))

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, image_files[i]))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    emb1 = emb[i, :]
                    emb2 = emb[j, :]
                    diff = np.subtract(emb1, emb2)
                    dist = np.sum(np.square(diff))
                    # dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    print('  %1.4f  ' % dist, end='')
                print('')


if __name__ == '__main__':
    main()
