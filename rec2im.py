import mxnet as mx
import mxnet.ndarray as nd
from skimage import io
import numpy as np
from mxnet import recordio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

path_prefix = 'datasets/faces_webface_112x112/train'
output_dir = 'datasets/align_webface_112x112'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


path_imgidx = 'datasets/faces_webface_112x112/train.idx'
path_imgrec = 'datasets/faces_webface_112x112/train.rec'

imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

for i in tqdm(range(501195)):
    header, s = recordio.unpack(imgrec.read_idx(i+1))
    img = mx.image.imdecode(s).asnumpy()
    label =str(header.label)
    id = str(i)

    label_dir = os.path.join(output_dir, label)
    # 检查标签文件夹是否存在
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    # plt.imshow(img)
    # plt.title('id=' + str(i) + 'label=' + str(header.label))
    # plt.pause(0.1)
    # print('id=' + str(i) + 'label=' + str(header.label))

    fname = 'Figure_{}.png'.format(id)
    fpath = os.path.join(label_dir, fname)
    io.imsave(fpath, img)

