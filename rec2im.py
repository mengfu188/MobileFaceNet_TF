import mxnet as mx
import mxnet.ndarray as nd
from skimage import io
import numpy as np
from mxnet import recordio
import matplotlib.pyplot as plt
import os, argparse
from tqdm import tqdm

path_prefix = 'datasets/faces_webface_112x112/train'


def _main(args):
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    path_imgidx = args.path_imgidx
    path_imgrec = args.path_imgrec
    count = args.image_count

    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

    for i in tqdm(range(count)):
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

def get_parser():
    parser = argparse.ArgumentParser(description='recovery mxnet rec file to image')
    parser.add_argument('path_imgrec', help='rec file e.g. datasets/faces_webface_112x112/train.rec',
                        type=str)
    parser.add_argument('path_imgidx', help='idx file e.g. datasets/faces_webface_112x112/train.idx',
                        type=str)
    parser.add_argument('output_dir',  help='image save directory e.g. datasets/align_webface_112x112',
                        type=str)
    parser.add_argument('image_count', help='number of images to extract',
                        type=int, default=501195)

    args = parser.parse_args()
    return args

if __name__ =='__main__':
    """
    e.g. command
    python rec2im.py \
    /workspace/datasets/face_emore/train.rec \
    /workspace/datasets/face_emore/train.idx \
    /workspace/datasets/face_ms1m/ \
    10000000
    """
    _main(get_parser())

