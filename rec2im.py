import mxnet as mx
import mxnet.ndarray as nd
from skimage import io
import numpy as np
from mxnet import recordio
import matplotlib.pyplot as plt
import os, argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

path_prefix = 'datasets/faces_webface_112x112/train'


def extrat_image(recordio, imgrec, output_dir, index):
    try:
        header, s = recordio.unpack(imgrec.read_idx(index + 1))
        img = mx.image.imdecode(s).asnumpy()
        label = str(header.label)
        id = str(index)

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
    except Exception as e:
        print(e)
        pass


def _main(args):
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    path_imgidx = args.path_imgidx
    path_imgrec = args.path_imgrec
    count = args.image_count

    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

    # 创建线程池
    # executor = ThreadPoolExecutor(max_workers=args.max_workers)

    for i in tqdm(range(count)):
        # executor.submit(extrat_image(recordio, imgrec, output_dir, i))
        extrat_image(recordio, imgrec, output_dir, i)

def get_parser():
    parser = argparse.ArgumentParser(description='recovery mxnet rec file to image')
    parser.add_argument('path_imgrec', help='rec file e.g. datasets/faces_webface_112x112/train.rec',
                        type=str)
    parser.add_argument('path_imgidx', help='idx file e.g. datasets/faces_webface_112x112/train.idx',
                        type=str)
    parser.add_argument('output_dir', help='image save directory e.g. datasets/align_webface_112x112',
                        type=str)
    parser.add_argument('image_count', help='number of images to extract',
                        type=int, default=501195)
    parser.add_argument('--max_workers', help='number of thread to extract image',
                        type=int, default=8)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    e.g. command
    python rec2im.py \
    /workspace/dataset/faces_emore/train.rec \
    /workspace/dataset/faces_emore/train.idx \
    /workspace/dataset/face_ms1m/ \
    10000000
    8
    """
    _main(get_parser())
