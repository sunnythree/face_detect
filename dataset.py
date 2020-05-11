from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torch.utils.data import Dataset
import torch
import random
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from utils import nms

TRAIN_IMG_PATH = "/home/javer/work/dataset/widerface/WIDER_train/images/"
VAL_IMG_PATH = "/home/javer/work/dataset/widerface/WIDER_val/images/"
SPLIT_PATH = "/home/javer/work/dataset/widerface/wider_face_split"
TRAIN_SET_FILE = "/wider_face_train_bbx_gt.txt"
VAL_SET_FILE = "/wider_face_val_bbx_gt.txt"
MTRAIN = "/mtrain.txt"


def getOne(file):
    len = int(file.readline())
    bboxes = []
    for i in range(len):
        tmp = file.readline()
        datas = tmp.split(' ')
        bbox = [float(datas[0]), float(datas[1]), float(datas[2]), float(datas[3])]
        bboxes.append(bbox)
    return bboxes


def get_all_files_and_bboxes(is_train=True):
    if is_train:
        file = open(SPLIT_PATH + TRAIN_SET_FILE)
    else:
        file = open(SPLIT_PATH + VAL_SET_FILE)
    datas = []
    for line in file:
        if line.find(".jpg") >= 0:
            bboxes = getOne(file)
            datas.append({"img": line, "bboxes": bboxes})
        else:
            continue
    file.close()
    return datas


class FaceDetectSet(Dataset):
    def __init__(self, img_size, is_train=True):
        if is_train:
            self.PIC_PATH = TRAIN_IMG_PATH
        else:
            self.PIC_PATH = VAL_IMG_PATH
        self.img_size = img_size
        self.datas = get_all_files_and_bboxes(is_train)
        self.pic_strong = tfs.Compose([
            tfs.ColorJitter(0.5, 0.2, 0.2, 0.1),
            tfs.ToTensor()
        ])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # img
        img_path = self.PIC_PATH + self.datas[item]['img']
        img_path = img_path.replace('\n', '')
        img = Image.open(img_path)
        img, scaled_bboxes = pic_resize2square(img, self.img_size, self.datas[item]['bboxes'], True)
        img_tensor = self.pic_strong(img)

        # label
        feature_map = [0, 0, 0]
        feature_map[0] = (self.img_size / (2 ** 3))
        feature_map[1] = (self.img_size / (2 ** 4))
        feature_map[2] = (self.img_size / (2 ** 5))
        label_tensor = bbox2tensor(scaled_bboxes, self.img_size, feature_map)
        return img_tensor, label_tensor


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

def pic_resize2square(img, des_size, bboxes, is_random=True):
    rows = img.height
    cols = img.width
    scale_rate = float(0)

    new_rows = des_size
    new_cols = des_size
    rand_x = 0
    rand_y = 0
    if rows > cols:
        scale_rate = des_size / rows
        new_cols = math.ceil(cols * scale_rate)
        # print(rows, cols, new_rows, new_cols, scale_rate)
        rand_x = random.randint(0, math.floor(new_rows - new_cols))

    elif cols > rows:
        scale_rate = des_size / cols
        new_rows = math.ceil(rows * scale_rate)
        # print(rows, cols, new_rows, new_cols, scale_rate)
        rand_y = random.randint(0, math.floor(new_cols - new_rows))

    new_img = img.resize((new_cols, new_rows))
    scaled_img = Image.new("RGB", (des_size, des_size))
    scaled_img.paste(new_img, box=(rand_x, rand_y))
    new_bboxes = []
    for box in bboxes:
        for i in range(len(box)):
            box[i] *= scale_rate
        new_bboxes.append(
            (box[0] + rand_x + box[2] / 2, box[1] + rand_y + box[3] / 2, box[2], box[3]))
    return scaled_img, new_bboxes


def bbox2tensor(bboxes, img_size, feature_map):
    label_tensor = torch.zeros((int((feature_map[0] ** 2) + (feature_map[1] ** 2) + (feature_map[2] ** 2)), 5))
    bbox_index = 0
    thresh1 = img_size/10
    thresh2 = img_size/20

    for box in bboxes:
        w = box[2]
        h = box[3]
        max_edge = max(w, h)
        cell_size = 0
        start_index = 0
        feature_size = 0
        if max_edge > thresh1:
            # predict by first feature map
            start_index = int((feature_map[0] ** 2) + (feature_map[1] ** 2))
            cell_size = img_size/feature_map[2]
            feature_size = feature_map[2]
        elif max_edge > thresh2:
            # predict by second feature map
            start_index = int((feature_map[0] ** 2))
            cell_size = img_size / feature_map[1]
            feature_size = feature_map[1]
        else:
            # predict by third feature map
            start_index = 0
            cell_size = img_size / feature_map[0]
            feature_size = feature_map[0]

        cell_x_index = math.floor(box[0] / cell_size)
        cell_x_bias = box[0] % cell_size
        cell_y_index = math.floor(box[1] / cell_size)
        cell_y_bias = box[1] % cell_size
        p_box = label_tensor[math.floor(start_index + cell_y_index * feature_size + cell_x_index), :]
        p_box[0] = 1
        p_box[1] = cell_x_bias/cell_size
        p_box[2] = cell_y_bias/cell_size
        p_box[3] = box[2]/img_size
        p_box[4] = box[3]/img_size


    return label_tensor


def tensor2bbox(out_tensor, img_size, feature_map, thresh=0.5):
    assert out_tensor.dim() == 2
    bboxes = []
    for i in range(out_tensor.shape[0]):
        bbox = out_tensor[i, :]
        # print(bbox)
        if bbox[0] > thresh:
            feature_size = 0
            r_index = 0
            thresh_1 = feature_map[0] ** 2 + feature_map[1] ** 2
            thresh_2 = feature_map[0] ** 2
            if i > thresh_1:
                feature_size = feature_map[2]
                r_index = i - thresh_1
            elif i > thresh_2:
                feature_size = feature_map[1]
                r_index = i - thresh_2
            else:
                feature_size = feature_map[0]
                r_index = i
            start_x = math.floor(r_index % feature_size)
            start_y = math.floor(r_index / feature_size)
            cell_size = img_size / feature_size
            bbox[1] = bbox[1] * cell_size + start_x * cell_size
            bbox[2] = bbox[2] * cell_size + start_y * cell_size
            bbox[3] = bbox[3] * img_size
            bbox[4] = bbox[4] * img_size
            bboxes.append(bbox)
    return bboxes


def test_dataset():
    data_loader = DataLoader(dataset=FaceDetectSet(416, True), batch_size=1, shuffle=True)
    transform = tfs.Compose([tfs.ToPILImage()])
    for i_batch, sample_batched in enumerate(data_loader):
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)  # 开启一个窗口，同时设置大小，分辨率
        origin_img = transform(sample_batched[0][0])
        bboxes = tensor2bbox(sample_batched[1][0], 416, [52, 26, 13])
        bboxes = nms(bboxes, 0.5, 0.5)
        draw = ImageDraw.Draw(origin_img)
        for bbox in bboxes:
            draw.rectangle((bbox[1] - bbox[3] / 2, bbox[2] - bbox[4] / 2, bbox[1] + bbox[3] / 2, bbox[2] + bbox[4] / 2),
                           outline=(0, 255, 0))
        print(bboxes)
        plt.imshow(origin_img)
        plt.show()
        plt.close()


# test_dataset()
