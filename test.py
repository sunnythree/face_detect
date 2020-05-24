from models import *
from dataset import FaceDetectSet, tensor2bbox
import torch
from utils import nms, box_iou
import time
import os
from PIL import Image
from dataset import pic_resize2square
from torchvision import transforms as tfs
import argparse
import math
from PIL import ImageDraw
import matplotlib.pyplot as plt

MODEL_SAVE_PATH = "./data/mssd_face_detect.pt"

TEST_IMG_PATH = "/home/javer/work/dataset/widerface/WIDER_test/images/"
SPLIT_PATH = "/home/javer/work/dataset/widerface/wider_face_split"
TEST_SET_FILE = "/wider_face_test_filelist.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence', "-c", type=float, default=0.6, help='confidence')
    parser.add_argument('--thresh', "-t", type=float, default=0.3, help='iou thresh')
    parser.add_argument('--show', "-s", type=bool, default=False, help='iou thresh')
    return parser.parse_args()


def statistics_result(pred_boxes, label_boxes, iou_thresh=0.5):
    correct_num = 0
    error_num = 0
    miss_num = 0
    for pbox in pred_boxes:
        is_exist = False
        for lbox in label_boxes:
            if lbox[4] == 0:
                continue
            iou = box_iou(pbox, lbox)
            if iou > iou_thresh:
                is_exist = True
                lbox[4] = 0
                break
        if is_exist:
            correct_num += 1
        else:
            error_num += 1
    miss_num = len(label_boxes) - correct_num
    return correct_num, error_num, miss_num


def eval(args):
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    model = MSSD().to(device)

    # load parameter
    state = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(state['net'])

    all_cost = 0

    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/test"):
        os.mkdir("data/test")

    to_tensor = tfs.Compose([
            tfs.ColorJitter(0., 0., 0., 0.),
            tfs.ToTensor()
        ])
    count = 0
    for line in open(SPLIT_PATH + TEST_SET_FILE):
        img_path = line
        img_path = img_path.replace('\n', '')
        img_origin = Image.open(TEST_IMG_PATH+img_path)
        scaled_img, scaled_bboxes = pic_resize2square(img_origin, 416, [], False)

        img_tensor = to_tensor(scaled_img).to(device)
        img_tensor = img_tensor.view(1, 3, 416, 416)
        path, file_name = img_path.split('/')
        if not os.path.exists("data/test/"+path):
            os.mkdir("data/test/"+path)
        path, img_name = img_path.split('/')
        out_name = img_name.replace("jpg", "txt")
        out_name = out_name.replace("png", "txt")
        if os.path.exists("data/test/"+path+"/"+out_name):
            os.remove("data/test/"+path+"/"+out_name)
        eval_result = open("data/test/"+path+"/"+out_name, "x")
        eval_result.write(file_name+"\n")
        start = time.time()
        output = model(img_tensor)
        end = time.time()
        all_cost += (end - start)

        bboxes = tensor2bbox(output[0].cpu(), 416, [52, 26, 13], thresh=args.confidence)
        bboxes = nms(bboxes, args.confidence, args.thresh)

        width = img_origin.width
        height = img_origin.height
        if width > height:
            scale_rate = 416.0 / width
            x_offset = 0
            y_offset = math.floor((416.0 - height * scale_rate) / 2)
        else:
            scale_rate = 416.0 / height
            x_offset = math.floor((416.0 - width * scale_rate) / 2)
            y_offset = 0
        for bbox in bboxes:
            bbox[1] = (bbox[1] - x_offset) / scale_rate
            bbox[2] = (bbox[2] - y_offset) / scale_rate
            bbox[3] = bbox[3] / scale_rate
            bbox[4] = bbox[4] / scale_rate
            # change format
            bbox[1] = bbox[1] - bbox[3] / 2
            bbox[2] = bbox[2] - bbox[4] / 2
            bbox[3] = bbox[1] + bbox[3] / 2
            bbox[4] = bbox[2] + bbox[4] / 2

        if args.show:
            draw = ImageDraw.Draw(img_origin)
            for bbox in bboxes:
                draw.rectangle((bbox[1], bbox[2], bbox[3], bbox[4]), outline=(0, 255, 0))
            plt.imshow(img_origin)
            plt.show()



        eval_result.write(str(len(bboxes)) + "\n")
        for bbox in bboxes:
            eval_result.write(str(bbox[1].item())+' '+str(bbox[2].item())+' '+str(bbox[3].item())+' '+str(bbox[4].item())+' '+str(bbox[0].item()) + "\n")
        eval_result.close()
        count += 1
        print("predicted: "+str(count), end='\r')




if __name__=='__main__':
    eval(parse_args())