from models import *
from ops import *
from dataset import FaceDetectSet, tensor2bbox
from torch.utils.data import DataLoader
import torch
from utils import nms, box_iou
import argparse
import time
import os
import math

MODEL_SAVE_PATH = "./data/mssd_face_detect.pt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence', "-c", type=float, default=0, help='confidence')
    parser.add_argument('--thresh', "-t", type=float, default=0, help='iou thresh')
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
    data_loader = DataLoader(dataset=FaceDetectSet(416, False, False), batch_size=1, shuffle=False, num_workers=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MSSD().to(device)

    # load parameter
    state = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(state['net'])

    # correct_num = 0
    # error_num = 0
    # miss_num = 0
    # all_num = 0
    all_cost = 0

    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/eval"):
        os.mkdir("data/eval")

    dataset_size = len(data_loader)
    for i_batch, sample_batched in enumerate(data_loader):
        img_tensor = sample_batched[0].to(device)
        label_tensor = sample_batched[1].to(device)
        img_path = sample_batched[2][0][0]
        img_path = img_path[img_path.find("images/")+7:len(img_path)]
        path, file_name = img_path.split('/')
        if not os.path.exists("data/eval/"+path):
            os.mkdir("data/eval/"+path)
        path, img_name = img_path.split('/')
        out_name = img_name.replace("jpg", "txt")
        out_name = out_name.replace("png", "txt")
        if os.path.exists("data/eval/"+path+"/"+out_name):
            os.remove("data/eval/"+path+"/"+out_name)
        eval_result = open("data/eval/"+path+"/"+out_name, "x")
        eval_result.write(file_name+"\n")
        start = time.time()
        output = model(img_tensor)
        end = time.time()
        all_cost += (end - start)

        bboxes = tensor2bbox(output[0], 416, [52, 26, 13], thresh=args.confidence)
        # bboxes = nms(bboxes, args.confidence, args.thresh)
        # label_boxes = tensor2bbox(label_tensor[0], 416, [52, 26, 13])
        # all_num += len(label_boxes)
        # c, e, m = statistics_result(bboxes, label_boxes, args.thresh)
        # correct_num += c
        # error_num += e
        # miss_num += m
        # print("c,e,m=", correct_num, error_num, miss_num)
        eval_result.write(str(len(bboxes)) + "\n")
        width = sample_batched[2][1][0].item()
        height = sample_batched[2][1][1].item()
        if width > height:
            scale_rate = 416.0/width
            x_offset = 0
            y_offset = math.floor((416.0 - height*scale_rate)/2)
        else:
            scale_rate = 416.0/height
            x_offset = math.floor((416.0 - width*scale_rate)/2)
            y_offset = 0
        for bbox in bboxes:
            bbox[1] = (bbox[1] - x_offset) / scale_rate
            bbox[2] = (bbox[2] - y_offset) / scale_rate
            bbox[3] = bbox[3] / scale_rate
            bbox[4] = bbox[4] / scale_rate
            # change format
            bbox[1] = bbox[1] - bbox[3]/2
            bbox[2] = bbox[2] - bbox[4]/2
            bbox[3] = bbox[1] + bbox[3]
            bbox[4] = bbox[2] + bbox[4]

            eval_result.write(str(bbox[1].item())+' '+str(bbox[2].item())+' '+str(bbox[3].item())+' '+str(bbox[4].item())+' '+str(bbox[0].item())+"\n")
        eval_result.close()
        print("process "+str(i_batch)+"/"+str(dataset_size))

    # print("correct rate: "+str(correct_num / all_num*100)+"%")
    # print("error rate: " + str(error_num / all_num*100)+"%")
    # print("miss rate: " + str(miss_num / all_num*100)+"%")
    # print("mean inferince is: " + str(all_cost / len(data_loader)))

if __name__=='__main__':
    eval(parse_args())