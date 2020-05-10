from models import *
from ops import *
from dataset import FaceDetectSet, tensor2bbox
from torch.utils.data import DataLoader
import torch
from utils import nms, box_iou
import argparse

MODEL_SAVE_PATH = "./data/mssd_face_detect.pt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence', "-c", type=float, default=0.6, help='confidence')
    parser.add_argument('--thresh', "-t", type=float, default=0.5, help='iou thresh')
    return parser.parse_args()

def statistics_result(pred_boxes, label_boxes, iou_thresh=0.5):
    correct_num = 0
    error_num = 0
    miss_num = 0
    for pbox in pred_boxes:
        is_exist = False
        for lbox in label_boxes:
            iou = box_iou(pbox, lbox)
            if iou > iou_thresh:
                is_exist = True
                label_boxes.remove(lbox)
                break
        if is_exist:
            correct_num += 1
        else:
            error_num += 1
    miss_num = len(label_boxes)
    return correct_num, error_num, miss_num


def eval(args):
    data_loader = DataLoader(dataset=FaceDetectSet(416, True), batch_size=1, shuffle=True, num_workers=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MSSD().to(device)

    # load parameter
    state = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(state['net'])

    pred_deal = MPred()
    correct_num = 0
    error_num = 0
    miss_num = 0
    for i_batch, sample_batched in enumerate(data_loader):
        img_tensor = sample_batched["img"].to(device)
        label_tensor = sample_batched["label"].to(device)
        output = model(img_tensor)
        output = pred_deal(output)

        bboxes = tensor2bbox(output[0], 416, [52, 26, 13])
        bboxes = nms(bboxes, args.confidence, args.thresh)
        label_boxes = tensor2bbox(label_tensor[0], 416, [52, 26, 13])
        c, e, m = statistics_result(bboxes, label_boxes, args.thresh)
        correct_num += c
        error_num += e
        miss_num += m

if __name__=='__main__':
    eval(parse_args())