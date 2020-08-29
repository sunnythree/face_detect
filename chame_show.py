from models import *
from dataset import FaceDetectSet, tensor2bbox
import torch
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms as tfs
import cv2
import numpy as np
from utils import nms, box_iou

MODEL_SAVE_PATH = "./data/mssd_face_detect.pt"



def cv2image(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image

def image2cv(image):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image

def get_pytorch_model(path):
    state = torch.load(path)
    model = MSSD()
    model.load_state_dict(state['net'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    face_detect = get_pytorch_model(MODEL_SAVE_PATH)
    face_detect.to(device)
    to_tensor = tfs.ToTensor()
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, img = cap.read()
        image = cv2image(img)
        image = image.resize((416, 416))
        output = face_detect(to_tensor(image).view(-1, 3, 416, 416).to(device))
        #save one pic and output
        bboxes = tensor2bbox(output[0], 416, [52, 26, 13], thresh=0.9)
        bboxes = nms(bboxes, 0.9, 0.5)
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.text((bbox[1] - bbox[3] / 2, bbox[2] - bbox[4] / 2 - 10), str(round(bbox[0].item(), 2)), fill=(255, 0, 0))
            draw.rectangle((bbox[1] - bbox[3] / 2, bbox[2] - bbox[4] / 2, bbox[1] + bbox[3] / 2, bbox[2] + bbox[4] / 2),
                           outline=(0, 255, 0))
            draw.rectangle((bbox[1] - bbox[3] / 2 + 1, bbox[2] - bbox[4] / 2 + 1, bbox[1] + bbox[3] / 2 - 1, bbox[2] + bbox[4] / 2 - 1),
                           outline=(0, 255, 0))
        img = image2cv(image)
        cv2.imshow("test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()