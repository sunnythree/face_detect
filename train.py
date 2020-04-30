from PIL import ImageDraw

from models import *
from ops import *
from summary import writer
from dataset import FaceDetectSet, tensor2bbox
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import torch
import os
from utils import progress_bar, nms
from torchvision import transforms as tfs

MODEL_SAVE_PATH = "./data/mssd_face_detect.pt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gama', "-g", type=float, default=0.9, help='train gama')
    parser.add_argument('--step', "-s", type=int, default=20, help='train step')
    parser.add_argument('--batch', "-b", type=int, default=1, help='train batch')
    parser.add_argument('--epoes', "-e", type=int, default=30, help='train epoes')
    parser.add_argument('--lr', "-l", type=float, default=0.001, help='learn rate')
    parser.add_argument('--pretrained', "-p", type=bool, default=False, help='prepare trained')
    return parser.parse_args()

def train(args):
    start_epoch = 0
    data_loader = DataLoader(dataset=FaceDetectSet(416, True), batch_size=args.batch, shuffle=True, num_workers=8)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MSSD()
    print("add graph")
    writer.add_graph(model, torch.zeros((1, 3, 416, 416)))
    print("add graph over")
    if args.pretrained and os.path.exists(MODEL_SAVE_PATH):
        print("loading ...")
        state = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(state['net'])
        start_epoch = state['epoch']
        print("loading over")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gama)
    train_loss = 0
    loss_func = MLoss()
    to_pil_img = tfs.ToPILImage()
    to_tensor = tfs.ToTensor()
    for epoch in range(start_epoch, start_epoch+args.epoes):
        model.train()
        for i_batch, sample_batched in enumerate(data_loader):
            optimizer.zero_grad()
            img_tensor = sample_batched["img"].to(device)
            label_tensor = sample_batched["label"].to(device)
            output = model(img_tensor)
            loss = loss_func(output, label_tensor)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            global_step = epoch*len(data_loader)+i_batch
            progress_bar(i_batch, len(data_loader), 'loss: %f, epeche: %d'%(train_loss, epoch))
            writer.add_scalar("loss", train_loss, global_step=global_step)

        #save one pic and output
        if epoch % 10 == 0:
            pil_img = to_pil_img(sample_batched['img'][0])
            bboxes = tensor2bbox(output[0], 416, [52, 26, 13])
            bboxes = nms(bboxes, 0.5)
            draw = ImageDraw.Draw(pil_img)
            for bbox in bboxes:
                draw.rectangle((bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2),
                               outline=(0, 255, 0))
            writer.add_image("img: "+str(epoch), to_tensor(pil_img))
        scheduler.step()

    if not os.path.isdir('data'):
        os.mkdir('data')
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, MODEL_SAVE_PATH)
    writer.close()

if __name__=='__main__':
    train(parse_args())

