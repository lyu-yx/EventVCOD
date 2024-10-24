# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
# torch libraries
import os
import logging
from sched import scheduler
import numpy as np
import sys
sys.path.append('')
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torchvision.utils import make_grid
# customized libraries
import metrics as Measure
from prompt_gen.prompt_generator_fpn import PromptGenerator as Network
from prompt_gen.utils import clip_gradient
from dataset import get_loader, get_test_loader

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def edge_loss(pred, edge):
    """
    dice_loss
    """
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(edge)
    pred = pred.contiguous().view(pred.shape[0], -1)
    edge = edge.contiguous().view(edge.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(pred, edge) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((pred.pow(p) + edge.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def uncertainty_loss(uncertainty, sample, mask):
    kl_loss = torch.nn.KLDivLoss(size_average=False, reduce=False)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    bce = criterion(sample, mask)
    
    uncertainty = uncertainty.squeeze(1)
    uncertainty = uncertainty.cuda(non_blocking=True)
    uncertainty = F.log_softmax(uncertainty, dim=1)
    uncertainty = uncertainty.unsqueeze(1).float()
    
    kl = kl_loss(uncertainty, mask).mean()
    
    loss_u = 0.3 * kl + bce 
    return loss_u

def giou_loss(pred_boxes, target_boxes):
    # Compute the coordinates for the smallest enclosing box
    x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
    y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
    x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
    y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
    
    enclosing_area = (x2 - x1) * (y2 - y1)

    # Compute the area of intersection and union
    intersection = torch.clamp(torch.min(pred_boxes[..., 2], target_boxes[..., 2]) - torch.max(pred_boxes[..., 0], target_boxes[..., 0]), min=0) * \
                   torch.clamp(torch.min(pred_boxes[..., 3], target_boxes[..., 3]) - torch.max(pred_boxes[..., 1], target_boxes[..., 1]), min=0)
    
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    union = pred_area + target_area - intersection
    
    iou = intersection / union
    giou = iou - (enclosing_area - union) / enclosing_area
    
    return 1 - giou.mean()


def total_loss(pred_bbox, target_bbox, λ1=1.0, λ2=1.0, λ3=0.5):
    # Smooth L1 Loss (Bounding box regression loss)
    loss_regression = F.smooth_l1_loss(pred_bbox, target_bbox)

    # IoU Loss (you can replace with GIoU, DIoU, CIoU as needed)
    loss_iou = giou_loss(pred_bbox, target_bbox)

    # Corner Loss (Top-left and bottom-right corners)
    pred_top_left = pred_bbox[..., :2]   # [x1, y1]
    pred_bottom_right = pred_bbox[..., 2:] # [x2, y2]
    target_top_left = target_bbox[..., :2] # [x1, y1]
    target_bottom_right = target_bbox[..., 2:] # [x2, y2]

    loss_corner = torch.sum((pred_top_left - target_top_left) ** 2) + torch.sum((pred_bottom_right - target_bottom_right) ** 2)

    # Total weighted loss
    total_loss = λ1 * loss_regression + λ2 * loss_iou + λ3 * loss_corner
    batch_size = pred_bbox.size(0)

    return total_loss / batch_size



def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    model.cuda() #change
    loss_all = 0
    epoch_step = 0
    
    try:
        for i, (images, gts, bbox) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            bbox = bbox.cuda()

            preds = model(images)

            loss = total_loss(preds, bbox)

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 50 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info('[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.format(epoch, opt.epoch, i, total_step, loss.data))

                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # TensorboardX-Outputs
                '''
                res = preds[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')
                res = preds[1][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_edge', torch.tensor(res), step, dataformats='HW')
                '''
        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global test_loss, best_score, best_epoch, best_metric_dict
    
    metrics_dict = dict()

    model.eval()
    with torch.no_grad():
        test_loss = 0
        total_samples = 0
        for i in range(test_loader.size):
            image, gt, bbox, _, _= test_loader.load_data()
            image = image.cuda()

            res = model(image)

            loss = total_loss(res, bbox)
            total_loss += loss.item() * bbox.size(0)  # Accumulate the loss, scaling by batch size
            total_samples += bbox.size(0)  # Accumulate the total number of samples

        cur_score = total_loss / total_samples

        if epoch == 1:
            best_score = cur_score
            print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
            logging.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')
            print('[Cur Epoch: {}, loss: {}]  [Best Epoch: {}, best loss: {}]'.format(
                epoch, best_epoch, cur_score, best_score))
            logging.info('[Cur Epoch: {}, loss: {}]  [Best Epoch: {}, best loss: {}]'.format(
                epoch, best_epoch, cur_score, best_score))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--model', type=str, default='PromptGenerator', help='main model')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='./dataset/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./dataset/TestDataset/CAMO/',
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--save_path', type=str, default='./prompt_gen/Promptgen_fpn/',
                        help='the path to save model and log')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    cudnn.benchmark = True


    model = Network(ckpt_pth='./checkpoints/sam2.1_hiera_base_plus.pt')

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = get_test_loader(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              batchsize=1,
                              testsize=opt.trainsize,
                              num_workers=4)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_score = 0
    best_epoch = 0

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print(">>> start train...")
    for epoch in tqdm(range(1, opt.epoch)):
        # schedule
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        # train
        train(train_loader, model, optimizer, epoch, save_path, writer)
        if epoch > opt.epoch//2:
            # validation
            val(val_loader, model, epoch, save_path, writer)
