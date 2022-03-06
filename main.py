import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SeqDataset, BatchSampler
from models.decision import DECISION
from models.inet import INet
from train import train_inet, train_decision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'Training INet models')
    parser.add_argument('--model', type=str, help='the cuda devices used for training',
                        choices=['inet, lstm, decision'], default='decision')
    parser.add_argument('--modes', type=int, help='number of modes', default=4)
    parser.add_argument('--k1', type=int, help='value of k1 for TBPTT', default=2)
    parser.add_argument('--k2-n', type=int, help='the multiplicative factor of k1 to obtain k2 in TBPTT', default=5)
    parser.add_argument('--input-size', type=int, help='the size of input visual percepts', default=112)
    parser.add_argument('--gpu', type=str, help='the cuda devices used for training', default="0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-frames', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--frame-interval', help='sample 1 frame every x frames', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.7)
    parser.add_argument('--intent-feat', help='whether or not to use intention features', type=bool, default=True)
    parser.add_argument('--num-modes', type=bool, default=4)
    parser.add_argument('--exp-log-path', help='path to log experiment data', type=str, default='exp/inet')
    parser.add_argument('--dataset-path', help='path to dataset', type=str, default='sample_dataset')
    parser.add_argument('--downsample-ratio', help='the ratio by which to downsample particular samples in the dataset',
                        type=int, default=0.1)

    # assume there are 3 robot cameras (left, mid, right)
    NUM_VIEWS = 3

    # basic configuration
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)

    # basic training setup
    writer = SummaryWriter(os.path.join(args.exp_log_path, 'board'))
    exp_dir = f'exp_data/inet'
    os.makedirs(args.exp_log_path, exist_ok=True)
    train_anno_path = os.path.join(args.dataset_path, 'train.txt')
    val_anno_path = os.path.join(args.dataset_path, 'test.txt')

    # data loaders
    image_shape = (args.input_size, args.input_size * NUM_VIEWS)
    train_set = SeqDataset(train_anno_path, args.dataset_path, image_shape, args.num_frames,
                           args.frame_interval, aug=True, keep_prob=args.downsample_ratio, flip=True,
                           num_intention=args.num_modes, elevator_only=False)
    train_sampler = BatchSampler(train_set, None, args.batch_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, shuffle=False,
                              num_workers=32, pin_memory=True, drop_last=False)

    test_set = SeqDataset(train_anno_path, args.dataset_path, image_shape, args.num_frames,
                          args.frame_interval, aug=False, keep_prob=args.downsample_ratio, flip=True,
                          num_intention=args.num_modes, elevator_only=False)
    test_sampler = BatchSampler(train_set, None, args.batch_size)
    test_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=test_sampler, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=False)

    # objective function
    criterion = torch.nn.MSELoss()

    # training
    if args.model == 'inet':
        model = INet(pretrained=True, fc_dropout_keep=args.dropout, intent_feat=False, num_modes=3,
                     num_frames=args.num_frames)
        model = nn.DataParallel(model).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7 * args.batch_size * args.num_frames,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 140], gamma=0.1)
        train_inet(args.epochs, model, scheduler, optimizer, train_set, train_loader, test_set, test_loader,
                   criterion, writer, exp_dir)
    else:
        model = DECISION((args.input_size, args.input_size), channels=[128, 192, 256], sep_lstm=True, sep_fc=True,
                         skip_depth=[], num_modes=args.modes, controller_name=args.model)
        model = nn.DataParallel(model).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7 * args.batch_size * args.num_frames,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 140], gamma=0.1)
        train_decision(args.epochs, model, scheduler, optimizer, train_set, train_loader, test_set, test_loader,
                       criterion, writer, exp_dir, args.k1, args.k2_n)
