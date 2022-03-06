import os
import time

import torch
from torch import nn
from tqdm import tqdm

from utils import AverageMeter, save_model


def train_inet(num_epochs, model, scheduler, optimizer, train_set, train_loader, test_set, test_loader, criterion,
               writer, exp_dir):
    for epoch in range(num_epochs):
        running_loss = AverageMeter()
        running_loss_val = AverageMeter()
        running_loss_val_train_mode = AverageMeter()
        start_time = time.time()

        # update learning rate at the beginning of epochs. ignore warnings
        scheduler.step(epoch)

        # train
        model.train()
        train_set.init_dataset()
        pg = tqdm(train_loader, leave=False, total=len(train_loader))
        for i, (visual_full, intentions_full, labels_full) in enumerate(pg):
            visual_full, intentions_full, labels_full = visual_full.cuda(), intentions_full.cuda(), \
                                                        labels_full[:, -1].cuda()
            left_full, mid_full, right_full = torch.split(visual_full, visual_full.size(4) // 3, dim=4)
            outs = model(left_full, mid_full, right_full, intentions_full)

            # compute loss
            loss = criterion(outs, labels_full)
            running_loss.update(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pg.set_postfix({
                'train loss': '{:.6f}'.format(running_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

        # test
        test_set.init_dataset()
        model.eval()
        with torch.no_grad():
            pg = tqdm(test_loader, leave=False, total=len(test_loader))
            for i, (visual_full, intentions_full, labels_full) in enumerate(pg):
                visual_full, intentions_full, labels_full = visual_full.cuda(), intentions_full.cuda(), \
                                                            labels_full[:, -1].cuda()
                left_full, mid_full, right_full = torch.split(visual_full, visual_full.size(4) // 3, dim=4)
                outs = model(left_full, mid_full, right_full, intentions_full)

                # compute loss
                loss = criterion(outs, labels_full)
                running_loss_val.update(loss.item())

                pg.set_postfix({
                    'test loss': '{:.6f}'.format(running_loss_val.avg),
                    'epoch': '{:03d}'.format(epoch)
                })

        # tensorboard logger
        writer.add_scalar("loss_epoch/train", running_loss.avg, epoch)
        writer.add_scalar("loss_epoch/test_train_mode", running_loss_val_train_mode.avg, epoch)
        writer.add_scalar("loss_epoch/test", running_loss_val.avg, epoch)

        print(
            f'[epoch {epoch}]: train loss {running_loss.avg:.6f},'
            f' val loss test mode {running_loss_val.avg:.6f}, time {(time.time() - start_time) / 60 :.3f} min \n')

        # checkpoint regularly
        if epoch % 40 == 39:
            save_model(os.path.join(exp_dir), f'e{epoch}.pth', model)

    writer.flush()
    writer.close()

    save_model(os.path.join(exp_dir), f'final_e{epoch}.pth', model)


def train_decision(num_epochs, model, scheduler, optimizer, train_set, train_loader, test_set, test_loader, criterion,
                   writer, exp_dir, k1, k2_n):
    for epoch in range(num_epochs):
        running_loss = AverageMeter()
        running_loss_val = AverageMeter()
        running_loss_val_train_mode = AverageMeter()
        start_time = time.time()

        # update learning rate at the beginning of epochs. ignore warnings
        scheduler.step(epoch)

        # training
        model.train()
        train_set.init_dataset()
        pg = tqdm(train_loader, leave=False, total=len(train_loader))
        for i, (visual_full, intentions_full, labels_full) in enumerate(pg):
            left_full, mid_full, right_full = torch.split(visual_full, visual_full.size(4) // 3, dim=4)
            orig_states, detached_states = [], []  # queue to store states

            # Truncated Backpropatation Through Time (TBPTT)
            for t in range(0, visual_full.size(1), k1):
                # compute predictions
                left, mid, right = left_full[:, t: t + k1].cuda(), mid_full[:, t: t + k1].cuda(), \
                                   right_full[:, t: t + k1].cuda()
                intentions, labels = intentions_full[:, t: t + k1].cuda(), labels_full[:, t: t + k1].cuda()
                outs, states = model(left, mid, right, intentions,
                                     None if len(detached_states) == 0 else detached_states[-1])

                # process states
                orig_states.append(states)
                detached_states.append(model.module.detach_states(states))
                orig_states, detached_states = orig_states[-k2_n - 1:], detached_states[-k2_n - 1:]

                # compute loss
                loss = criterion(outs, labels)
                running_loss.update(loss.item())
                loss.backward(retain_graph=False)  # backprop the loss to the last state, False still in testing
                if k2_n > 1 and t > k2_n * k1:
                    # backprop from the last state to previous k2_n states
                    for count in range(k2_n):
                        model.module.derive_grad(detached_states[-count - 2], orig_states[-count - 2])

                # optimization
                nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
                optimizer.step()
                optimizer.zero_grad()

            pg.set_postfix({
                'train loss': '{:.6f}'.format(running_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

        # test
        test_set.init_dataset()
        model.eval()
        with torch.no_grad():
            pg = tqdm(test_loader, leave=False, total=len(test_loader))
            for i, (visual_full, intentions_full, labels_full) in enumerate(pg):
                left_full, mid_full, right_full = torch.split(visual_full, visual_full.size(4) // 3, dim=4)
                states = None  # reset states
                for t in range(0, left_full.size(1), k1):
                    left, mid, right = left_full[:, t: t + k1].cuda(), mid_full[:, t: t + k1].cuda(), \
                                       right_full[:, t: t + k1].cuda()
                    intentions, labels = intentions_full[:, t: t + k1].cuda(), labels_full[:, t: t + k1].cuda()
                    outs, states = model(left, mid, right, intentions, states)

                    loss = criterion(outs, labels)
                    running_loss_val.update(loss.item())

                pg.set_postfix({
                    'test loss': '{:.6f}'.format(running_loss_val.avg),
                    'epoch': '{:03d}'.format(epoch)
                })

        # tensorboard logger
        writer.add_scalar("loss_epoch/train", running_loss.avg, epoch)
        writer.add_scalar("loss_epoch/test_train_mode", running_loss_val_train_mode.avg, epoch)
        writer.add_scalar("loss_epoch/test", running_loss_val.avg, epoch)

        print(
            f'[epoch {epoch}]: train loss {running_loss.avg:.6f},'
            f' val loss test mode {running_loss_val.avg:.6f}, time {(time.time() - start_time) / 60 :.3f} min \n')

        # save checkpoints regularly
        if epoch % 40 == 39:
            save_model(os.path.join(exp_dir), f'e{epoch}.pth', model)

    writer.flush()
    writer.close()

    save_model(os.path.join(exp_dir), f'final_e{epoch}.pth', model)
