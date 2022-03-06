import csv
import os

import torch
from torchvision import transforms


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """
    Log values to file
    """

    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values, f'{col} not in {values}'
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def save_model(ckpt_dir, cp_name, model):
    """
    Create directory /Checkpoint under exp_data_path and save encoder as cp_name
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    saving_model_path = os.path.join(ckpt_dir, cp_name)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # convert to non-parallel form
    torch.save(model.state_dict(), saving_model_path)
    print(f'Model saved: {saving_model_path}')


def load_model_dic(model, ckpt_path, verbose=True, strict=True):
    """
    Load weights to encoder and take care of weight parallelism
    """
    assert os.path.exists(ckpt_path), f"trained encoder {ckpt_path} does not exist"

    try:
        model.load_state_dict(torch.load(ckpt_path), strict=strict)
    except:
        state_dict = torch.load(ckpt_path)
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=strict)
    if verbose:
        print(f'Model loaded: {ckpt_path}')

    return model


def denorm_pic(data, mean_std=([0.5071, 0.4866, 0.4409], [0.2675, 0.2565, 0.2761])):
    """
    Take in l normalized tensor of l pic and return the denormalized pic.
    :param data: A normalized tensor of l picture, shape (c, h, w)
    :return: The corresponding denormalized picture.
    """
    assert len(data.size()) == 3, f'input shape {data.shape} should be (c, h, w)'

    inverse_trans = transforms.Compose([
        transforms.Normalize(mean=-torch.tensor(mean_std[0]) / torch.tensor(mean_std[1]),
                             std=1 / torch.tensor(mean_std[1])),
        transforms.ToPILImage()

    ])
    return inverse_trans(data)


def save_normed_tensor_to_pic(tensor, path):
    pic = denorm_pic(tensor)
    pic.save(path)


def topk_accuracy(k, outputs, targets):
    """
    Compute top k accuracy
    """
    batch_size = targets.size(0)

    _, pred = outputs.topk(k, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.type(torch.FloatTensor).sum().item()

    return n_correct_elems / batch_size


def top1_accuracy(outputs, targets):
    return topk_accuracy(1, outputs, targets)


def top5_accuracy(outputs, targets):
    return topk_accuracy(5, outputs, targets)
