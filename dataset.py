import math
import os

import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data
from torch.utils.data import Sampler

from augmentations import *


def div_sub_lists(lst):
    """
    Divide a list into sub-lists, each contains the same elements, and return their indices
    :param lst: a list of elements
    :return: indices of elements
    """
    if len(lst) == 0:
        return []

    last = lst[0]
    lsts, current_lst = [], []
    for i, ele in enumerate(lst):
        if ele == last:
            current_lst.append(i)
        else:
            lsts.append(current_lst)
            current_lst = [i]
            last = ele
    lsts.append(current_lst)

    return lsts


def get_items(lst, indices):
    return [lst[i] for i in indices]


def chunk_by_max_len(lst, max_len, drop_last=False, rand_start=False, cover_all=True, interval=1):
    """
    divide a list into sub-lists by max length
    :param lst: the original list
    :param max_len: max length of sub lists
    :param rand_start: whether to choose the starting index in the original list randomly
    :param cover_all: whether to cover all indices falling in the interval
    :param interval: sample one item every interval items
    :return: a list containing sublists
    """
    if len(lst) == 0:
        return []

    start = random.randrange(0, max(max_len // interval, 1)) if rand_start else 0
    if rand_start:
        if cover_all:
            raise NotImplementedError()
        result = []
        while True:
            result.append(lst[start: start + max_len: interval])
            start += random.randrange(0, max(1, max_len * 2))
            if len(result[-1]) < max_len // interval:
                if drop_last:
                    result.pop()
                return result
    else:
        result = [lst[i: i + max_len: interval] for i in range(start, len(lst), max_len)]
        if cover_all:
            for j in range(1, interval):
                result.extend([lst[i: i + max_len: interval] for i in range(start + j, len(lst), max_len)])
        if drop_last:
            result = list(filter(lambda x: len(x) == math.ceil(max_len / interval), result))

    # modify each seq s.t. the interval between the last and sec last element is 1.
    # Check duplicates in lst.
    assert sorted(list(set(lst))) == sorted(lst), f'list {lst} has duplicates. Not accepted.'

    return result


class SeqDataset(data.Dataset):
    """
    The dataset class.
    """
    MAX_SKIP_FRAMES = 3  # the max number of frames that can be skipped
    INTENTION_MAPPING = {'forward': 0, 'left': 1, 'right': 2, 'elevator': 3, 'unknown': 4}
    VIEW_DIRS = ['left_color', 'mid_color', 'right_color']
    NUM_REPEAT = 1
    MIN_CHUNK_LEN = 50
    SEED = 0

    def __init__(self, annotation_path, data_directory, spatial_size, seq_len, interval,
                 mean_std=([0.5071, 0.4866, 0.4409], [0.2675, 0.2565, 0.2761]), aug=True, keep_prob=0.1, flip=False,
                 num_intention=None, elevator_only=False, views=[0, 1, 2]):
        super().__init__()
        assert os.path.exists(data_directory), f'data directory {data_directory} does not exist'
        self.fix_seed(self.SEED)

        self.views_dir = [os.path.join(data_directory, folder) for folder in self.VIEW_DIRS]
        self.spatial_size = spatial_size
        self.seq_len = seq_len
        self.MIN_CHUNK_LEN = 1  # modified to accommodate random length sampling
        self.interval = interval
        self.aug = aug
        self.keep_prob = keep_prob
        self.rand_len = False
        self.flip = flip
        self.elevator_only = elevator_only
        self.VIEW_DIRS = [self.VIEW_DIRS[view] for view in views]

        # filter samples with invalid intentions
        anno = pd.read_csv(annotation_path, sep=' ')
        anno_cleaned = anno.loc[anno['dlm'].apply(lambda x: self.INTENTION_MAPPING[x] < num_intention)].reset_index(
            drop=True)
        if elevator_only:
            anno_cleaned = anno.loc[anno['dlm'].apply(lambda x: x == 'elevator')].reset_index(drop=True)
        self.annotation = anno_cleaned
        if len(anno) != len(anno_cleaned):
            print(f'{len(anno) - len(anno_cleaned)} samples filtered from annotation due to invalid intention. '
                  f'Num of intention = {num_intention}')

        if aug:
            if flip:
                self.preprocess = Compose([
                    ToPILImage(),
                    ColorJitter(brightness=(0.67, 1.33), contrast=(0.67, 1.33), saturation=(0.67, 1.33),
                                hue=(-0.11, 0.11),
                                differ_for_each_frame=False),  # the perturbation ranges should center around 0 or 1
                    Grayscale(p=0.11),
                    HorizontalFlip(),
                    Resize((spatial_size[0], spatial_size[1])),
                    ToTensor(),
                    Normalize(mean=mean_std[0], std=mean_std[1])
                ])
            else:
                self.preprocess = Compose([
                    ToPILImage(),
                    ColorJitter(brightness=(0.67, 1.33), contrast=(0.67, 1.33), saturation=(0.67, 1.33),
                                hue=(-0.11, 0.11),
                                differ_for_each_frame=False),
                    Grayscale(p=0.11),
                    Resize((spatial_size[0], spatial_size[1])),
                    ToTensor(),
                    Normalize(mean=mean_std[0], std=mean_std[1])
                ])
        else:
            self.preprocess = Compose([
                ToPILImage(),
                Resize((spatial_size[0], spatial_size[1])),
                ToTensor(),
                Normalize(mean=mean_std[0], std=mean_std[1])
            ])

        # read data
        self.orig_len = len(self.annotation)  # original length of train_set
        print(f'train_set located at {data_directory} of original length {self.orig_len} , '
              f'spatial size {spatial_size}')

        # divide into raw chunks
        chunks, chunk = [], []
        for idx in range(self.orig_len):
            if idx == 0 or int(self.annotation['frame'][idx]) - int(self.annotation['frame'][idx - 1]) \
                    < self.MAX_SKIP_FRAMES:
                chunk.append(idx)
            else:
                chunks.append(chunk)
                chunk = [idx]

            if idx % 1e4 == 0:
                print(f'data loading [ {idx} / {self.orig_len} ]')
        chunks.append(chunk)  # deal with the last chunk

        self.chunks = chunks
        print(f'train_set divided into {len(chunks)} raw chunks of length {[len(c) for c in chunks]}')

        self.init_dataset()

    @staticmethod
    def fix_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_dataset(self):
        # divide each chunk into trajectories
        new_chunks = []
        for chunk in self.chunks:
            intentions = get_items(self.annotation['dlm'], chunk)
            sub_idss = div_sub_lists(intentions)
            for sub_ids in sub_idss:
                anno_ids = get_items(chunk, sub_ids)
                new_chunks.append(anno_ids)

        # cut sequences longer than upper bound and filter short sequences
        chunks = []
        for chunk in new_chunks:
            seq_len = 2 * self.seq_len if get_items(self.annotation['dlm'], chunk)[0] == 'elevator' else self.seq_len
            chunks.extend(
                chunk_by_max_len(chunk, seq_len * self.interval + 1, rand_start=True, drop_last=True,
                                 interval=self.interval, cover_all=False))

        # size should be one larger, and at least 2
        chunks = list(
            filter(lambda x: len(x) > (max(self.MIN_CHUNK_LEN, 1) if self.rand_len else max(self.seq_len, 1)), chunks))

        # construct sample
        dataset = []
        num_filtered, num_flipped = 0, 0
        for i, chunk in enumerate(chunks):
            chunk = chunk[:-1]  # drop the last one (implicit assumption: length > 1)
            # note the +1 here
            vs = get_items(self.annotation['current_velocity'],
                           [idx + max(1, self.interval) for idx in chunk])
            ss = get_items(self.annotation['steering_wheel_angle'],
                           [idx + max(1, self.interval) for idx in chunk])
            labels = [[vs[j], ss[j]] for j in range(len(chunk))]
            num_nonzero_velocity = len(list(filter(lambda v: v != 0, vs)))
            num_nonzero_steerings = len(list(filter(lambda s: s != 0, ss)))
            # construct sample
            sample_1 = {
                'frames': get_items(self.annotation['frame'], chunk),
                'intentions': get_items(self.annotation['dlm'], chunk),
                'labels': labels,
                'num_nonzero_steerings': num_nonzero_steerings,
                'flip': False  # False for the first sample! This is the real sample.
            }
            sample_2 = {
                'frames': get_items(self.annotation['frame'], chunk),
                'intentions': get_items(self.annotation['dlm'], chunk),
                'labels': labels,
                'num_nonzero_steerings': num_nonzero_steerings,
                'flip': True
            }

            # take care of stair case climbing samples: no flipping and skipping
            if 8e5 <= sample_1['frames'][0] <= 9e5:
                dataset.append(sample_1)
                continue

            # filter intention == 'forward' and steerings are all zero samples
            if num_nonzero_velocity == len(vs) and num_nonzero_steerings == 0 \
                    and sample_1['intentions'].count('forward') == len(sample_1['intentions']) \
                    and random.random() > self.keep_prob:  # double test loss in addition to the augmentation
                num_filtered += 1
                continue
            dataset.append(sample_1)

            # flip forward intention v8 samples only (obstacle avoidance), assuming 4e4 < frames < 5e5
            if self.aug and self.flip and 'forward' in sample_1['intentions'] and 4e5 <= sample_1['frames'][0] < 5e5:
                dataset.append(sample_2)
                num_flipped += 1

        self.dataset = dataset
        print(f'train_set init, length {len(self.dataset)}, aug = {self.aug}, flip = {self.flip}, '
              f'rand length = {self.rand_len}\n'
              f'num of filtered samples (ratio): {num_filtered} ({1 - self.keep_prob}), '
              f'num of flipped samples = {num_flipped}, '
              f'views are {self.VIEW_DIRS}')

    def _read_img(self, name, view):
        # view should be in [0, 1, 2]
        path = os.path.join(self.views_dir[view], name)
        return cv2.cvtColor(cv2.imread(path).astype(np.uint8), cv2.COLOR_BGR2RGB)  # convert from BGR to RGB!

    def _read_multiview_imgs(self, name):
        return [self._read_img(name, view) for view in range(0, len(self.VIEW_DIRS))]

    def _read_images(self, frames):
        visuals = []
        for i, frame in enumerate(frames):
            images = self._read_multiview_imgs(str(frame) + '.jpg')
            visuals.append(np.concatenate(images, axis=1))  # axis=1 because it's cv2 image format! (H, W, C)
        return visuals

    def __getitem__(self, idx):
        # only retrieve one sample in samples
        if torch.is_tensor(idx):
            idx = idx.item()

        # basic info
        instance = self.dataset[idx]
        frames = instance['frames']

        # read frames
        visuals = self._read_images(frames)

        # intention and label
        intents, labels, flip = instance['intentions'], instance['labels'], instance['flip']
        visuals, intents, labels = self.preprocess(visuals, intents, labels, flip)

        # to tensor
        visuals = torch.stack(visuals, dim=0)
        intentions = [torch.tensor(self.INTENTION_MAPPING[intent]) for intent in intents]
        intentions = torch.stack(intentions, dim=0)
        labels = torch.tensor(labels)

        # maybe this type cast is more efficient than the above commented out lines
        new_visuals, new_intentions, new_labels = visuals.float(), intentions.float(), labels.float()

        return new_visuals, new_intentions, new_labels

    def __len__(self):
        return len(self.dataset)


class BatchSampler(Sampler):
    """
    The sampler that forces the intentions of samples in every batch are the same.
    Switch off shuffle in the torch Dataloader if sampler is specified.
    Instead, manually shuffle the indices via shuffle() method.
    """

    def __init__(self, dataset, subset, batch_size, drop_last=True, shuffle=True):
        super().__init__(None)
        self.dataset = dataset
        self.subset = subset
        self.forward, self.left, self.right, self.elevator = self.group_samples()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_on = shuffle
        self.length = len(list(self.__iter__()))

    def group_samples(self):
        indices = list(range(len(self.dataset)))
        forward, left, right, elevator = {}, {}, {}, {}
        for idx, dataset_idx in enumerate(indices):
            sample = self.dataset.dataset[dataset_idx]
            intention = sample['intentions'][0]
            flip = sample['flip']
            sample_len = len(sample['intentions'])
            if intention == 'forward':
                dic = forward
            elif (intention == 'left' and not flip) or (intention == 'right' and flip):
                dic = left
            elif (intention == 'right' and not flip) or (intention == 'left' and flip):
                dic = right
            elif intention == 'elevator':
                dic = elevator
            else:
                raise NotImplementedError(f'unknown intention {intention}')
            if sample_len not in dic.keys():
                dic[sample_len] = []
            dic[sample_len].append(idx)
        return forward, left, right, elevator

    def shuffle(self):
        for group in [self.forward, self.left, self.right, self.elevator]:  # for each group
            for value in group.values():
                # shuffle the list
                random.shuffle(value)
            print(f'group sample length: {group.keys()}, group size {[len(x) for x in group.values()]}')
        print(f'sampler shuffled')

    def __iter__(self):
        # self.train_set.init_dataset()   # bugs here? Batch intention messed up. Manual init preferred.
        self.forward, self.left, self.right, self.elevator = self.group_samples()
        if self.shuffle_on:
            self.shuffle()        
        batch_lists = []
        for group in [self.forward, self.left, self.right, self.elevator]:
            # for each group. easy samples at first when no shuffle
            for value in group.values():
                batch_by_seq_len = chunk_by_max_len(value, self.batch_size, drop_last=self.drop_last)
                for batch_list in batch_by_seq_len:
                    batch_lists.append(batch_list)
        if self.shuffle_on:
            random.shuffle(batch_lists)
        flattened = [idx for batch_list in batch_lists for idx in batch_list]
        return iter(flattened)

    def __len__(self):
        return self.length
