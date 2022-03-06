import numbers
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        """
        Compose transforms for a sequence images
        """
        self.transforms = transforms

    def __call__(self, imgs, intentions, labels, flip):
        for t in self.transforms:
            imgs, intentions, labels, flip = t(imgs, intentions, labels, flip)
        return imgs, intentions, labels


class ToPILImage(object):
    def __call__(self, imgs, intentions, labels, flip):
        new_imgs = [TF.to_pil_image(img, mode=None) for img in imgs]
        return new_imgs, intentions, labels, flip


class Crop(object):
    def __init__(self, ratio_range):
        self.ratio_range = ratio_range

    def __call__(self, imgs, intentions, labels, flip):
        ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

        w, h = imgs[0][0].size
        h2, w2 = ratio * h, ratio * w
        top_left_x = random.uniform(0, w - w2)
        top_left_y = random.uniform(0, h - h2)

        new_imgs = [TF.resized_crop(img, top_left_y, top_left_x, h2, w2, (h, w)) for img in imgs]
        return new_imgs, intentions, labels, flip


class HorizontalFlip(object):
    def _flip_intention(self, intention):
        if intention == 'left':
            return 'right'
        elif intention == 'right':
            return 'left'
        elif intention == 'forward':
            return intention
        else:
            raise NotImplementedError(f"unknown intention {intention}")

    def _flip_angle(self, angle):
        return -1 * angle

    def __call__(self, imgs, intentions, labels, flip):
        if flip:
            imgs = [TF.hflip(img) for img in imgs]
            intentions = [self._flip_intention(intention) for intention in intentions]  # flip intention
            labels = [[label[0], self._flip_angle(label[1])] for label in labels]  # flip angle
        return imgs, intentions, labels, flip


class Normalize(object):
    def __init__(self, mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]):
        self.mean = mean
        self.std = std

    def __call__(self, imgs, intentions, labels, flip):
        new_imgs = [TF.normalize(img, self.mean, self.std) for img in imgs]
        return new_imgs, intentions, labels, flip


class ColorJitter(object):
    """
    Randomly change the brightness, contrast and saturation of an image.

    brightness (float or tuple of python:float (min, max)) – How much to jitter brightness.
    brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
    Should be non negative numbers.

    contrast (float or tuple of python:float (min, max)) – How much to jitter contrast.
    contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
    Should be non negative numbers.

    saturation (float or tuple of python:float (min, max)) – How much to jitter saturation.
    saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
    Should be non negative numbers.

    hue (float or tuple of python:float (min, max)) – How much to jitter hue.
    hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
    Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, differ_for_each_frame=False):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.differ_for_each_frame = differ_for_each_frame
        from torchvision.transforms import ColorJitter as RandColorJitter
        self.rand_jitter = RandColorJitter(brightness, contrast, saturation, hue)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, imgs, intentions, labels, flip):
        fn_idx = torch.randperm(4)
        if not self.differ_for_each_frame:
            for fn_id in fn_idx:
                if fn_id == 0 and self.brightness is not None:
                    brightness = self.brightness
                    brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                    imgs = [TF.adjust_brightness(img, brightness_factor) for img in imgs]

                if fn_id == 1 and self.contrast is not None:
                    contrast = self.contrast
                    contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                    imgs = [TF.adjust_contrast(img, contrast_factor) for img in imgs]

                if fn_id == 2 and self.saturation is not None:
                    saturation = self.saturation
                    saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                    imgs = [TF.adjust_saturation(img, saturation_factor) for img in imgs]

                if fn_id == 3 and self.hue is not None:
                    hue = self.hue
                    hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                    imgs = [TF.adjust_hue(img, hue_factor) for img in imgs]
        else:
            imgs = [self.rand_jitter(img) for img in imgs]

        return imgs, intentions, labels, flip


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, imgs, intentions, labels, flip):
        new_imgs = [TF.to_tensor(img) for img in imgs]
        return new_imgs, intentions, labels, flip


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs, intentions, labels, flip):
        new_imgs = [TF.resize(img, self.size, interpolation=Image.BICUBIC) for img in imgs]
        return new_imgs, intentions, labels, flip


class Grayscale(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, imgs, intentions, labels, flip):
        if random.random() < self.p:
            imgs = [TF.to_grayscale(img, num_output_channels=3) for img in imgs]
        return imgs, intentions, labels, flip
