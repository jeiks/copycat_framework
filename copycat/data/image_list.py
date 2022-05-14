import warnings
import torch.utils.data as data
import torchvision.transforms as transforms
import os.path
import numpy as np
import torch
import cv2
#from PIL import Image
from torchvision.transforms import ToPILImage, Grayscale
from tqdm import tqdm
from sys import stderr
import torchvision

from io import StringIO
import bz2

class OpenImage:
    @classmethod
    def color(cls, img_fn):
        img = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        return img # np.ndarray
    
    @classmethod
    def grayscale(cls, img_fn):
        img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
        return img # np.ndarray

    @classmethod
    def color_rgb(cls, img_fn):
        img = cls.color(img_fn)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # np.ndarray
    
    @classmethod
    def color_bgr(cls, img_fn):
        return cls.color(img_fn) # np.ndarray

    @classmethod
    def color_pil(cls, img_fn):
        img = cls.color(img_fn)
        return ToPILImage()(img) # PIL.Image.Image

    @classmethod
    def grayscale_pil(cls, img_fn, num_output_channels=3):
        img = cls.grayscale(img_fn)
        # PIL.Image.Image, but with 3 channels to work with
        # VGG and AlexNet without changing the architectures structure
        return Grayscale(num_output_channels=num_output_channels)(ToPILImage()(img))

    @classmethod
    def color_rgb_pil(cls, img_fn):
        img = cls.color_rgb(img_fn)
        return ToPILImage()(img) # PIL.Image.Image

    @classmethod
    def color_bgr_pil(cls, img_fn):
        img = cls.color_bgr(img_fn)
        return ToPILImage()(img) # PIL.Image.Image

class ImageList(data.Dataset):
    '''
    Image List Dataset
    Args:
        filename (string): Image List Filename
        color (optional): Open images as RGB instead of Grayscale
        root (string, optional): Root directory of image files
        transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): same as transform but applied only
            on target(labels, outputs)
        return_filename (boolean, optional): In addition to the image and label, it
            also returns the image filename: (image, label, filename)
    '''
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, filename,
                 root=None,
                 color=True,
                 num_output_channels=3,
                 transform=None, target_transform=None,
                 return_filename=False,
                 balance_dataset=False):
        self.filename = filename
        self.color = color
        self.num_output_channels = num_output_channels
        self.set_root(root)
        self.transform = transform
        self.target_transform = target_transform
        self.ret_fn(return_filename)
        self.data, self.targets, self.logits, self.void_labels = self.__process_data()
        self.categories = np.unique(self.targets).astype(np.uint8)
        self.balance_dataset(balance_dataset)
        self.balanced_indexes = None
        self.__select_open_image()

    def __select_open_image(self):
        if self.color:
            if self.transform is None:
                self.open_image = lambda x: torch.tensor(OpenImage.color_rgb(x))
            else:
                self.open_image = lambda x: self.transform(OpenImage.color_rgb_pil(x))
        else:
            if self.transform is None:
                self.open_image = lambda x: torch.tensor(OpenImage.grayscale(x))
            else:
                self.open_image = lambda x: self.transform(OpenImage.grayscale_pil(x, num_output_channels=self.num_output_channels))

    def __open(self):
        assert type(self.filename) is str, f'@param filename "{self.filename}" must be a string (.txt or .bz2).'
        if self.filename.endswith('.bz2'):
            file_bz2 = bz2.open(self.filename, mode='rt')
            file_contents = StringIO(file_bz2.read())
        else:
            file_contents = StringIO(open(self.filename).read())
        return file_contents

    def __process_data(self):
        file_contents = self.__open()
        logits = False
        void_labels = False
        number_of_columns = len(file_contents.readline().split())
        file_contents.seek(0)
        if number_of_columns > 2:
            logits = True
            dtype=['S255']+[np.float32 for _ in range(number_of_columns-1)]
            usecols = tuple(range(len(dtype)))
            contents = np.genfromtxt(file_contents, dtype=dtype, usecols=usecols)
            data = np.array(contents['f0'], dtype=str)
            targets = np.array( [ contents[f'f{x}'] for x in usecols[1:] ]).T
        elif number_of_columns > 1:
            contents = np.genfromtxt(file_contents, dtype=['S255', int], usecols=(0,1))
            data = np.array(contents['f0'], dtype=str)
            targets = np.array(contents['f1'], dtype=np.uint8)
        else:
            contents = np.atleast_1d( np.genfromtxt(file_contents, dtype=['S255'],usecols=0) )
            data = np.array(contents, dtype=str)
            targets = np.zeros(len(data), dtype=np.uint8)
            void_labels = True
        
        return data, targets, logits, void_labels

    # Replace the method name seems faster, but also needs performance tests
    def __getitem__(self, index):
        return self.getitem(index)

    def __getitem__with_filename__(self, index):
        return self.__getitem__aux__(index)

    def __getitem_simple__(self, index):
        return self.__getitem__aux__(index)[:-1]

    def __getitem__aux__(self, index):
        if self.logits:
            img_fn, target = self.data[index], self.targets[index]
        elif self.is_balanced():
            img_fn, target = self.data[self.balanced_indexes[index]], int(self.targets[self.balanced_indexes[index]])
        else:
            img_fn, target = self.data[index], int(self.targets[index])
        
        if self.root != '':
            img_fn = os.path.join(self.root, img_fn)
        
        try:
            img = self.open_image(img_fn)
            #if self.transform is None:
            #    img = transforms.ToTensor()(np.array(img))
            #else:
            #    img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target, img_fn
        except:
            print(f'There was some error opening "{img_fn}"')

    def balance_dataset(self, balance):
        if balance and not self.is_balanced():
            assert self.logits == False, '"balance_dataset" with logits is not implemented yet.'
            total_len = self.__len__()
            n_per_class = {ii:np.where(self.targets == ii)[0] for ii in self.categories}
            desired_n_per_class = total_len // len(self.categories)
            indexes = []
            for i in self.categories:
                p = np.random.permutation(len(n_per_class[i]))
                permut = n_per_class[i][p]
                if len(permut) < desired_n_per_class:
                    permut = permut.repeat(desired_n_per_class//len(permut)+1)
                permut = permut[:desired_n_per_class]
                indexes += permut.tolist()
            self.balanced_indexes = np.array(indexes)
        
    def is_balanced(self):
        return self.balanced_indexes is not None

    def __len__(self):
        if self.is_balanced():
            return len(self.balanced_indexes)
        else:
            return len(self.targets)

    def __len_per_cat__(self):
        if self.is_balanced():
            return sum(self.targets[self.balanced_indexes] == self.categories[0])
        else:
            return [len(np.where(self.targets == ii)[0]) for ii in self.categories]

    def __repr__(self):
        def get_trans_repr(t, ident_size):
            return t.__repr__().replace("\n", "\n"+" "*ident_size)
        fmt_str  = f'Dataset {self.__class__.__name__}{" (Balanced)" if self.is_balanced() else ""}:\n'
        fmt_str += f'    Number of datapoints: {self.__len__()}\n'
        if self.void_labels:
            fmt_str += f'     * Void labels\n'
        elif self.is_balanced():
            fmt_str += f'     * Samples per class: {self.__len_per_cat__()}\n'
        else:
            tmp = self.__len_per_cat__()
            tmp = [f'     * Class {idx}: {tmp[ii]:-6d} samples\n' for ii, idx in enumerate(self.categories)]
            fmt_str += f'{"".join(tmp)}'
        tmp      =  '    Transforms (if any): '
        fmt_str += f'{tmp}{get_trans_repr(self.transform, len(tmp))}\n' 
        tmp      =  '    Target Transforms (if any): '
        fmt_str += f'{tmp}{get_trans_repr(self.target_transform, len(tmp))}'
        return fmt_str

    def has_transform(self, trans):
        if self.transform.__class__ == trans:
            return True
        elif self.transform.__class__ == transforms.Compose:
            return trans in [x.__class__ for x in self.transform.transforms]
        return False

    def ret_fn(self, return_fn=True):
        self.getitem = self.__getitem__with_filename__ if return_fn else self.__getitem_simple__
    
    def set_root(self, root=None):
        self.root = os.path.expanduser(root)
    
    def update_labels(self, labels, force_logits=False, force_labels=False):
        assert type(labels) == dict, '@param "labels" must be a dict of "filename": int(new_label) or "filename": [predictions,...]'
        assert not force_logits or not force_labels, 'you must select only "force_logits" or "force_labels"'
        if force_logits:
            n_items = len(self.targets)
            new_shape = len( list(labels.items())[0] )
            self.logits = True
            self.targets = np.zeros([n_items, new_shape], dtype=np.uint8)
        elif force_labels:
            n_items = len(self.targets)
            self.logits = False
            self.targets = np.zeros([n_items], dtype=np.uint8)

        for pos, fn in enumerate(self.data):
            fn = os.path.join(self.root, fn)
            if fn in labels:
                self.targets[pos] = labels[fn]

        self.categories = np.unique(self.targets).astype(np.uint8)
        self.void_labels = False
    
    def save(self, filename=None):
        fn = filename if filename is not None else self.filename
        if fn.endswith('.bz2'): fd = bz2.open(fn, mode='wt')
        else:                   fd = open(fn, mode='w')
        for pos in range(len(self.targets)):
            targets = int(self.targets[pos]) if not self.logits else ' '.join([f'{x:.6f}' for x in self.targets[pos]])
            fd.write(f'{self.data[pos]} {targets}\n')
        fd.close()