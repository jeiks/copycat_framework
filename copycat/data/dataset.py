#torch
from pyparsing import col
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
#system
from os import path as os_path
from hashlib import sha256 as hashlib_sha256
#utils
from .image_list import ImageList, OpenImage
from ..config import Config

default_collate = torch.utils.data._utils.collate.default_collate

class Transform:
    '''
        Class to automate the process create the Oracle/Copycat transformer and also to maintain a default
        It generates a transformer with:
            * for Copycat: Resize the image to @img_size and provide a Tensor. It does NOT use normalization, mean, and std.
            * for Oracle: Resize the image to @img_size and provide a Tensor. But it includes a normalization using mean and std.
                          The mean and std must be provided by (priority order) args @mean and @std, or in the config file.
        It also provides a method to check if "a specific transformer" is present in the transformer list.
        Args:
            img_size: image size
            include_normalize: include a normalization transformer using mean and std (provided as args or in config file)
            mean: mean to use in normalization transformer. It also override the "mean" defined in the config file.
            std: standard deviation to use in normalization transformer. It also override the "std" defined in the config file.
            config_std: provide a name to replace the default config file (copycat/config.yaml)
    '''
    def __init__(self, problem, img_size=(128,128), include_normalize=False, mean=None, std=None, config_fn=None):
        self.problem = problem
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.config_fn = config_fn
        self.config = Config(self.config_fn)
        self.transform = transforms.Compose([
            transforms.Resize( self.img_size ),
            transforms.ToTensor(),
        ])
        if include_normalize:
            m, s = self.get_mean_std()
            assert m is not None and s is not None, f"mean ({m}) and std ({s}) are not defined."
            self.transform.transforms.append(
                transforms.Normalize(m, s)
            )
    
    def get_mean_std(self):
        '''
            Method to return the mean and std, checking the arguments and also the config file.
            Note: The arguments has higher priority.
        '''
        if self.mean is None or self.std is None:
            mean, std = self.config.get_mean_std(self.problem)
            self.mean = mean if self.mean is None else self.mean
            self.std  = std  if self.std  is None else self.std
        return self.mean, self.std

    def has_transformer(self, trans):
        '''
            Method to check if transformer "trans" is in the transformer list (self.transform)
            Args:
                trans: PyTorch's transformer
        '''
        if self.transform.__class__ == trans:
            return True
        elif self.transform.__class__ == transforms.Compose:
            return trans in [x.__class__ for x in self.transform.transforms]
        return False
    
    def __repr__(self):
        return str(self.transform)
    
    def __call__(self, *args, **kwords):
        return self.transform(*args)

class Dataset:
    '''Copycat Dataset
    @param root: root path to find the images image list filenames and also their images
    @param img_size: tuple, size to return the images. Ex.: img_size=(224,224)
    @param normalize: boolean, return the images normalized.
     -> when @param mean and @param std are not provided as parameters, their values are loaded from the configuration file
     -> when only @param mean or @param std is provided, the other value is loaded from the configuration file
    @param problem: string, name of the problem to load its configurations 
    @param classes: list, name of problem classes
     -> them @param classes is not provided, the values can be loaded from the configuration file
    @param data_filenames: dict, database name and its filename (you can also add its root path)
    NOTE: If you have a different root for each database, you can also provide them in @param data_filenames.
          Example:
            self.filenames = {'train': ['NAME_od.txt', '/home/data/NAME/OD'],
                               'test' : ['NAME_td.txt', '/home/data/NAME/TD'],
                               'pd'   : 'NAME_pd.txt',
                               'npd'  : 'NAME_npd.txt'}
            The "root" for the filenames provided in NAME_pd.txt and NAME_npd.txt will follow
            the class constructor parameter "root".
    @param config_fn: string, filename of the configuration file (use only to change default values)
    '''
    filenames = {
                  'od'   : '', #original data
                  'test' : '', #test data
                  'pd'   : '', #problem domain data
                  'pd_sl': '', #problem domain data with stolen labels
                  'npd'  : ''  #non-problem domain data
                  }
    classes = []
    def __init__(self, root='', color=True, img_size=(128,128), normalize=None, mean=None, std=None, problem=None, classes=None, data_filenames=None, config_fn=None):
        self.problem = problem
        self.config_fn = config_fn
        self.config = Config(self.config_fn)
        self.classes = classes
        self.filenames = data_filenames
        self.include_normalize = normalize
        self.mean = mean
        self.std = std
        self.root = root
        self.img_size = img_size
        self.color = True if color is None else color
        self.config_datasets()
        self.config_classes()
    
    def config_datasets(self):
        if self.filenames is None:
            assert self.problem is not None, 'If you do not provide @data_filenames, you must write them in the configuration file and provide the @problem name'
            self.filenames = self.config.get_dataset_options(self.problem)
            assert len(self.filenames) > 0, \
                'You must provide the dataset filenames in the configuration file.\n'\
                'Example:\n'\
                '        data:\n            datasets:\n'\
                f'            train = data/train_data.txt.bz2'
        self.__check_filenames()
        self.set_root(self.root, init=True)
        self.transform = self.new_transform()
        for db_name in self.filenames:
            self.add_db(db_name)

    def add_db(self, db_name, db_filename=None):
        if db_name not in self.filenames:
            assert db_filename is not None, f"Please inform the 'filename' which contains data for 'db_name'"
            self.filenames[db_name] = db_filename
        db_fn   = self.__get_db_fn(db_name)
        db_root = self.__get_db_root(db_name)
        #print(f'Loading {db_name}({db_fn})...')
        setattr(self, db_name, ImageList( filename=db_fn, root=db_root,
                                          color=self.color, transform=self.transform,
                                          return_filename=False ))

    def config_classes(self):
        if self.classes is None:
            self.classes = self.config.get_classes(self.problem)
            if len(self.classes) == 0:
                try:
                    self.classes = list(getattr(self, list(self.filenames.items())[0][0]).categories)
                except:
                    print("Could not get number of classes for this problem")
                    pass

    def new_transform(self):
        return Transform(problem=self.problem, img_size=self.img_size,
                         include_normalize=self.include_normalize, mean=self.mean, std=self.std, config_fn=self.config_fn)

    def has_transformer(self, trans):
        return self.transform.has_transformer(trans)

    def return_fn(self, ret_fn=True):
        for db_name in self.filenames:
            db = getattr(self, db_name)
            db.ret_fn(ret_fn)

    def set_root(self, root=None, init=False, db_name=None):
        if db_name is not None:
            self.get_dataset(db_name).set_root(root)
            if len(self.filenames[db_name]) == 2:
                self.filenames[db_name] = [root, self.filenames[db_name][-1]]
            else:
                self.filenames[db_name] = [root, self.filenames[db_name]]
        else:
            if root is not None and init is False:
                for db_name in self.filenames:
                    db_root = self.__get_db_root(db_name)
                    if db_root != root:
                        getattr(self, db_name).set_root(root)
            self.root = root

    def get_db_names(self):
        return self.filenames.keys()

    def get_dataset(self, db_name):
        assert db_name in self.filenames, f'First you should set "{db_name}" in self.filenames'
        return getattr(self, db_name)

    def get_loader(self, db_name):
        assert db_name in self.filenames, f'First you should set "{db_name}" in self.filenames'
        return getattr(self, f'{db_name}_loader')

    def __get_db_fn(self, db_name):
        assert db_name in self.filenames, f'First you should set "{db_name}" in self.filenames'
        if type(self.filenames[db_name]) is list:
            root = self.filenames[db_name][1]
            fn1 = self.filenames[db_name][0]
            fn2 = os_path.join(root, fn1)
            if   os_path.isfile(fn2): return fn2
            elif os_path.isfile(fn1): return fn1
        elif type(self.filenames[db_name]) is str:
            fn = self.filenames[db_name]
            if os_path.isfile(fn): return fn
        raise ValueError(f'Could not find the "{db_name}" dataset file: "{self.filenames[db_name]}". Please, check it.')
    
    def __get_db_root(self, db_name):
        assert db_name in self.filenames, f'First you should set "{db_name}" in self.filenames'
        if type(self.filenames[db_name]) is list:
            for root in self.filenames[db_name]:
                if os_path.isdir(root): return root
        return self.root

    def __loader(self, db_name, shuffle=True, batch_size=1, num_workers=2):
        db = getattr(self, db_name)
        return DataLoader(db, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    def __dynamic_loader(self, method, *args, shuffle=True, batch_size=1, num_workers=2):
        db_name = method[::-1].replace('_loader'[::-1], '', 1)[::-1]
        assert db_name in self.filenames, f'Dataset {db_name} does not exist.'
        return self.__loader(db_name, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

    def __dynamic_set_root(self, method, *args, root=''):
        db_name = method.replace('set_root_', '', 1)
        assert db_name in self.filenames, f'Dataset {db_name} does not exist.'
        if type(self.filenames[db_name]) == str:
            self.filenames[db_name] = [self.filenames[db_name], root]
        else:
            self.filenames[db_name][-1] = root
        getattr(self, db_name).set_root(root)

    def __getattr__(self, method):
        def loader(shuffle=True, batch_size=1, num_workers=2):
            return self.__dynamic_loader(method, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers) 
        def set_root(root=''):
            return self.__dynamic_set_root(method, root=root)
        if   method.endswith('loader'): return loader
        elif method.startswith('set_root_'): return set_root
        else: raise ValueError(f'The method "{method}" does not exist.')

    def __check_filenames(self):
        for k in self.filenames.keys():
            fn = self.__get_db_fn(k)
    
    def update_dataset(self, db_name, labels, force_logits=False, force_labels=False):
        #fn = self.__get_db_fn(db_name)
        dataset = self.get_dataset(db_name)
        dataset.update_labels(labels, force_logits, force_labels)
    
    def save_dataset(self, db_name, filename=None):
        #fn = self.__get_db_fn(db_name)
        dataset = self.get_dataset(db_name)
        dataset.save(filename)

class Common(Dataset):
    '''
    Common dataset.
    @param filenames: dataset information. Example: filenames = {'train': 'train_file_list.txt'}
    '''
    def __init__(self, data_filenames=None, classes=None, root='', img_size=(128,128), normalize=False, mean=None, std=None, config_fn=None, problem=None):
        '''
        data_filenames: {'train': 'train.txt', 'test': 'test.txt.bz2'}
          \-> OR write it in configuration file [problem.datasets] and provide the problem name at @problem
        classes = ['one', 'two']
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        '''
        if normalize:
            assert mean is not None and std is not None, 'To normalize the dataset, you must provide "mean" and "std"'
            self.mean = mean
            self.std  = std
        
        self.classes = classes
        super().__init__(root=root, img_size=img_size, normalize=normalize, data_filenames=data_filenames, config_fn=config_fn, problem=problem)
        
    def get_mean_std(self):
        return self.mean, self.std