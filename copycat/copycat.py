from sys import stderr
from .model import Model
from .utils import train, test, save_model
import torchvision.transforms as transforms
from .data import Dataset
from .config import Config

class Copycat(object):
    """
    This class is responsible to train a Copycat model.
    """
    def __init__(self, problem, model_arch=None,
                 save_filename=None, resume_filename=None,
                 dataset_root='', db_name_train=None, db_name_test=None,
                 new_dataset=None,
                 dont_load_datasets=False, outputs=None, config_fn=None, finetune=False, data_filenames=None):
        if save_filename is None: print(f'"save_filename" is None. You need to set it before training the Copycat.')
        self.problem = problem
        self.save_filename = save_filename
        self.config_fn = config_fn
        self.finetune = finetune
        self.__load_config()
        #Dataset:
        if dont_load_datasets:
            outputs = self.__get_opt('outputs', outputs)
            if outputs is None:
                aux = self.__get_opt('classes')
                outputs = len(aux) if aux is not None else None
            assert outputs is not None and outputs != 0, "Please specify (as parameter or in configutation file) the Oracle number of outputs."
            self.outputs = outputs
            self.dataset = None
        else:
            self.db_name_train = self.__get_opt('db_train', db_name_train)
            self.db_name_test = self.__get_opt('db_test' , db_name_test)
            if new_dataset is None:
                self.dataset = Dataset(problem=self.problem, color=self.__get_opt('color', None), normalize=False, root=dataset_root, config_fn=self.config_fn, data_filenames=data_filenames)
            else:
                self.dataset = new_dataset
            #checking if dataset is using normalize:
            self.__check_dataset(self.dataset)
            self.outputs = len(self.dataset.classes)
        #loading the oracle model:
        self.model = Model(self.outputs, name='Finetune' if finetune else 'Copycat', pretrained=True, model_arch=model_arch, state_dict=resume_filename, save_filename=save_filename)
        self.save_filename = save_filename
    
    def __load_config(self):
        default = {'max_epochs':10, 'batch_size':16, 'lr':1e-4 * (1e-2 if self.finetune else 1), 'gamma':0.3, 'criterion':'CrossEntropyLoss',
                   'optimizer':'SGD', 'weight_decay': True, 'validation_step':0, 'save_snapshot':False, 'balance_dataset':True, 'color': True}
        self.config = Config(self.config_fn).get_problem_options(self.problem, 'finetune' if self.finetune else 'copycat')
        for k, v in default.items():
            if k not in self.config:
                self.config[k] = v

    def __get_opt(self, name, value=None):
        return self.config[name] if value is None and name in self.config else value
      
    def __check_dataset(self, dataset):
        assert not dataset.has_transformer(transforms.Normalize), \
            "You cannot include transforms.Normalize in copycat's datasets. It is used only in Oracle."
        if self.db_name_train is None:
            print('WARNING: train dataset is None', file=stderr)
        else:
            assert getattr(self.dataset, self.db_name_train) is not None, f'Train dataset"{self.db_name_train}" does not exists in configuration file'
        if self.db_name_test is None:
            print('WARNING: test dataset is None', file=stderr)
        else:
            assert getattr(self.dataset, self.db_name_test) is not None, f'Test dataset "{self.db_name_test}" does not exists in configuration file'

    def train(self, max_epochs=None, batch_size=None, criterion=None, optimizer=None, weight_decay=None,
              lr=None, gamma=None, validation_step=None, save_snapshot=None, balance_dataset=None):
        assert self.save_filename is not None, 'The param "save_filename" must be set before training.'
        assert self.dataset is not None, 'Datasets not provided'
        assert getattr(self.dataset, self.db_name_train) and getattr(self.dataset, self.db_name_test), 'Datasets not provided'
        #ps: the validation (db_name_validate) here is not used during the training process.
        #    It is only used to show the model performance during training
        self.model = train(self.model, self.dataset,
                           db_name          = self.db_name_train,
                           db_name_validate = self.db_name_test,
                           snapshot_prefix  = self.save_filename if self.__get_opt('save_snapshot', save_snapshot) else None,
                           max_epochs       = self.__get_opt('max_epochs', max_epochs),
                           batch_size       = self.__get_opt('batch_size', batch_size),
                           criterion        = self.__get_opt('criterion',criterion),
                           optimizer        = self.__get_opt('optimizer',optimizer),
                           lr               = self.__get_opt('lr',lr),
                           gamma            = self.__get_opt('gamma',gamma),
                           validation_step  = self.__get_opt('validation_step',validation_step),
                           weight_decay     = self.__get_opt('weight_decay', weight_decay),
                           balance_dataset  = self.__get_opt('balance_dataset', balance_dataset))
        print(f'Saving Copycat model in: "{self.save_filename}"')
        save_model(self.model, self.save_filename)
    
    def label_training_dataset(self, oracle, hard_labels=True):
        assert self.dataset is not None, 'datasets not provided'
        new_labels = {k:v for k,v in oracle.label(self.db_name_train, hard_labels=hard_labels)}
        self.dataset.update_dataset(self.db_name_train, new_labels)
        self.dataset.save_dataset(self.db_name_train)
    
    def test(self, metric='report'):
        assert self.dataset is not None, 'datasets not provided'
        return test(self.model, self.dataset, db_name=self.db_name_test, metric=metric)
    
    def cpu(self):
        self.model.cpu()
    
    def cuda(self):
        self.model.cuda()