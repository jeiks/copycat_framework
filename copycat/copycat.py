from .model import Model
from .utils import train, test, save_model
import torchvision.transforms as transforms
from .data import Dataset
from .config import Config

class Copycat(object):
    def __init__(self, problem, use_oracle_arch=True,
                 save_filename=None, resume_filename=None,
                 dataset_root='', db_name_train='npd', db_name_test='test',
                 dont_load_datasets=False, outputs=None,
                 config_fn=None):
        if save_filename is None: print(f'"save_filename" is None. You need to set it before training the Copycat.')
        self.problem = problem
        self.save_filename = save_filename
        self.config_fn = config_fn
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
            self.db_name_train = db_name_train
            self.db_name_test = db_name_test
            self.dataset = Dataset(problem=self.problem, normalize=False, root=dataset_root, config_fn=self.config_fn)
            #checking if dataset is using normalize:
            self.__check_dataset(self.dataset)
            self.outputs = len(self.dataset.classes)
        #loading the oracle model:
        self.model = Model(self.outputs, pretrained=True, oracle_arch=use_oracle_arch, state_dict=resume_filename, save_filename=save_filename)
        self.save_filename = save_filename
    
    def __load_config(self):
        default = {'max_epochs':10, 'batch_size':16, 'lr':1e-4, 'gamma':0.3, 'criterion':'CrossEntropyLoss',
                   'optimizer':'SGD', 'weight_decay': True, 'validation_step':0, 'save_snapshot':False, 'balance_dataset':True}
        self.config = Config(self.config_fn).get_problem_options(self.problem, 'copycat')
        for k, v in default.items():
            if k not in self.config:
                self.config[k] = v

    def __get_opt(self, name, value=None):
        return self.config[name] if value is None and name in self.config else value
    
    def __check_dataset(self, dataset):
        assert not dataset.has_transformer(transforms.Normalize), \
            "You cannot include transforms.Normalize in copycat's datasets. It is used only in Oracle."
        for db in self.db_name_train, self.db_name_test:
            assert getattr(self.dataset, db) is not None, f'"{db}" does not exists in dataset'

    def train(self, max_epochs=None, batch_size=None, criterion=None, optimizer=None, weight_decay=None,
              lr=None, gamma=None, validation_step=None, save_snapshot=None, balance_dataset=None):
        assert self.save_filename is not None, 'The param "save_filename" must be set before training.'
        assert self.dataset is not None, 'datasets not provided'
        #ps: the validation here is not used in training process. It is only used to the user see the model efficiency during training
        self.model = train(self.model, self.dataset,
                           db_name          = self.db_name_train,
                           db_name_validate = self.db_name_test,
                           snapshot_prefix  = self.save_filename if self.__get_opt('save_snapshot') else None,
                           max_epochs       = self.__get_opt('max_epochs', max_epochs),
                           batch_size       = self.__get_opt('batch_size', batch_size),
                           criterion        = self.__get_opt('criterion',criterion),
                           optimizer        = self.__get_opt('optimizer',optimizer),
                           lr               = self.__get_opt('lr',lr),
                           gamma            = self.__get_opt('gamma',gamma),
                           validation_step    = self.__get_opt('validation_step',validation_step),
                           weight_decay     = self.__get_opt('weight_decay', weight_decay),
                           balance_dataset  = self.__get_opt('balance_dataset', balance_dataset))
        print(f'Saving model in: "{self.save_filename}"')
        save_model(self.model, self.save_filename)

    def test(self, metric='report'):
        assert self.dataset is not None, 'datasets not provided'
        return test(self.model, self.dataset, db_name=self.db_name_test, metric=metric)
    
    def label_training_dataset(self, oracle, hard_labels=True):
        assert self.dataset is not None, 'datasets not provided'
        new_labels = {k:v for k,v in oracle.label(self.db_name_train, hard_labels=hard_labels)}
        self.dataset.update_dataset(self.db_name_train, new_labels)
        self.dataset.save_dataset(self.db_name_train)
