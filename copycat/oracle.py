from .model import Model
from .utils import train, test, label, save_model, label_image, calculate_mean_std
import torchvision.transforms as transforms
from .data import Dataset, Transform, OpenImage
from .config import Config

class Oracle(object):
    def __init__(self, problem, model_arch=None, save_filename=None, resume_filename=None,
                 dataset_root='', db_name_train=None, db_name_test=None,
                 new_dataset=None,
                 dont_load_datasets=False, outputs=None,
                 config_fn=None, baseline=False, data_filenames=None): #'baseline' is the second baseline besides 'Oracle'
        if save_filename is None: print(f'"save_filename" is None. You need to set it before training the Oracle.')
        self.problem = problem
        self.save_filename = save_filename
        self.config_fn = config_fn
        self.baseline = baseline
        self.__load_config()
        #Dataset:
        if dont_load_datasets:
            outputs = self.__get_opt('outputs', outputs)
            if outputs is None:
                aux = self.__get_opt('classes')
                outputs = len(aux) if aux is not None else None
            if type(model_arch) == str:
                assert outputs is not None and outputs != 0, "Please specify (as parameter or in configutation file) the Oracle number of outputs."
            self.outputs = outputs
            self.dataset = None
        else:
            self.db_name_train = self.__get_opt('db_train', db_name_train)
            self.db_name_test  = self.__get_opt('db_test', db_name_test)
            if new_dataset is None:
                self.dataset = Dataset(problem=self.problem, color=self.__get_opt('color', None), normalize=True, root=dataset_root, config_fn=self.config_fn, data_filenames=data_filenames)
            else:
                self.dataset = new_dataset
            #checking if dataset is using normalize:
            self.__check_dataset(self.dataset)
            self.outputs = len(self.dataset.classes)
        #loading the oracle model:
        self.model = Model(self.outputs, name='Baseline' if self.baseline else 'Oracle', pretrained=True, model_arch=model_arch, state_dict=resume_filename, save_filename=save_filename)
    
    def __load_config(self):
        default = {'max_epochs':10, 'batch_size':16, 'lr':1e-4, 'gamma':0.1, 'criterion':'CrossEntropyLoss',
                   'optimizer':'SGD', 'weight_decay': True, 'validation_step':0, 'save_snapshot':False, 'color': True}
        self.config = Config(self.config_fn).get_problem_options(self.problem, 'baseline' if self.baseline else 'oracle')
        for k, v in default.items():
            if k not in self.config:
                self.config[k] = v
        #print(self.config)

    def __get_opt(self, name, value=None):
        return self.config[name] if value is None and name in self.config else value

    def __check_dataset(self, dataset):
        assert dataset.has_transformer(transforms.Normalize) or self.baseline, \
            "You must include transforms.Normalize in oracle's datasets"
        for db in self.db_name_train, self.db_name_test:
            assert getattr(self.dataset, db) is not None, f'"{db}" does not exists in dataset'

    def train(self, max_epochs=None, batch_size=None, lr=None, gamma=None, criterion=None,
              optimizer=None, weight_decay=None, validation_step=None, save_snapshot=None):
        assert self.save_filename is not None, 'The param "save_filename" must be set before training.'
        assert self.dataset is not None, 'datasets not provided'
        #ps: the validation here is not used in training process. It is only used to the user see the model efficiency during training
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
                           validation_step    = self.__get_opt('validation_step',validation_step),
                           weight_decay     = self.__get_opt('weight_decay', weight_decay))
        print(f'Saving model in: "{self.save_filename}"')
        save_model(self.model, self.save_filename)

    def test(self, metric='report'):
        assert self.dataset is not None, 'datasets not provided'
        return test(self.model, self.dataset, db_name=self.db_name_test, metric=metric)
    
    def label(self, db_name='npd', hard_labels=True):
        assert self.dataset is not None, 'datasets not provided'
        return label(self.model, self.dataset, db_name=db_name, hard_labels=hard_labels)
    
    def query(self, db_name='npd', hard_labels=True):
        return self.label(db_name=db_name, hard_labels=hard_labels)
    
    def query_hard_label(self, db_name='npd'):
        return self.label(db_name=db_name, hard_labels=True)

    def query_soft_label(self, db_name='npd'):
        return self.label(db_name=db_name, hard_labels=False)
    
    def query_single_image(self, img, hard_labels=True):
        trans = Transform(problem=self.problem, include_normalize=True)
        img = trans(img)
        return label_image(model=self.model, image=img, hard_labels=hard_labels)
    
    def cpu(self):
        self.model.cpu()
    
    def cuda(self):
        self.model.cuda()
