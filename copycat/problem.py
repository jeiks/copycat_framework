from imp import load_source
from os import path
from .oracle import Oracle
from .copycat import Copycat
from .config import Config

class Problem(object):
    def __init__(self, problem, oracle_filename, copycat_filename, oracle_dataset_root='', copycat_dataset_root='', use_same_arch=True, config_fn=None):
        self.problem = problem
        self.oracle = None
        self.oracle_filename = oracle_filename
        self.oracle_dataset_root = oracle_dataset_root
        self.copycat = None
        self.copycat_filename = copycat_filename
        self.copycat_dataset_root = copycat_dataset_root
        self.use_same_arch = use_same_arch
        self.config_fn = config_fn

    def load_oracle(self):
        if self.oracle is None:
            self.oracle = Oracle(problem=self.problem,
                                 save_filename=self.oracle_filename,
                                 resume_filename=self.oracle_filename if path.isfile(self.oracle_filename) else None,
                                 dataset_root=self.oracle_dataset_root, config_fn=self.config_fn)

    def load_copycat(self):
        if self.copycat is None:
            self.copycat = Copycat(problem=self.problem,
                                   save_filename=self.copycat_filename,
                                   resume_filename=self.copycat_filename if path.isfile(self.copycat_filename) else None,
                                   dataset_root=self.copycat_dataset_root,
                                   use_oracle_arch=self.use_same_arch,
                                   config_fn=self.config_fn)

    def unload_oracle(self):
        if self.oracle is not None:
            try: self.oracle.cpu()
            except: pass
            del self.oracle
            self.oracle = None

    def unload_copycat(self):
        if self.copycat is not None:
            try: self.copycat.cpu()
            except: pass
            del self.copycat
            self.copycat = None

    def train_oracle(self, max_epochs=None, batch_size=None, lr=None, gamma=None, criterion=None,
                     optimizer=None, weight_decay=None, validation_step=None, save_snapshot=None):
        self.load_oracle()
        self.oracle.train(max_epochs=max_epochs, batch_size=batch_size, lr=lr, gamma=gamma, criterion=criterion,
                          optimizer=optimizer, weight_decay=weight_decay, validation_step=validation_step, save_snapshot=save_snapshot)

    def train_copycat(self, label_copycat_dataset=True, max_epochs=None, batch_size=None, criterion=None,
                      optimizer=None, weight_decay=None, lr=None, gamma=None, validation_step=None, save_snapshot=None, balance_dataset=None):
        if label_copycat_dataset: self.label_copycat_dataset()
        self.load_copycat()
        self.copycat.train(max_epochs=max_epochs, batch_size=batch_size, criterion=criterion, optimizer=optimizer, weight_decay=weight_decay,
                           lr=lr, gamma=gamma, validation_step=validation_step, save_snapshot=save_snapshot, balance_dataset=balance_dataset)

    def label_copycat_dataset(self):
        self.load_oracle()
        self.load_copycat()
        self.copycat.label_training_dataset(oracle=self.oracle, hard_labels=True)
        self.unload_oracle()
        self.unload_copycat()

    def __report(self, obj, metric='report', show_datasets=True):
        name = str(obj.__class__.__weakref__).split("'")[-2]
        msg = f'{name} reports:\n'
        if show_datasets:
            db_train = getattr(obj.dataset, obj.db_name_train)
            db_test = getattr(obj.dataset, obj.db_name_test)
            msg += 'Training - '
            msg += str(db_train)+'\n'
            msg += 'Testing - '
            msg += str(db_test)+'\n'
        msg += '\nMetrics:\n'
        msg += obj.test(metric)
        print(msg)
    
    def oracle_report(self, metric='report', show_datasets=True):
        self.load_oracle()
        self.__report(self.oracle, metric=metric, show_datasets=show_datasets)
    
    def copycat_report(self, metric='report', show_datasets=True):
        self.load_copycat()
        self.__report(self.copycat, metric=metric, show_datasets=show_datasets)

    def run(self, train_oracle=True, train_copycat=True, label_copycat_dataset=True,
            validation_step=None, save_snapshot=None, #for both
            #Oracle:
            oracle_max_epochs=None,
            oracle_batch_size=None,
            oracle_lr=None,
            oracle_gamma=None,
            #Copycat:
            copycat_max_epochs=None,
            copycat_batch_size=None,
            copycat_lr=None,
            copycat_gamma=None,
            copycat_balance_dataset=None):
        if train_oracle:
            print('Training Oracle:')
            self.train_oracle(validation_step=validation_step, save_snapshot=save_snapshot,
                              max_epochs=oracle_max_epochs, batch_size=oracle_batch_size, lr=oracle_lr, gamma=oracle_gamma)
            self.oracle_report()
            print('\n')
        if label_copycat_dataset:
            self.label_copycat_dataset()
        if train_copycat:
            print('Training Copycat:')
            self.train_copycat(validation_step=validation_step, save_snapshot=save_snapshot, label_copycat_dataset=False, #it is labeled in the last "if"
                              max_epochs=copycat_max_epochs,  batch_size=copycat_batch_size, lr=copycat_lr, gamma=copycat_gamma, balance_dataset=copycat_balance_dataset)
            self.copycat_report()
        self.print_performance()
        self.unload_oracle()
        self.unload_copycat()
    
    def print_performance(self):
        self.load_oracle()
        _, oracle_f1_macro = self.oracle.test(metric='f1_score')
        self.load_copycat()
        _, copycat_f1_macro = self.copycat.test(metric='f1_score')
        msg =   'Accuracy (Macro F1-Score):\n'
        msg += f'  Oracle:  {oracle_f1_macro:.6f}\n'
        msg += f'  Copycat: {copycat_f1_macro:.6f}\n'
        msg += f'Attack Performance: {copycat_f1_macro/oracle_f1_macro*100:.2f}%'
        print(msg)
    
    def print_reports(self):
        print('Oracle:')
        self.oracle_report()
        print('\n')
        print('Training Copycat:')
        self.copycat_report()
        print('\n')
        self.print_performance()