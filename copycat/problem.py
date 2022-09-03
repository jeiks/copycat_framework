from os import path
from .oracle import Oracle
from .copycat import Copycat
from .finetune import Finetune
from .utils import set_seeds
from time import asctime

class Problem(object):
    def __init__(self, problem, oracle_filename, copycat_filename, finetune_filename,
                 oracle_dataset_root='', copycat_dataset_root='', finetune_dataset_root='',
                 oracle_resume_filename=None, copycat_resume_filename=None, finetune_resume_filename=None,
                 use_same_arch=True, config_fn=None, seed=None):
        # Oracle:
        self.oracle = None
        self.oracle_filename = oracle_filename
        self.oracle_resume_fn = oracle_resume_filename
        self.oracle_dataset_root = oracle_dataset_root
        # Copycat:
        self.copycat = None
        self.copycat_filename = copycat_filename
        self.copycat_resume_fn = copycat_resume_filename
        self.copycat_dataset_root = copycat_dataset_root
        # Finetune:
        self.finetune = None
        self.finetune_filename = finetune_filename
        self.finetune_resume_fn = finetune_resume_filename
        self.finetune_dataset_root = finetune_dataset_root
        # Common
        self.problem = problem
        self.use_same_arch = use_same_arch
        self.config_fn = config_fn
        self.results = {'copycat': None, 'oracle': None, 'finetune': None}
        #Seed:
        if seed is not None: set_seeds(seed)

    def load_oracle(self):
        '''
            Load model from disk or move it from CPU to CUDA
        '''
        if self.oracle is None:
            self.oracle = Oracle(problem=self.problem, save_filename=self.oracle_filename, resume_filename=self.oracle_resume_fn,
                                 dataset_root=self.oracle_dataset_root, config_fn=self.config_fn)
        else:
            self.oracle.cuda()

    def load_copycat(self):
        '''
            Load model from disk or move it from CPU to CUDA
        '''
        if self.copycat is None:
            self.copycat = Copycat(problem=self.problem, save_filename=self.copycat_filename, resume_filename=self.copycat_resume_fn,
                                   dataset_root=self.copycat_dataset_root, model_arch='vgg16' if self.use_same_arch else 'alexnet', config_fn=self.config_fn)
        else:
            self.copycat.cuda()

    def load_finetune(self):
        '''
            Load model from disk or move it from CPU to CUDA
            If resume_filename is provided, it will be used to resume the
            finetune model
        '''
        if self.finetune is None:
            if self.finetune_resume_fn:
                #resume an older finetune model (priority 1):
                resume_filename = self.finetune_resume_fn
            elif path.isfile('' if self.copycat_resume_fn is None else self.copycat_resume_fn):
                #resume a last copycat model and finetune it  (priority 2):
                resume_filename=self.copycat_resume_fn
            elif path.isfile('' if self.copycat_filename is None else self.copycat_filename):
                #load the last copycat model to finetune it  (priority 3):
                resume_filename = self.copycat_filename
            else:
                print('(Finetune) ALERT: The Copycat model must be trained or resumed before start the finetuning.')
                print('--------------->  GENERATING A MODEL FOR FINETUNE WITH RANDOM PARAMETERS...')
                resume_filename=None
            
            self.finetune = Finetune(problem=self.problem, save_filename=self.finetune_filename,
                                     resume_filename=resume_filename, dataset_root=self.finetune_dataset_root,
                                     model_arch='vgg16' if self.use_same_arch else 'alexnet', config_fn=self.config_fn)
        else:
            self.finetune.cuda()

    def unload_oracle(self):
        '''
            Move model from CUDA to CPU
        '''
        if self.oracle is not None:
            try: self.oracle.cpu()
            except: pass

    def unload_copycat(self):
        '''
            Move model from CUDA to CPU
        '''
        if self.copycat is not None:
            try: self.copycat.cpu()
            except: pass

    def unload_finetune(self):
        '''
            Move model from CUDA to CPU
        '''
        if self.finetune is not None:
            try: self.finetune.cpu()
            except: pass

    def train_oracle(self, max_epochs=None, batch_size=None, lr=None, gamma=None, criterion=None,
                     optimizer=None, weight_decay=None, validation_step=None, save_snapshot=None):
        self.load_oracle()
        self.oracle.train(max_epochs=max_epochs, batch_size=batch_size, lr=lr, gamma=gamma, criterion=criterion,
                          optimizer=optimizer, weight_decay=weight_decay, validation_step=validation_step, save_snapshot=save_snapshot)
        self.unload_oracle()

    def train_copycat(self, label_copycat_dataset=True, max_epochs=None, batch_size=None, criterion=None,
                      optimizer=None, weight_decay=None, lr=None, gamma=None, validation_step=None, save_snapshot=None, balance_dataset=None):
        if label_copycat_dataset: self.label_copycat_dataset()
        self.load_copycat()
        self.copycat.train(max_epochs=max_epochs, batch_size=batch_size, criterion=criterion, optimizer=optimizer, weight_decay=weight_decay,
                           lr=lr, gamma=gamma, validation_step=validation_step, save_snapshot=save_snapshot, balance_dataset=balance_dataset)
        self.unload_copycat()

    def finetune_copycat(self, label_finetune_dataset=True, max_epochs=None, batch_size=None, criterion=None,
                         optimizer=None, weight_decay=None, lr=None, gamma=None, validation_step=None, save_snapshot=None, balance_dataset=None):
        if label_finetune_dataset: self.label_finetune_dataset()
        self.load_finetune()
        self.finetune.train(max_epochs=max_epochs, batch_size=batch_size, criterion=criterion, optimizer=optimizer, weight_decay=weight_decay,
                              lr=lr, gamma=gamma, validation_step=validation_step, save_snapshot=save_snapshot, balance_dataset=balance_dataset)
        self.unload_finetune()

    def label_copycat_dataset(self):
        self.load_oracle()
        self.load_copycat()
        self.copycat.label_training_dataset(oracle=self.oracle, hard_labels=True)
        self.unload_oracle()
        self.unload_copycat()

    def label_finetune_dataset(self):
        self.load_oracle()
        self.load_finetune()
        self.finetune.label_training_dataset(oracle=self.oracle, hard_labels=True)
        self.unload_oracle()
        self.unload_finetune()

    def __report(self, obj, metric='report', show_datasets=True):
        name = str(obj.__class__.__weakref__).split("'")[-2]
        test_results = obj.test(metric=[metric, 'f1_score'])
        msg = f'{name} reports:\n'
        if show_datasets:
            db_train = getattr(obj.dataset, obj.db_name_train)
            db_test = getattr(obj.dataset, obj.db_name_test)
            msg += 'Training - '
            msg += str(db_train)+'\n'
            msg += 'Testing - '
            msg += str(db_test)+'\n'
        msg += '\nMetrics:\n'
        msg += test_results[0]
        print(msg)
        return test_results[1]
    
    def oracle_report(self, metric='report', show_datasets=True):
        self.load_oracle()
        self.results['oracle'] = self.__report(self.oracle, metric=metric, show_datasets=show_datasets)
    
    def copycat_report(self, metric='report', show_datasets=True):
        self.load_copycat()
        self.results['copycat'] = self.__report(self.copycat, metric=metric, show_datasets=show_datasets)

    def finetune_report(self, metric='report', show_datasets=True):
        self.load_finetune()
        self.results['finetune'] = self.__report(self.finetune, metric=metric, show_datasets=show_datasets)

    def run(self,
            #Common arguments:
            validation_step=None, save_snapshot=None,
            #Oracle:
            train_oracle=True,
            oracle_max_epochs=None,
            oracle_batch_size=None,
            oracle_lr=None,
            oracle_gamma=None,
            #Copycat:
            train_copycat=True,
            label_copycat_dataset=True,
            copycat_max_epochs=None,
            copycat_batch_size=None,
            copycat_lr=None,
            copycat_gamma=None,
            copycat_balance_dataset=None,
            #Copycat Finetune:
            finetune_copycat=True,
            label_finetune_dataset=True,
            finetune_max_epochs=None,
            finetune_batch_size=None,
            finetune_lr=None,
            finetune_gamma=None,
            finetune_balance_dataset=None):
        #Oracle
        if train_oracle:
            print(f'==> Training Oracle ({asctime()}):')
            self.train_oracle(validation_step=validation_step, save_snapshot=save_snapshot,
                              max_epochs=oracle_max_epochs, batch_size=oracle_batch_size, lr=oracle_lr, gamma=oracle_gamma)
            self.oracle_report()
            print('\n')
        
        #Copycat
        if train_copycat:
            print(f'==> Training Copycat ({asctime()}):')
            self.train_copycat(validation_step=validation_step, save_snapshot=save_snapshot, label_copycat_dataset=label_copycat_dataset,
                              max_epochs=copycat_max_epochs,  batch_size=copycat_batch_size, lr=copycat_lr, gamma=copycat_gamma, balance_dataset=copycat_balance_dataset)
            self.copycat_report()
            print('\n')

        #Finetune
        if finetune_copycat:
            print(f'==> Finetuning Copycat ({asctime()}):')
            self.finetune_copycat(validation_step=validation_step, save_snapshot=save_snapshot, label_finetune_dataset=label_finetune_dataset,
                                  max_epochs=finetune_max_epochs,  batch_size=finetune_batch_size, lr=finetune_lr, gamma=finetune_gamma, balance_dataset=finetune_balance_dataset)
            self.finetune_report()
            print('\n')
        
        print(f'Done ({asctime()})\n')
        self.print_performance()
        print(f'Finished ({asctime()})\n')
    
    def __save_results(self):
        if self.results['oracle'] is None:
            self.load_oracle()
            self.results['oracle'] = self.oracle.test(metric='f1_score')
            self.unload_oracle()
        if self.results['copycat'] is None:
            self.load_copycat()
            self.results['copycat'] = self.copycat.test(metric='f1_score')
            self.unload_copycat()
        if self.results['finetune'] is None:
            self.load_finetune()
            self.results['finetune'] = self.finetune.test(metric='f1_score')
            self.unload_finetune()

    def print_performance(self):
        self.__save_results()
        oracle_f1_macro = self.results['oracle'][1]
        copycat_f1_macro = self.results['copycat'][1]
        finetune_f1_macro = self.results['finetune'][1]
        msg =   'Accuracy (Macro F1-Score):\n'
        msg += f'  Oracle...: {oracle_f1_macro:.6f}\n'
        msg += f'  Copycat..: {copycat_f1_macro:.6f}\n'
        msg += f'  Finetune.: {finetune_f1_macro:.6f}\n'
        msg += f'Attack Performance\n'
        msg += f'  Copycat on Oracle...........: {copycat_f1_macro/oracle_f1_macro*100:.2f}%\n'
        msg += f'  Finetuned Copycat on Oracle.: {finetune_f1_macro/oracle_f1_macro*100:.2f}%'
        print(msg)
    
    def print_reports(self):
        print('Oracle:')
        self.oracle_report()
        print('\n')
        print('Copycat:')
        self.copycat_report()
        print('\n')
        print('Finetuned Copycat:')
        self.finetune_report()
        print('\n')
        self.print_performance()