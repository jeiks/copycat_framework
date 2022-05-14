#!/usr/bin/env python

import signal
from sys import exit
import argparse

from copycat import Problem, Config
from torch import cuda

from copycat.utils import set_seeds
set_seeds(7)

class Options:
    def __init__(self,
                 #required parameters:
                 problem_name, oracle_filename, copycat_filename, finetune_filename,
                 train_oracle,
                 train_copycat, label_copycat_dataset,
                 train_finetune, label_finetune_dataset,
                 only_print_reports,
                 #general configutation:
                 config_file=None,
                 validation_step=None,
                 save_snapshot=None,
                 #oracle:
                 oracle_max_epochs=None, oracle_batch_size=None, oracle_lr=None,
                 oracle_gamma=None, oracle_dataset_root=None, oracle_resume_filename=None,
                 #copycat:
                 copycat_max_epochs=None, copycat_batch_size=None, copycat_lr=None,
                 copycat_gamma=None, copycat_dataset_root=None, copycat_balance_dataset=None, copycat_resume_filename=None,
                 #finetune:
                 finetune_max_epochs=None, finetune_batch_size=None, finetune_lr=None,
                 finetune_gamma=None, finetune_dataset_root=None, finetune_balance_dataset=None, finetune_resume_filename=None,
                 #seed
                 seed=None):
        #required:
        self.problem_name = problem_name
        self.oracle_filename = oracle_filename
        self.copycat_filename = copycat_filename
        self.finetune_filename = finetune_filename
        self.train_oracle = train_oracle
        self.train_copycat = train_copycat
        self.label_copycat_dataset = label_copycat_dataset
        self.train_finetune = train_finetune
        self.label_finetune_dataset = label_finetune_dataset
        self.only_print_reports = only_print_reports
        #genearal configuration:
        self.config = None
        self.config_file = config_file
        self.validation_step = validation_step
        self.save_snapshot = save_snapshot
        self.problems = self.get_problem_names()
        #oracle:
        self.oracle_resume_filename = oracle_resume_filename
        self.oracle_max_epochs = oracle_max_epochs
        self.oracle_batch_size = oracle_batch_size
        self.oracle_lr = oracle_lr
        self.oracle_gamma = oracle_gamma
        self.oracle_dataset_root = oracle_dataset_root
        #copycat:
        self.copycat_resume_filename = copycat_resume_filename
        self.copycat_max_epochs = copycat_max_epochs
        self.copycat_batch_size = copycat_batch_size
        self.copycat_lr = copycat_lr
        self.copycat_gamma = copycat_gamma
        self.copycat_dataset_root = copycat_dataset_root
        self.copycat_balance_dataset = copycat_balance_dataset
        #finetune:
        self.finetune_resume_filename = finetune_resume_filename
        self.finetune_max_epochs = finetune_max_epochs
        self.finetune_batch_size = finetune_batch_size
        self.finetune_lr = finetune_lr
        self.finetune_gamma = finetune_gamma
        self.finetune_dataset_root = finetune_dataset_root
        self.finetune_balance_dataset = finetune_balance_dataset
        #seed
        self.seed = seed

    def __load_config(self):
        if self.config is None:
            self.config = Config(self.config_file)
        return self.config

    def parse_value(self, attr_name):
        self.__load_config()
        if attr_name.startswith('oracle'):
            opts = self.config.get_oracle_options(self.problem_name)
            config_key = attr_name.replace('oracle_','')
        elif attr_name.startswith('copycat'):
            opts = self.config.get_copycat_options(self.problem_name)
            config_key = attr_name.replace('copycat_','')
        elif attr_name.startswith('finetune'):
            opts = self.config.get_finetune_options(self.problem_name)
            config_key = attr_name.replace('finetune_','')
        else:
            opts = self.config.get_general_options()
            config_key = attr_name
        try:
            local_value = getattr(self, attr_name)
            config_value = opts[config_key] if config_key in opts else None
            return local_value if local_value is not None else config_value
        except:
            return None

    def get_db_name(self, model, db_name):
        self.__load_config()
        name = self.config.get_value(self.problem_name, db_name, model=model)
        if name is not None:
            aux = self.config.get_value(self.problem_name, 'data', model=model)['datasets']
            if name in aux:
                return name, aux[name]
            else:
                return name, None
        return None, None

    def get_problem_names(self):
        return Config(self.config_file).get_problem_names()

    def __get_repr_oracle(self):
        fmt = f"  Oracle:\n"
        if self.train_oracle:
            fmt+= f"     Model filename: '{self.oracle_filename}'\n"
        if self.oracle_resume_filename is not None:
            fmt+= f"     Resume filename: '{self.oracle_resume_filename}'\n"
        # max epochs
        if self.train_oracle and not self.only_print_reports:
            fmt+= f"     Maximum training epochs: {self.parse_value('oracle_max_epochs')}\n"
            # batch size
            fmt+= f"     Batch size: {self.parse_value('oracle_batch_size')}\n"
            # lr
            fmt+= f"     Learning Rate: {self.parse_value('oracle_lr')}\n"
            # gamma
            fmt+= f"     Gamma: {self.parse_value('oracle_gamma')}\n"
            db = self.get_db_name(model='oracle', db_name='db_train')
            if db[0] is not None:
                fmt+= f"     Dataset: {db[0]}"
                if db[1] is not None:
                    fmt+= f" ('{db[1]}')"
                fmt+= '\n'
        else:
            fmt+= f"     It will NOT be trained.\n"
        db_root = self.parse_value('oracle_dataset_root')
        if db_root != '': fmt+= f"     Dataset root: '{db_root}'\n"
        return fmt

    def __get_repr_copycat(self):
        fmt = f"  Copycat:\n"
        if self.train_copycat:
            fmt+= f"     Model filename: '{self.copycat_filename}'\n"
        if self.copycat_resume_filename is not None:
            fmt+= f"     Resume filename: '{self.copycat_resume_filename}'\n"
        if self.train_copycat and not self.only_print_reports:
            fmt+= f"     Maximum training epochs: {self.parse_value('copycat_max_epochs')}\n"
            # batch size
            fmt+= f"     Batch size: {self.parse_value('copycat_batch_size')}\n"
            # lr
            fmt+= f"     Learning Rate: {self.parse_value('copycat_lr')}\n"
            # gamma
            fmt+= f"     Gamma: {self.parse_value('copycat_gamma')}\n"
            db = self.get_db_name(model='copycat', db_name='db_train')
            if db[0] is not None:
                fmt+= f"     Dataset: {db[0]}"
                if db[1] is not None:
                    fmt+= f" ('{db[1]}')"
                fmt+= '\n'
            fmt+= f"     The dataset will {'' if self.parse_value('copycat_balance_dataset') else 'NOT '}be balanced.\n"
            fmt+= f"     The training dataset will {'' if self.label_copycat_dataset else 'NOT '}be labeled by the Oracle Model.\n"
        else:
            fmt+= f"     It will NOT be trained.\n"
        db_root = self.parse_value('copycat_dataset_root')
        if db_root != '': fmt+= f"     Dataset root: '{db_root}'\n"
        return fmt

    def __get_repr_finetune(self):
        fmt = f"  Copycat Finetuning:\n"
        if self.train_finetune:
            fmt+= f"     Model filename: '{self.finetune_filename}'\n"
        if self.finetune_resume_filename is not None:
            fmt+= f"     Resume filename: '{self.finetune_resume_filename}'\n"
        if self.train_finetune and not self.only_print_reports:
            fmt+= f"     Maximum training epochs: {self.parse_value('finetune_max_epochs')}\n"
            # batch size
            fmt+= f"     Batch size: {self.parse_value('finetune_batch_size')}\n"
            # lr
            fmt+= f"     Learning Rate: {self.parse_value('finetune_lr')}\n"
            # gamma
            fmt+= f"     Gamma: {self.parse_value('finetune_gamma')}\n"
            db = self.get_db_name(model='finetune', db_name='db_train')
            if db[0] is not None:
                fmt+= f"     Dataset: {db[0]}"
                if db[1] is not None:
                    fmt+= f" ('{db[1]}')"
                fmt+= '\n'
            fmt+= f"     The dataset will {'' if self.parse_value('finetune_balance_dataset') else 'NOT '}be balanced.\n"
            fmt+= f"     The training dataset will {'' if self.label_finetune_dataset else 'NOT '}be labeled by the Oracle Model.\n"
        else:
            fmt+= f"     It will NOT be trained.\n"
        db_root = self.parse_value('finetune_dataset_root')
        if db_root != '': fmt+= f"     Dataset root: '{db_root}'\n"
        return fmt

    def __repr__(self) -> str:
        fmt = "Options:\n"
        fmt+= f"  Problem: {self.problem_name}\n"
        ## ORACLE
        fmt+=self.__get_repr_oracle()
        ## COPYCAT:
        fmt+=self.__get_repr_copycat()
        ## COPYCAT FINETUNE:
        fmt+=self.__get_repr_finetune()
        ## REPORTS:
        fmt+='\n'
        if self.validation_step != 0 and not self.only_print_reports:
            fmt+= f"  Validation Steps: {self.parse_value('validation_step')}\n"
            fmt+= f"  A snapshot of the model will {'' if self.parse_value('save_snapshot') else 'NOT '}be saved for each validation step.\n"
        
        if self.only_print_reports:
            fmt+= "\nNOTE: As 'only-print-reports' was selected, the models will only be loaded (or created with random parameters) and tested.\n"
            fmt+= "NOTE: THE MODELS WILL NOT BE TRAINED!!!"
        
        fmt+= f"\nDevice to use: '{cuda.get_device_name()}'\n"
        
        if self.seed is not None:
            fmt+= f"\nThe following seed will be used for Torch, Numpy and Random: {self.seed}\n"
        return fmt

def parse_boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n', 'not', 'dont'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y', 'yeah', 'yeap', 'ofcouse'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def get_problem_names():
    p_aux = argparse.ArgumentParser()
    p_aux.add_argument('--config-file')
    aux_arg, _ = p_aux.parse_known_args()    
    return Config(aux_arg.config_file).get_problem_names()

def parse_params():
    problem_names = get_problem_names()
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem',  required=True, type=str, help='Problem name', choices=problem_names)

    parser.add_argument('--config-file', help="Use this option to use a different configuration file and override the default options set in 'config.toml'")
    parser.add_argument('--only-print-reports', action='store_true', help='Use this option to only load the models and print their reports.')
    parser.add_argument('--seed', type=int, help='Use this option to provide a new seed to Pytorch/Numpy/Random')

    parser.add_argument('--validation-step', type=int, help="Change validation step. Set 0 (zero) to disable validation during training.")
    parser.add_argument('--save-snapshot', type=parse_boolean, nargs='?', const=True, help="Save snapshots at each validation step")
    #Oracle
    parser.add_argument('--oracle', type=str, default='Oracle.pth', help="Filename to save the Oracle Model")
    parser.add_argument('--oracle-resume', type=str, help="Filename to resume the Oracle Model")
    parser.add_argument('--dont-train-oracle', action='store_true', help="You can use this option to: Resume the Oracle's Model, or test the problem on an Oracle's Model with random weights")
    parser.add_argument('--oracle-max-epochs', type=int, help="Change maximum epochs to train Oracle's Model")
    parser.add_argument('--oracle-batch-size', type=int, help="Batch size to train Oracle's Model")
    parser.add_argument('--oracle-lr', type=float, help="Learning rate to train Oracle's Model")
    parser.add_argument('--oracle-gamma', help="Gamma to train Oracle's Model. It is the value to decrease the learning rate (lr*gamma)")
    parser.add_argument('--oracle-dataset-root', type=str, default='', help="Root folder of dataset files (image list and images listes in it)")
    #Copycat:
    parser.add_argument('--copycat', type=str, default='Copycat.pth', help="Filename to save the Copycat Model")
    parser.add_argument('--copycat-resume', type=str, help="Filename to resume the Copycat Model")
    parser.add_argument('--dont-train-copycat', action='store_true', help="You can use this option to test a Copycat's Model with random weights")
    parser.add_argument('--copycat-max-epochs', type=int, help="Change maximum epochs to train Copycat's Model")
    parser.add_argument('--copycat-batch-size', type=int, help="Batch size to train Copycat's Model")
    parser.add_argument('--copycat-lr', type=float, help="Learning rate to train Copycat's Model")
    parser.add_argument('--copycat-gamma', help="Gamma to train Copycat's Model. It is the value to decrease the learning rate (lr*gamma)")
    parser.add_argument('--copycat-dataset-root', type=str, default='', help="Root folder of Copycat problem dataset")
    parser.add_argument('--copycat-balance-dataset', type=parse_boolean, help="Replicate or drop images to balance the number of images per class")
    parser.add_argument('--dont-label-copycat-dataset', action='store_true', help='Use this option to avoid labeling the Copycat training dataset (NPD)')
    #Finetune:
    parser.add_argument('--finetune', type=str, default='Copycat-Finetune.pth', help="Filename to save the Copycat Finetune Model")
    parser.add_argument('--finetune-resume', type=str, help="Filename to resume the Copycat Finetune Model")
    parser.add_argument('--dont-train-finetune', action='store_true', help="You can use this option to avoid finetuning Copycat model")
    parser.add_argument('--finetune-max-epochs', type=int, help="Change maximum epochs to finetune Copycat's Model")
    parser.add_argument('--finetune-batch-size', type=int, help="Batch size to finetune Copycat's Model")
    parser.add_argument('--finetune-lr', type=float, help="Learning rate to finetune Copycat's Model")
    parser.add_argument('--finetune-gamma', help="Gamma to finetune Copycat's Model. It is the value to decrease the learning rate (lr*gamma)")
    parser.add_argument('--finetune-dataset-root', type=str, default='', help="Root folder of Copycat problem dataset")
    parser.add_argument('--finetune-balance-dataset', type=parse_boolean, help="Replicate or drop images to balance the number of images per class")
    parser.add_argument('--dont-label-finetune-dataset', action='store_true', help='Use this option to avoid labeling the Copycat finetune dataset (NPD)')
    
    args = parser.parse_args()

    return Options(problem_name=args.problem,
                   only_print_reports=args.only_print_reports,
                   config_file=args.config_file, validation_step=args.validation_step, save_snapshot=args.save_snapshot,
                   #oracle
                   oracle_filename=args.oracle,
                   oracle_resume_filename=args.oracle_resume,
                   train_oracle=not args.dont_train_oracle,
                   oracle_max_epochs=args.oracle_max_epochs,
                   oracle_batch_size=args.oracle_batch_size,
                   oracle_lr=args.oracle_lr,
                   oracle_gamma=args.oracle_gamma,
                   oracle_dataset_root=args.oracle_dataset_root,
                   #copycat
                   copycat_filename=args.copycat,
                   copycat_resume_filename=args.copycat_resume,
                   train_copycat=not args.dont_train_copycat,
                   copycat_max_epochs=args.copycat_max_epochs,
                   copycat_batch_size=args.copycat_batch_size,
                   copycat_lr=args.copycat_lr,
                   copycat_gamma=args.copycat_gamma,
                   copycat_dataset_root=args.copycat_dataset_root,
                   copycat_balance_dataset=args.copycat_balance_dataset,
                   label_copycat_dataset=not args.dont_label_copycat_dataset,
                   #finetune
                   finetune_filename=args.finetune,
                   finetune_resume_filename=args.finetune_resume,
                   train_finetune=not args.dont_train_finetune,
                   finetune_max_epochs=args.finetune_max_epochs,
                   finetune_batch_size=args.finetune_batch_size,
                   finetune_lr=args.finetune_lr,
                   finetune_gamma=args.finetune_gamma,
                   finetune_dataset_root=args.finetune_dataset_root,
                   finetune_balance_dataset=args.finetune_balance_dataset,
                   label_finetune_dataset=not args.dont_label_finetune_dataset,
                   #seed
                   seed=args.seed)

def signal_handler(sig, frame):
    print("\nQuitting...")
    exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    options = parse_params()
    print(options)
    input('\nCheck the parameters and press ENTER to continue...\n')
    problem = Problem(problem=options.problem_name,
                      #oracle
                      oracle_filename=options.oracle_filename,
                      oracle_resume_filename=options.oracle_resume_filename,
                      oracle_dataset_root=options.oracle_dataset_root,
                      #copycat
                      copycat_filename=options.copycat_filename,
                      copycat_resume_filename=options.copycat_resume_filename,
                      copycat_dataset_root=options.copycat_dataset_root,
                      #finetune
                      finetune_filename=options.finetune_filename,
                      finetune_resume_filename=options.finetune_resume_filename,
                      finetune_dataset_root=options.finetune_dataset_root,
                      #configuration file
                      config_fn=options.config_file,
                      #seed
                      seed=options.seed)

    if options.only_print_reports:
        problem.print_reports()
    else:
        problem.run(validation_step=options.validation_step,
                    save_snapshot=options.save_snapshot,
                    #Oracle options:
                    train_oracle=options.train_oracle,
                    oracle_max_epochs=options.oracle_max_epochs,
                    oracle_batch_size=options.oracle_batch_size,
                    oracle_lr=options.oracle_lr,
                    oracle_gamma=options.oracle_gamma,
                    #Copycat options:
                    train_copycat=options.train_copycat,
                    label_copycat_dataset=options.label_copycat_dataset,
                    copycat_max_epochs=options.copycat_max_epochs,
                    copycat_batch_size=options.copycat_batch_size,
                    copycat_lr=options.copycat_lr,
                    copycat_gamma=options.copycat_gamma,
                    copycat_balance_dataset=options.copycat_balance_dataset,
                    #Copycat Finetune options:
                    finetune_copycat=options.train_finetune,
                    label_finetune_dataset=options.label_finetune_dataset,
                    finetune_max_epochs=options.finetune_max_epochs,
                    finetune_batch_size=options.finetune_batch_size,
                    finetune_lr=options.finetune_lr,
                    finetune_gamma=options.finetune_gamma,
                    finetune_balance_dataset=options.finetune_balance_dataset
                 )