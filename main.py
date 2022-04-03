#!/usr/bin/env python

import signal
from sys import exit, argv, stderr
import argparse
from threading import local

from idna import valid_contextj
from copycat import Problem, Config
from torch import cuda

class Options:
    def __init__(self,
                 #required parameters:
                 problem_name, oracle_filename, copycat_filename,
                 train_oracle, train_copycat, label_copycat_dataset,
                 only_print_reports,
                 #general configutation:
                 config_file=None,
                 validation_step=None,
                 save_snapshot=None,
                 #oracle:
                 oracle_max_epochs=None, oracle_batch_size=None, oracle_lr=None, oracle_gamma=None, oracle_dataset_root=None,
                 #copycat:
                 copycat_max_epochs=None, copycat_batch_size=None, copycat_lr=None, copycat_gamma=None, copycat_dataset_root=None, copycat_balance_dataset=None):
        #required:
        self.problem_name = problem_name
        self.oracle_filename = oracle_filename
        self.copycat_filename = copycat_filename
        self.train_oracle = train_oracle
        self.train_copycat = train_copycat
        self.label_copycat_dataset = label_copycat_dataset
        self.only_print_reports = only_print_reports
        #genearal configuration:
        self.config_file = config_file
        self.validation_step = validation_step
        self.save_snapshot = save_snapshot
        self.problems = self.get_problem_names()
        #oracle:
        self.oracle_max_epochs = oracle_max_epochs
        self.oracle_batch_size = oracle_batch_size
        self.oracle_lr = oracle_lr
        self.oracle_gamma = oracle_gamma
        self.oracle_dataset_root = oracle_dataset_root
        #copycat:
        self.copycat_max_epochs = copycat_max_epochs
        self.copycat_batch_size = copycat_batch_size
        self.copycat_lr = copycat_lr
        self.copycat_gamma = copycat_gamma
        self.copycat_dataset_root = copycat_dataset_root
        self.copycat_balance_dataset = copycat_balance_dataset

    def parse_value(self, attr_name):
        config = Config(self.config_file)
        if attr_name.startswith('oracle'):
            opts = config.get_oracle_options(self.problem_name)
            config_key = attr_name.replace('oracle_','')
        elif attr_name.startswith('copycat'):
            opts = config.get_copycat_options(self.problem_name)
            config_key = attr_name.replace('copycat_','')
        else:
            opts = config.get_general_options()
            config_key = attr_name
        try:
            local_value = getattr(self, attr_name)
            config_value = opts[config_key] if config_key in opts else None
            return local_value if local_value is not None else config_value
        except:
            return None

    def get_problem_names(self):
        return Config(self.config_file).get_problem_names()

    def __repr__(self) -> str:
        fmt = "Options:\n"
        fmt+= f"  Problem: {self.problem_name}\n"
        ## ORACLE
        fmt+= f"  Oracle:\n"
        fmt+= f"     Model filename: '{self.oracle_filename}'\n"
        # max epochs
        if self.train_oracle and not self.only_print_reports:
            fmt+= f"     Maximum training epochs: {self.parse_value('oracle_max_epochs')}\n"
            # batch size
            fmt+= f"     Batch size: {self.parse_value('oracle_batch_size')}\n"
            # lr
            fmt+= f"     Learning Rate: {self.parse_value('oracle_lr')}\n"
            # gamma
            fmt+= f"     Gamma: {self.parse_value('oracle_gamma')}\n"
        else:
            fmt+= f"     It will NOT be trained.\n"
        db_root = self.parse_value('oracle_dataset_root')
        if db_root != '': fmt+= f"     Dataset root: '{db_root}'\n"
        ## COPYCAT:
        fmt+= f"  Copycat:\n"
        fmt+= f"     Model filename: '{self.copycat_filename}'\n"
        if self.train_copycat and not self.only_print_reports:
            fmt+= f"     Maximum training epochs: {self.parse_value('copycat_max_epochs')}\n"
            # batch size
            fmt+= f"     Batch size: {self.parse_value('copycat_batch_size')}\n"
            # lr
            fmt+= f"     Learning Rate: {self.parse_value('copycat_lr')}\n"
            # gamma
            fmt+= f"     Gamma: {self.parse_value('copycat_gamma')}\n"
            fmt+= f"     The dataset will {'' if self.parse_value('copycat_balance_dataset') else 'NOT '}be balanced.\n"
            fmt+= f"     The training dataset will {'' if self.label_copycat_dataset else 'NOT '}be labeled by the Oracle Model.\n"
        else:
            fmt+= f"     It will NOT be trained.\n"
        db_root = self.parse_value('copycat_dataset_root')
        if db_root != '': fmt+= f"     Dataset root: '{db_root}'\n"
        ## REPORTS:
        fmt+='\n'
        if self.validation_step != 0 and not self.only_print_reports:
            fmt+= f"  Validation Steps: {self.parse_value('validation_step')}\n"
            fmt+= f"  A snapshot of the model will {'' if self.parse_value('save_snapshot') else 'NOT '}be saved for each validation step.\n"
        
        if self.only_print_reports:
            fmt+= "\nNOTE: As 'only-print-reports' was selected, the models will only be loaded (or created with random parameters) and tested.\n"
            fmt+= "NOTE: THE MODELS WILL NOT BE TRAINED!!!"
        else:
            fmt+= f"\nThe model will be trained on '{cuda.get_device_name()}'\n"
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
    parser.add_argument('--oracle',  type=str, default='Oracle.pth', help="Filename to save or resume the Oracle Model")
    parser.add_argument('--copycat', type=str, default='Copycat.pth', help="Filename to save or resume the Copycat Model")
    parser.add_argument('--dont-train-oracle', action='store_true', help="You can use this option to: Resume the Oracle's Model,"
                                                                         "or test the problem on an Oracle's Model with random weights")
    parser.add_argument('--dont-train-copycat', action='store_true', help="You can use this option to test a Copycat's Model with random weights")
    parser.add_argument('--dont-label-copycat-dataset', action='store_true', help='Use this option to avoid labeling the Copycat training dataset (NPD)')

    parser.add_argument('--config-file', help="Use this option to use a different configuration file and override the default options set in 'config.toml'")

    parser.add_argument('--validation-step', type=int, help="Change validation step. Set 0 (zero) to disable validation during training.")
    parser.add_argument('--save-snapshot', type=parse_boolean, nargs='?', const=True, help="Save snapshots at each validation step")

    parser.add_argument('--oracle-max-epochs', type=int, help="Change maximum epochs to train Oracle's Model")
    parser.add_argument('--oracle-batch-size', type=int, help="Batch size to train Oracle's Model")
    parser.add_argument('--oracle-lr', type=float, help="Learning rate to train Oracle's Model")
    parser.add_argument('--oracle-gamma', help="Gamma to train Oracle's Model. It is the value to decrease the learning rate (lr*gamma)")
    parser.add_argument('--oracle-dataset-root', type=str, default='', help="Root folder of dataset files (image list and images listes in it)")
    
    parser.add_argument('--copycat-max-epochs', type=int, help="Change maximum epochs to train Copycat's Model")
    parser.add_argument('--copycat-batch-size', type=int, help="Batch size to train Copycat's Model")
    parser.add_argument('--copycat-lr', type=float, help="Learning rate to train Copycat's Model")
    parser.add_argument('--copycat-gamma', help="Gamma to train Copycat's Model. It is the value to decrease the learning rate (lr*gamma)")
    parser.add_argument('--copycat-dataset-root', type=str, default='', help="Root folder of Copycat problem dataset")
    parser.add_argument('--copycat-balance-dataset', type=parse_boolean, help="Replicate or drop images to balance the number of images per class")

    parser.add_argument('--copycat_train_dataset', type=str)
    
    parser.add_argument('--only-print-reports', action='store_true', help='Use this option to only load the models and print their reports.')
    args = parser.parse_args()

    return Options(problem_name=args.problem, oracle_filename=args.oracle, copycat_filename=args.copycat,
                   train_oracle=not args.dont_train_oracle, train_copycat=not args.dont_train_copycat,
                   label_copycat_dataset=not args.dont_label_copycat_dataset,
                   only_print_reports=args.only_print_reports,
                   config_file=args.config_file, validation_step=args.validation_step, save_snapshot=args.save_snapshot,
                   oracle_max_epochs=args.oracle_max_epochs,
                   oracle_batch_size=args.oracle_batch_size,
                   oracle_lr=args.oracle_lr,
                   oracle_gamma=args.oracle_gamma,
                   oracle_dataset_root=args.oracle_dataset_root,
                   copycat_max_epochs=args.copycat_max_epochs,
                   copycat_batch_size=args.copycat_batch_size,
                   copycat_lr=args.copycat_lr,
                   copycat_gamma=args.copycat_gamma,
                   copycat_dataset_root=args.copycat_dataset_root,
                   copycat_balance_dataset=args.copycat_balance_dataset)

def signal_handler(sig, frame):
    print("\nQuitting...")
    exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    options = parse_params()
    print(options)
    input('\nCheck the parameters and press ENTER to continue... ')
    problem = Problem(problem=options.problem_name,
                      oracle_filename=options.oracle_filename,
                      oracle_dataset_root=options.oracle_dataset_root,
                      copycat_filename=options.copycat_filename,
                      copycat_dataset_root=options.copycat_dataset_root,
                      config_fn=options.config_file)
    if options.only_print_reports:
        problem.print_reports()
    else:
        problem.run(train_oracle=options.train_oracle,
                    train_copycat=options.train_copycat,
                    label_copycat_dataset=options.label_copycat_dataset,
                    validation_step=options.validation_step,
                    save_snapshot=options.save_snapshot,
                    #Oracle:
                    oracle_max_epochs=options.oracle_max_epochs,
                    oracle_batch_size=options.oracle_batch_size,
                    oracle_lr=options.oracle_lr,
                    oracle_gamma=options.oracle_gamma,
                    #Copycat:
                    copycat_max_epochs=options.copycat_max_epochs,
                    copycat_batch_size=options.copycat_batch_size,
                    copycat_lr=options.copycat_lr,
                    copycat_gamma=options.copycat_gamma,
                    copycat_balance_dataset=options.copycat_balance_dataset)
