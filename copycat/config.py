import yaml
import inspect
import os
import re
import copy

class Config:
    '''
    Class to read configuration file and easily get specific configuration
    Args:
        filename: use it only to use other yaml file (default: copycat/config.yaml)
    Example:
        from copycat import config
        c = config.Config()
        print(c.get_problem_options('DIG'))
    '''
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    def __init__(self, filename=None):
        if filename is None:
            d = os.path.dirname(inspect.getfile(self.__class__))
            self.fn = os.path.join(d, 'config.yaml')
        else:
            self.fn = filename
        os.path.exists(self.fn), f"Cannot access '{self.fn}'"
        self.config = yaml.load(open(self.fn), Loader=self.loader)
        self.models = ['oracle', 'copycat', 'finetune']

    def get_problem_options(self, problem, model=None):
        '''
        Args:
            problem: problem name
            model: 'oracle', 'copycat' or 'finetune' (None to get general problem options)
        Return:
            A dictionary with the problem options/configs
            If any option is not specified in the model scope, the general option will be returned
            (general option: options bellow problem label in "config.yaml")
        '''
        assert model in self.models or model == None, f'Model argument must be {self.models} or None'
        assert 'problems' in self.config, f'You have to add problems inside "problems" section in config file'
        # copying default options:
        config = copy.deepcopy(self.config['default'])
        # removing unnecessary model subsections:
        for m in self.models:
            if m in config: config.pop(m)
        # inserting/updating specific default_scope options using general model options:
        if model in self.config['default']:
            config.update(copy.deepcopy(self.config['default'][model]))
        # selecting and inserting/updating specific problem options:
        if problem != 'default':
            if problem in self.config['problems']:
                # problem general options:
                config.update(copy.deepcopy(self.config['problems'][problem]))
                # removing level 2
                for m in self.models:
                    if m in config:
                        config.pop(m)
                # inserting/updating specific model options:
                if model in self.config['problems'][problem]:
                    config.update(copy.deepcopy(self.config['problems'][problem][model]))
                if 'data' in config:
                    #only oracle needs the measures
                    if model not in ['oracle', None]:
                        if 'measures' in config['data']:
                            config['data'].pop('measures')
            else:
                print(f'It was not found configutation for "{problem}", "{model}". The default parameters will be provided instead.')

        # trick to provide the model outputs:
        if 'outputs' not in config and 'classes' in config:
            len_classes = len(config['classes'])
            if len_classes > 0:
                config['outputs'] = len_classes

        return config

    def get_value(self, problem, option, model=None):
        '''
            method to get a specific value for an option
        Args:
            problem: problem name
            option: the requested option
            model: 'oracle', 'copycat' or 'finetune' (None to get general problem options)
        '''
        assert model in self.models or model == None, f'Model argument must be {self.models} or None'
        config = self.get_problem_options(problem, model)
        if option in config:
            return config[option]
        else:
            return None

    def get_general_options(self):
        '''
            method to get general/default options for all problems
        '''
        return self.get_problem_options('default')

    def get_oracle_options(self, problem):
        '''
            method to get oracle options for a specific problem
        '''
        return self.get_problem_options(problem, 'oracle')

    def get_copycat_options(self, problem):
        '''
            method to get copycat options for a specific problem
        '''
        return self.get_problem_options(problem, 'copycat')

    def get_finetune_options(self, problem):
        '''
            method to get finetune options for a specific problem
        '''
        return self.get_problem_options(problem, 'finetune')

    def get_model_db_names(self, problem, model):
        '''
            method to get the model's dataset names
        Args:
            problem: problem name
            model: oracle, copycat, finetune, or None
        '''
        ret = {}
        for db_name in ['db_train', 'db_test']:
            aux = self.get_value(problem, db_name, model)
            if aux is not None:
                ret[db_name] = aux
        return ret

    def get_data_options(self, problem, model=None):
        '''
            method to get all data options/configs for the problem and model (if specified)
        Args:
            problem: problem name
            model: oracle, copycat, finetune, or None
        '''
        p_opts = self.get_problem_options(problem, model)
        if 'data' in p_opts:
            ret = {}
            ret.update(p_opts['data'])
            if model is None:
                #get db_names for all models:
                for m in self.models:
                    ret[m] = {}
                    db_names = self.get_model_db_names(problem, m)
                    if len(db_names) != 0: ret[m].update(db_names)
            else:
                #get db_names only for related model:
                db_names = self.get_model_db_names(problem, model)
                if len(db_names) != 0: ret.update(db_names)
            return ret
        else:
            print(f"The {problem} data's information was not found in the configuration file.")
            return None

    def get_dataset_options(self, problem):
        '''
            method to get all dataset filenames of the problem
        '''
        ret = self.get_data_options(problem)
        if ret is None:
            return []
        elif 'datasets' in ret:
            return ret['datasets']
        else:
            print(f"The {problem} data's information was not found in the configuration file.")
            return {}

    def get_dataset_filename(self, problem, model, name):
        '''
            method to get a filename for a specific dataset
        Examples:
            In : c.get_dataset_filename('DIG', 'copycat', 'db_train')
            Out: 'data/dig_npd.txt.bz2'
            (value present at problems->DIG->copycat->db_train)
        '''
        d_opts = self.get_data_options(problem, model)
        if name in d_opts and 'datasets' in d_opts:
            return d_opts['datasets'][d_opts[name]]
        else:
            return None

    def get_problem_names(self):
        '''
        method to get the problem names
        '''
        if 'problems' in self.config:
            return list(self.config['problems'].keys())
        else:
            return []

    def get_classes(self, problem):
        '''
            method to get classes names of the problem
        '''
        ret = self.get_dataset_options(problem)
        if 'classes' in ret:
            return ret['classes']
        else:
            ret = self.get_problem_options(problem)
            if 'classes' in ret:
                return ret['classes']
            else:
                print(f"The {problem}'s classes were not found in the configuration file.")
                return []

    def get_mean_std(self, problem):
        '''
            method to get the data's mean and std of the problem
        '''
        ret = self.get_data_options(problem)
        if ret is not None:
            if 'measures' in ret:
                ret = ret['measures']
                if 'mean' in ret and 'std' in ret:
                    return tuple(ret['mean']), tuple(ret['std'])
        return None, None

    def get_outputs(self, problem):
        '''
            method to get the number of problem outputs
        '''
        ret = self.get_problem_options(problem)
        return ret['outputs'] if 'outputs' in ret else None

    def get_config(self, problem, scope=None):
        '''
            method to get the problem configuration
        Args:
            problem: problem name
            scope: copycat, oracle, finetune, data, classes
        '''
        if scope in self.models:
            return self.get_problem_options(problem, model=scope)
        else:
            d_opts = self.get_problem_options(problem)
            if scope is None:
                return d_opts
            elif scope in d_opts:
                return d_opts[scope]
            else:
                return None