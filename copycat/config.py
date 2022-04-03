import yaml
import inspect
import os
import re

class Config:
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

    def get_problem_options(self, problem, model=None):
        '''returns a level 1 dictionary with its options as keys, except datasets and classes that is level 2'''
        # copying default options:
        config = self.config['default'].copy()
        # removing subsection copycat and oracle:
        for k in ['copycat', 'oracle']:
            if k in config: config.pop(k)
        # inserting/updating specific default_scope options:
        if model in self.config['default']:
            config.update(self.config['default'][model])
        # selecting and inserting/updating specific problem options:
        if 'problems' in self.config and problem != 'default':
            if problem in self.config['problems'].keys():
                # problem general options:
                config.update(self.config['problems'][problem])
                # removing level 2
                for k in ['copycat', 'oracle']:
                    if k in config: config.pop(k)
                if model in self.config['problems'][problem]:
                    # inserting/updating specific model options:
                    config.update(self.config['problems'][problem][model])
                if 'data' in config:
                    #only oracle needs the measures
                    if 'measures' in config['data'] and model == 'copycat':
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
        config = self.get_problem_options(problem, model)
        return option if option in self.config else None

    def get_general_options(self):
        return self.get_problem_options('default')

    def get_oracle_options(self, problem):
        return self.get_problem_options(problem, 'oracle')

    def get_copycat_options(self, problem):
        return self.get_problem_options(problem, 'copycat')

    def get_data_options(self, problem):
        ret = self.get_problem_options(problem)
        if 'data' in ret:
            return ret['data']
        else:
            print(f"The {problem} data's information was not found in the configuration file.")
            return []

    def get_dataset_options(self, problem):
        ret = self.get_data_options(problem)
        if 'datasets' in ret:
            return ret['datasets']
        else:
            print(f"The {problem} data's information was not found in the configuration file.")
            return {}

    def get_problem_names(self):
        if 'problems' in self.config:
            return list(self.config['problems'].keys())
        else:
            return []

    def get_classes(self, problem):
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
        ret = self.get_data_options(problem)
        if 'measures' in ret:
            ret = ret['measures']
            if 'mean' in ret and 'std' in ret:
                return tuple(ret['mean']), tuple(ret['std'])
        return None, None

    def get_outputs(self, problem):
        ret = self.get_problem_options(problem)
        return ret['outputs'] if 'outputs' in ret else None

    def get_config(self, problem, scope=None):
        '''
        problem: ACT, DIG, FER, GOC, PED, SHN, SIG
        scope: copycat, oracle, datasets, classes
        '''
        if scope == 'data':
            return self.get_dataset_options(problem, scope)
        else:
            return self.get_problem_options(problem, scope)