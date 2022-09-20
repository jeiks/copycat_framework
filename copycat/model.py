import torch
import torchvision.models as models
from .utils import save_model
from os import path as os_path

class Model():
    '''
    note: arch = None (it assumes vgg16), or string related to torchvision model, or a dict with {'arch': nn.Module child, 'kwargs': {}} or {'module': object_already_initializated}
    '''
    def __init__(self, n_outputs=None, name=None, pretrained=True, model_arch=None, state_dict=None, save_filename=None):
        if type(model_arch) == str:
            assert n_outputs is not None, 'When @n_outputs is not provided, a torchvision model cannot be used'
        self.n_outputs = n_outputs
        self.pretrained = pretrained
        self.model = self.__check_model_arch(model_arch)
        if name is not None:
            print(f'({name}) ', end='')
        if state_dict is not None:
            print(f'Loading model from "{state_dict}"...')
            self.load_state_dict(state_dict)
        else:
            print(f'Starting a new model with random parameters...')
        self.save_filename = save_filename

    def __check_model_arch(self, model_arch):
        if model_arch is None: model_arch = 'vgg16'
        if type(model_arch) == str:
            return self.__load_torchvision_model(model_arch)
        elif type(model_arch) == dict:
            if 'module' in model_arch:
                return model_arch['module']
            elif 'arch' in model_arch:
                if 'kwargs' in model_arch:
                    model = model_arch['arch'](**model_arch['kwargs'])
                else:
                    model = model_arch['arch']()
                return model
        raise Exception(f'"{model_arch}" is invalid for model_arch. You must use a string related to a torchvision model or a dict with your own model')

    def __load_torchvision_model(self, model_arch):
        model_list = [x for x in dir(models) if callable(getattr(models, x)) and x[0].islower()]
        assert model_arch in model_list, f"{model_arch} if not a valid torchvision's model. Options: {', '.join(model_list)}"
        model = getattr(models,model_arch)(pretrained=self.pretrained)
        last_layer = next(reversed(model._modules))
        if type(getattr(model, last_layer)) == torch.nn.modules.linear.Linear:
            #model.__dict__['_modules'][last_layer]
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features=in_features, out_features=self.n_outputs, bias=True)
        elif type(getattr(model, last_layer)) == torch.nn.modules.container.Sequential:      
            #model.__dict__['_modules'][last_layer]
            in_features  = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features=in_features, out_features=self.n_outputs, bias=True)
        else:
            raise ValueError(f'The last layer of the architecture "{model_arch}" is not Linear or Sequential.. I do not know how to proceed.')

        return model
    
    def load_state_dict(self, state_dict):
        if type(state_dict) is str:
            #trying to open a state_dict from a file
            if os_path.isfile(state_dict):
                state_dict = torch.load(state_dict)
                self.model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f'Could not access "{state_dict}"')
        else:
            try:
                self.model.load_state_dict(state_dict)
            except:
                raise AttributeError(f"'state_dict' must be a filename or a torch's state dict")
    
    def save(self, filename=None):
        assert filename is not None and self.save_filename is not None, 'You must provide a filename to save the model.'
        if filename is None: filename = self.save_filename
        save_model(self.model.state_dict(), filename)

    def __call__(self):
        return self.model
    
    def __repr__(self):
        return 'Copycat Model\n' + str(self.model)
    
    def cpu(self):
        self.model.cpu()
    
    def cuda(self):
        self.model.cuda()