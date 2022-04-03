import torch
import torchvision.models as models
from .utils import save_model
from os import path as os_path

class Model():
    def __init__(self, n_outputs, pretrained=True, oracle_arch=True, state_dict=None, save_filename=None):
        self.n_outputs = n_outputs
        self.pretrained = pretrained
        self.__oracle_arch = oracle_arch
        if oracle_arch: self.model = self.__load_oracle_model()
        else:           self.model = self.__load_different_model()
        if state_dict is not None:
            print(f'Loading model from "{state_dict}"')
            self.load_state_dict(state_dict)
        self.save_filename = save_filename

    def __load_oracle_model(self):
        model = models.vgg16(pretrained=self.pretrained)
        in_features  = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features=in_features, out_features=self.n_outputs, bias=True)
        return model
    
    def __load_different_model(self):
        model = models.alexnet(pretrained=self.pretrained)
        in_features  = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features=in_features, out_features=self.n_outputs, bias=True)
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