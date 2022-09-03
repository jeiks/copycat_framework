from .copycat import Copycat

class Finetune(Copycat):
    """
    This class is responsible to finetune the Copycat model.
    """
    def __init__(self, problem, model_arch=None,
                 save_filename=None, resume_filename=None,
                 dataset_root='', db_name_train=None, db_name_test=None,
                 dont_load_datasets=False, outputs=None, config_fn=None):
        super().__init__(problem=problem, model_arch=model_arch,
                         save_filename=save_filename, resume_filename=resume_filename,
                         dataset_root=dataset_root, db_name_train=db_name_train, db_name_test=db_name_test,
                         dont_load_datasets=dont_load_datasets, outputs=outputs, config_fn=config_fn, finetune=True)