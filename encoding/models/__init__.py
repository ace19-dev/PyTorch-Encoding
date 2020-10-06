from .model_zoo import get_model
from .model_zoo import model_list
from .model_store import get_model_file, pretrained_model_list

<<<<<<< HEAD
def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        # 'atten': get_atten,
        'encnet': get_encnet,
        # 'encnetv2': get_encnetv2,
        'deeplab': get_deeplab,
    }
    return models[name.lower()](**kwargs)
=======
from .sseg import get_segmentation_model, MultiEvalModule
>>>>>>> upstream/master
