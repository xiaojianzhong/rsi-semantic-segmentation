import segmentation_models_pytorch as smp

from configs import CFG
from .bce import BCELoss, SigmoidBCELoss
from .ce import CELoss, SoftmaxCELoss
from .dice import DiceLoss, SigmoidDiceLoss


def build_criterion():
    if CFG.CRITERION.NAME == 'ce':
        criterion = smp.losses.SoftCrossEntropyLoss()
    elif CFG.CRITERION.NAME == 'bce':
        criterion = smp.losses.SoftBCEWithLogitsLoss()
    elif CFG.CRITERION.NAME == 'dice':
        criterion = smp.losses.DiceLoss('multiclass', from_logits=True)
    else:
        raise NotImplementedError('invalid criterion: {}'.format(CFG.CRITERION.NAME))
    return criterion
