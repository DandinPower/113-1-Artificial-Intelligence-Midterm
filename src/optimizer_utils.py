import math
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_scheduler, SchedulerType
from torch.nn import Module

def create_optimizer(model: Module, lr: float, weight_decay: float, betas_0: float, betas_1: float) ->DeepSpeedCPUAdam:
    assert model is not None, "model must be provided"
    assert lr is not None, "lr must be provided"
    assert weight_decay is not None, "weight_decay must be provided"
    assert betas_0 is not None, "betas_0 must be provided"
    assert betas_1 is not None, "betas_1 must be provided"

    return DeepSpeedCPUAdam(model.parameters(), lr=lr, betas=(betas_0, betas_1))