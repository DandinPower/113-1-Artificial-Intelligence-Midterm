from torch.nn import Module
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from liger_kernel.transformers import apply_liger_kernel_to_llama
from src.utils import LigerKernelConfig

def create_model_by_deepspeed_liger_kernel(ds_config: dict, config: LigerKernelConfig, model_name: str, gradient_checkpointing: bool) -> Module:
    assert ds_config is not None, "ds_config must be provided"
    assert model_name is not None, "model_name must be provided"
    assert gradient_checkpointing is not None, "gradient_checkpoint must be provided"

    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    apply_liger_kernel_to_llama(
        rope=config.enable_rope_optimization,
        swiglu=config.enable_swiglu_optimization,
        cross_entropy=False,
        fused_linear_cross_entropy=config.enable_fused_linear_cross_entropy,
        rms_norm=config.enable_rms_norm_optimization
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


    
