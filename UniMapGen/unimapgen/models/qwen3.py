from __future__ import annotations

from typing import Any, Type

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from unimapgen.token_format import DiscreteMapTokenFormatter


def _resolve_qwen3_vl_model_class() -> Type[torch.nn.Module]:
    try:
        from transformers import Qwen3_VLForConditionalGeneration  # type: ignore

        return Qwen3_VLForConditionalGeneration
    except Exception:
        pass
    return AutoModelForCausalLM


def load_processor(model_name_or_path: str, formatter: DiscreteMapTokenFormatter) -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    formatter.register_tokens_with_processor(processor)
    return processor


def load_inference_model(model_or_checkpoint: str, device: str) -> torch.nn.Module:
    model_cls = _resolve_qwen3_vl_model_class()
    torch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    load_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_or_checkpoint,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if str(device).startswith("cuda"):
        load_kwargs["device_map"] = "auto"
    model = model_cls.from_pretrained(**load_kwargs)
    if not str(device).startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model
