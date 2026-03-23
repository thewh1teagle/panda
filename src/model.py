from transformers import Qwen3ForCausalLM
from src.config import get_model_config

def get_model():
    config = get_model_config()
    model = Qwen3ForCausalLM(config)
    return model
