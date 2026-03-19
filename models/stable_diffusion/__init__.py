from models import MODEL_REGISTRY
from .sd_wrapper import StableDiffusion

MODEL_REGISTRY["sd21_t50"] = {
    "class": StableDiffusion, "model_id_mapping": "SD21-T50"}
MODEL_REGISTRY["sd21_t200"] = {
    "class": StableDiffusion, "model_id_mapping": "SD21-T200"}
MODEL_REGISTRY["sd21_t500"] = {
    "class": StableDiffusion, "model_id_mapping": "SD21-T500"}
MODEL_REGISTRY["sd21_t999"] = {
    "class": StableDiffusion, "model_id_mapping": "SD21-T999"}
