import importlib
import traceback

MODEL_REGISTRY = {}

MODEL_MODULES = [
    "alexnet",
    "alexnet_sin",
    "avid",
    "blip",
    "convnext",
    "convlstm",
    "cornet",
    "clip",
    "dfm",
    "dino",
    "dinov2",
    "dinov2_lstm",
    "fitvid",
    "gpt2",
    "gdt",
    "hubert",
    "glm",
    "hmax",
    "ijepa",
    "llama3",
    "mae",
    "mae_lstm",
    "mcvd",
    "mim",
    "motion_energy",
    "motion_net",
    "pixel",
    "pixelnerf",
    "pixelnerf_lstm",
    "predrnn",
    "r3m_resnet",
    "r3m_lstm",
    "resnet",
    "resnet_lstm",
    "resnext101wsl",
    "random_models",
    "robust_models",
    "s3d_text_video",
    "sam",
    "scaling_models",
    "selavi",
    "stable_diffusion",
    "simvp",
    "swin",
    "tau",
    "torch_video",
    "timesformer",
    "vgg",
    "voneresnet",
    "videomae",
    "vit",
    "vjepa",
    "vjepa2",
    "wav2vec2",
    "whisper",
]

for module_name in MODEL_MODULES:
    try:
        importlib.import_module(f"models.{module_name}")
    except ImportError as e:
        # Optional: log or warn
        print(f"[Warning] Failed to import model module: {module_name}\n{e}")
    except Exception:
        # Catch any other initialization errors
        print(
            f"[Warning] Unexpected error importing model module: {module_name}")
        traceback.print_exc()

# If you have other model folders, import them here as well:
# from models import ModelWrapper


def get_model_class_and_id(identifier):
    """
    Returns the model class and its identifier mapping from the registry.

    Args:
        identifier (str): Model identifier (e.g., "sam_base").

    Returns:
        tuple: (ModelClass, model_identifier_string) or None if identifier is
            unknown.
    Raises:
        ValueError: If the identifier is unknown.
    """
    model_info = MODEL_REGISTRY.get(identifier)
    if model_info:
        return model_info["class"], model_info["model_id_mapping"]
    else:
        raise ValueError(
            "Model Class Requested is unavailable. "
            "Here is a list of our models: ",
            list_available_models(),
        )


def get_model_instance(identifier):
    """
    Instantiates and returns a model instance based on the identifier.

    Args:
        identifier (str): Model identifier (e.g., "sam_base").

    Returns:
        object: An instance of the model class, or None if identifier is unknown.
    """
    model_class, model_id = get_model_class_and_id(identifier)
    if model_class:
        instance = model_class()
        # Initialize the model inside the instance
        instance.get_model(model_id)
        return instance
    return None


def list_available_models():
    """
    Returns a list of available model identifiers.
    """
    return list(MODEL_REGISTRY.keys())
