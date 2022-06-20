from config import Config
from models.model import BaseModel
import importlib


def find_model_using_name(model_name):
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace("_", "") + "model"
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase."
            % (model_filename, target_model_name)
        )
        exit(0)

    return model


def create_model(config: Config) -> BaseModel:
    model = find_model_using_name(config.model)
    instance = model(config)
    print("model {} was created".format(type(instance).__name__))
    return instance
