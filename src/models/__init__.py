_REGISTRY = {}


def register(name):
    def decorator(cls):
        _REGISTRY[name] = cls  # e.g. "resnet50" -> ResNet class
        return cls  # return cls unchanged so it works normally too

    return decorator


def build_model(cfg):
    cls = _REGISTRY[cfg.model]  # look up by string name from config
    return cls.from_config(cfg)  # instantiate with its config group


from . import compass_2d_resnet
