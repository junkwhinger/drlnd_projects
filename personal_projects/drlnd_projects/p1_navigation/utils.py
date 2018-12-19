import os
import json
import torch

class Params():

    def __init__(self, experiment_name):
        json_file_path = os.path.join("experiments", experiment_name, "params.json")
        with open(json_file_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__



def load_checkpoint(checkpoint, model):
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)

    return checkpoint
