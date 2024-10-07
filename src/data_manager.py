import pickle
import os
import numpy as np
class DictToProps:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    def __getitem__(self, key):
        return self.__dict__[key]
    def keys(self):
        return self.__dict__.keys()

def save_obj(obj, name, keys = None, style = "separete", ):
    if style == "pkl":
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    elif style == "separete":
        if not os.path.exists(name):
            os.makedirs(name)
        sim_id = 0
        if keys is None:
            print('Warning: keys is None. All keys will be saved.')
            keys = obj.keys()
        while any([os.path.exists(f"{name}/{key}_{sim_id}.pkl") for key in obj.keys()]):
            sim_id += 1
        for key in obj.keys():

            with open(f"{name}/{key}_{sim_id}.pkl", 'wb') as f:
                pickle.dump(obj[key], f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, keys = None, number = None):
    if type(keys) == str:
        keys = [keys]
    if os.path.isdir(name):
        if keys is None:
            obj = {}
            for file in os.listdir(name):
                file = file.split(".")[0]
                with open(name + f"/{file}.pkl", 'rb') as f:
                    obj[file] = pickle.load(f)
            return DictToProps(obj)
        else:
            file_index = f'_{number}' if number is not None else ''
            obj = {}
            for key in keys:
                if os.path.exists(f"{name}/{key}{file_index}.pkl"):
                    with open(f"{name}/{key}{file_index}.pkl", 'rb') as f:
                        obj[key] = pickle.load(f)
                elif os.path.exists(name + f"{key[9:]}{file_index}.pkl"):
                    with open(name + f"{key[9:]}{file_index}.pkl", 'rb') as f:
                        obj[key[9:]] = pickle.load(f)
                else:
                    print(f'{key}{file_index} is not in {name}')
            return DictToProps(obj)
    if os.path.exists(name + '.pkl'):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    


