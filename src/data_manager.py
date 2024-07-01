import pickle
import os
import numpy as np

def save_obj(obj, name, keys = None, style = "separete", ):
    if style == "pkl":
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    elif style == "separete":
        os.makedirs(name, exist_ok=True)
        if keys is None:
            for key in obj.keys():
                with open(name + f"/{key}.pkl", 'wb') as f:
                    pickle.dump(obj[key], f, pickle.HIGHEST_PROTOCOL)
        else:
            for key in keys:
                if key in obj.keys():
                    with open(name + f"/{key}.pkl", 'wb') as f:
                        pickle.dump(obj[key], f, pickle.HIGHEST_PROTOCOL)
                else:
                    print(f"{key} is not in obj.keys()")


def load_obj(name, keys = None):
    if os.isdir(name):
        if keys is None:
            obj = {}
            for file in os.listdir(name):
                with open(name + f"/{file}.pkl", 'rb') as f:
                    obj[file] = pickle.load(f)
            return obj
        else:
            obj = {}
            for key in keys:
                with open(name + f"/{key}.pkl", 'rb') as f:
                    obj[key] = pickle.load(f)
            return obj
    if os.path.exists(name + '.pkl'):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


