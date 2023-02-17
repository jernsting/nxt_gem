import pickle


def read_gpickle(path: str):
    with open(path, 'rb') as f:
        g = pickle.load(f)
    return g
