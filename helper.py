import torch 

def normalize(x):
    """normalize pixel value between [0,1] """
    return x/255.0

def normalize0(x):
    """normalize pixel value between [-1,1] """
    return x/127.5 -1

def denormalize(x):
    """convert back pixel value from [-1,1] to [0,255] """
    return (x+1)*127.5

