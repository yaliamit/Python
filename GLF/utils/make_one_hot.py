import torch

def make_one_hot(labels, C=2):
    '''
    Converts an integer label to a one-hot tensor.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
        
    return target