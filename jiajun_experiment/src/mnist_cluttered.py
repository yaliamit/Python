import numpy as np
from CNNForMnist import load_data
import os

def selectSamples(examples, nSamples, dataSize):
    nExamples = examples.shape[0]
    samples = []
    for i in range(nSamples):
        samples.append(examples[np.random.randint(0, nExamples, dataSize)])
    return samples

def placeDistractions(config, examples, dataSize):
    distractors = selectSamples(examples, config['num_dist'], dataSize)
    dist_w = config['dist_w']
    megapatch_w = config['megapatch_w']
    patch = np.zeros((dataSize, megapatch_w, megapatch_w))
    for d_patch in distractors:
        t_y = np.random.randint(0, megapatch_w - dist_w + 1, dataSize)
        t_x = np.random.randint(0, megapatch_w - dist_w + 1, dataSize)
        s_y = np.random.randint(0, d_patch.shape[2] - dist_w + 1, dataSize)
        s_x = np.random.randint(0, d_patch.shape[3] - dist_w + 1, dataSize)
        for sampleIndex in range(dataSize):
            patch[sampleIndex, t_y[sampleIndex]:t_y[sampleIndex] + dist_w, t_x[sampleIndex]:t_x[sampleIndex] + dist_w] += d_patch[sampleIndex, 0, s_y[sampleIndex]:s_y[sampleIndex]+dist_w, s_x[sampleIndex]:s_x[sampleIndex]+dist_w]
    patch[patch > 1] = 1
    return patch


def placeSpriteRandomly(obs, sprite, boarder):
    h = obs.shape[1]
    w = obs.shape[2]
    spriteH = sprite.shape[1]
    spriteW = sprite.shape[2]
    print(h, w, spriteH, spriteW)
    y = np.random.randint(boarder, h - spriteH - boarder + 1)
    x = np.random.randint(boarder, w - spriteW - boarder + 1)
    obs[:, y:y+spriteH, x:x+spriteW] = obs[:, y:y+spriteH, x:x+spriteW] + sprite
    obs[obs > 1] = 1
    obs[obs < 0] = 0

    return obs


def updateConfig(config, extraConfig):
    if extraConfig != None:
        for key, value in extraConfig:
            config[key] = value
    return config

def createData(extraConfig = None, dataSize = 50, dataset = "training"):
    config = {
        'x_train_path': "/X_train.npy",
        'y_train_path': "/Y_train.npy",
        'x_test_path': "/X_test.npy",
        'y_test_path': "/Y_test.npy",
        'megapatch_w': 28,
        'num_dist': 1,
        'dist_w': 10,
        'boarder': 0,
        'nDigits': 0,
        'nClasses': 10,
    }
    config = updateConfig(config, extraConfig)
    X_train, y_train, X_test, y_test = load_data(config['x_train_path'], config['y_train_path'], config['x_test_path'], config['y_test_path'])
     
    if dataset == "training":
        x_data = X_train
        y_data = y_train
    else:
        x_data = X_test
        y_data = y_test

    nExamples = x_data.shape[0]
    obs = np.zeros((dataSize, config['megapatch_w'], config['megapatch_w']))
    step = nExamples
    obs = placeDistractions(config, x_data, dataSize)
    perm = np.arange(nExamples)
    for i in range(config['nDigits']):
        step = step + 1
        if step > nExamples:
            np.random.permutation(perm)
            step = 1

        sprite = x_data[perm[step][:dataSize]] 
        obs = placeSpriteRandomly(obs, sprite, config['boarder'])
        selectedDigit = y_data[perm[step][:dataSize]]

    return obs

