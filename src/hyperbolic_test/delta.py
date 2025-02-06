
import math
import torch
import numpy as np

from scipy.spatial.distance import squareform, pdist

def get_delta(embeddings, sample_size=1500):
    idx = np.random.choice(len(embeddings), min(sample_size, len(embeddings)))
    all_features_small = embeddings[idx]

    dists = pdist(all_features_small)
    dists = squareform(dists)
    delta = delta_hyp(dists)
    diam = np.max(dists)

    relative_delta = 2*delta / diam
    e = np.finfo(float).eps
    best_possible_delta = (8*(1-e)**2)/((1-(1-e)**2)**2)
    best_possible_delta = math.acosh(best_possible_delta+1)
    best_possible_delta = 2*math.log(1+2**0.5)/best_possible_delta
    relative_delta -= best_possible_delta

    return delta, diam, relative_delta

def delta_hyp(dismat):
    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.mean(maxmin - XY_p)
