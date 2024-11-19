#!/usr/bin/env python

import numpy as np

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30
    return x

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # Vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x

# c
# 图 word_vectors.png 显示出由于不同的范围尺度，x 轴的变化比 y 轴更大。
# 一些词聚类得很好，例如，“amazing”（惊人的）、“wonderful”（美妙的）和“great”（伟大的）都是相似的形容词，彼此靠近。
# 看到“rain”（雨）和“snow”（雪）或“tea”（茶）和“coffee”（咖啡）这样的词彼此不远也很有趣。
# 然而，大多数聚类中都有一些离群点，例如，“boring”（无聊的）在上述第一个聚类中，尽管它更像是一个反义词；
# 一些相似的词如“king”（国王）和“queen”（王后）或“male”（男性）和“man”（男人）应该更靠近。