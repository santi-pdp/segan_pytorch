"""
MIT License
Copyright (c) 2016 Santi Dsp
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from sys import version_info as py_version

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def RMSE(groundtruth, prediction, mask=None):
    """
    groundtruth: matrix containing the real samples to be predicted (N samples,
                 sample dim)
    prediction: matrix containing the prediction (N samples,
                sample dim)
    mask: optional binary mask to not consider certain samples
          (0 in mask)
    """
    groundtruth = np.array(groundtruth, dtype=np.float32)
    prediction = np.array(prediction, dtype=np.float32)
    assert groundtruth.shape == prediction.shape
    if mask is not None:
        mask = np.array(mask)
        groundtruth = groundtruth[mask == 1]
        prediction = prediction[mask == 1]
    D = (groundtruth - prediction) ** 2
    D = np.mean(D, axis=0)
    return np.sqrt(D)


def AFPR(groundtruth, prediction):
    """
    Evaluate Accuracy, F-measure, Precision and Recall for binary inputs
    """
    groundtruth = np.array(groundtruth)
    prediction = np.array(prediction)
    assert groundtruth.shape == prediction.shape
    # A: accuracy
    I = np.mean(groundtruth == prediction)
    F = f1_score(groundtruth, prediction)
    P = precision_score(groundtruth, prediction)
    R = recall_score(groundtruth, prediction)
    return I, F, P, R


def MCD(gt_cep, pr_cep):
    from six.moves import xrange
    """
    Mel Cepstral Distortion
    Input are matrices with shape (time, cc_dim)
    """
    MCD = 0
    for t in xrange(gt_cep.shape[0]):
        acum = 0
        # TODO Vectoritzar segon bucle
        for n in xrange(gt_cep.shape[1]):
            acum += (gt_cep[t, n] - pr_cep[t, n]) ** 2
        MCD += np.sqrt(acum)

    # scale factor
    alpha = ((10. * np.sqrt(2)) / (gt_cep.shape[0] * np.log(10)))
    MCD *= alpha
    return MCD
