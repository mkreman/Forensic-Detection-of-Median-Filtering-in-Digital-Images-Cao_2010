# Importing required packages
import numpy as np
import math
from scipy.signal import convolve2d


# Dataset path variable
dataset_path = './Data/UCID/ucid.v2/'

# Useful functions
def conv2(x, base):    
    y = np.array([[1, -1]])
    if base == 'horizontal':
        return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode='same'), 2)
    if base == 'vertical':
        return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 1), mode='same'), 2)


def im2col(A, size):
    M, N = A.shape
    col = N - size[1] + 1
    row = M - size[0] + 1
    
    start_idx = (np.arange(size[0])[:, None]*N + np.arange(size[1])).T 
    end_idx = (np.arange(row)[:, None]*N + np.arange(col)).T

    return np.take(A, start_idx.ravel()[:, None] + end_idx.ravel()[::1])


# For computing TP and FP rates for a single image
def TPFP_calculator(X, Y, threshold):
    rates = [0, 0, 0, 0]
    #       [TP, FP, FN, TN]
    for f in X:              # X is Positive-Class data set
        if f >= threshold:
            rates[0] += 1
        else:
            rates[2] += 1
    for f in Y:             # Y is Negative-Class data set
        if f >= threshold:
            rates[1] += 1
        else:
            rates[3] += 1

    TP_rate = rates[0] / (rates[0] + rates[2])  # Recall of class 1
    FP_rate = rates[1] / (rates[1] + rates[3])  # Recall of class 0
    return (TP_rate, FP_rate)


# Function for extracting feature from the image array
def caoICME10(I, d, tau):
    # Inputs: 
    #    I: 3D-Numpy presentation of an image
    #    d: The width of the square statistical region
    #    tau: A threshold to determined a region is textured or not
    
    # Limit the image pixel values between 0 and 255
    np.clip(I, 0, 255, out=I)
    # change the data type to uint8
    I = I.astype('uint8')
    I = np.double(I)
    nh, nw = I.shape
    hlfwins = math.ceil(d/2)

    # Computing row difference of pixels and selecting dxd square statistic region
    row_difference_matrix = conv2(I, 'horizontal') == 0
    row_difference_matrix = row_difference_matrix[hlfwins:-hlfwins, hlfwins:-hlfwins]
    # Computing column difference of pixels and selecting dxd square statistic region
    column_difference_matrix = conv2(I, 'vertical') == 0
    column_difference_matrix = column_difference_matrix[hlfwins:-hlfwins, hlfwins:-hlfwins]

    z = im2col(I, (2*hlfwins+1, 2*hlfwins+1))

    # Variance matrix
    varMap = (np.var(z, axis=0, ddof=1).reshape(nw-2*hlfwins, nh-2*hlfwins)).T
    varMap = varMap >= tau
    
    # Row and column fingerprint for indetifying the medain filtering
    f_r = np.sum(row_difference_matrix*varMap)/np.sum(varMap)
    f_c = np.sum(column_difference_matrix*varMap)/np.sum(varMap)

    f = (f_r+f_c)/np.sqrt(2)
    return f

# # Example of feature of an image
# I = cv2.imread(os.path.join(dataset_path, 'ucid00003.tif'), 0)
# print(caoICME10(I, 7, 100))
