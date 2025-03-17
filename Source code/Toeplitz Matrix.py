import cv2
import numpy as np
from scipy import linalg
from fractions import Fraction


# height, width, number of channels in image
# height = img.shape[0]
# width = img.shape[1]
# channels = img.shape[2]


# h = np.arange(1,4)

# padding = np.zeros(h.shape[0] - 1, h.dtype)
# first_col = np.r_[h, padding]
# first_row = np.r_[h[0], padding]

# H = linalg.toeplitz(first_col, first_row) # scipy.linalg.toeplitz(First column of the matrix, First row of the matrix)

# print(repr(H))

# -------------------------------------------------------------------------------------------------------------------------------

# dragging effect (toeplitz matrix reference: https://stackoverflow.com/questions/34536264/how-can-i-generate-a-toeplitz-matrix-in-the-correct-form-for-performing-discrete)
def toeplitz_matrix(n):
    # fraction = Fraction(1,n)
    m = np.full(n,1/n)
    padding = np.zeros(img.shape[0])
    first_col = np.r_[m[0], padding[1:]]    # numpy.r_: Translates slice objects to concatenation along the first axis.
    first_row = np.r_[m, padding[n:]]

    H = linalg.toeplitz(first_col, first_row)
    print(repr(H))
    return H


# convert image type to double (https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python)
def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float64) / info.max # Divide all values by the largest possible value in the datatype


# read in an image
img = cv2.imread("lena.tif",0)
img = im2double(img)
cv2.imshow("Lena",img)
print(img)


toe = toeplitz_matrix(20)
res = np.dot(toe,img)
cv2.imshow("Result",res)


cv2.waitKey(0)
cv2.destroyAllWindows()