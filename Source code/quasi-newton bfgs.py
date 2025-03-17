import cv2
import math
import numpy as np
from scipy import linalg
from numpy import linalg as LA
import matplotlib.pyplot as plt
from numpy.linalg import norm


# dragging effect (toeplitz matrix reference: https://stackoverflow.com/questions/34536264/how-can-i-generate-a-toeplitz-matrix-in-the-correct-form-for-performing-discrete)
def toeplitz_matrix(n):
    m = np.full(n, 1/n)
    padding = np.zeros(img.shape[0])
    # numpy.r_: Translates slice objects to concatenation along the first axis.
    first_col = np.r_[m[0], padding[1:]]
    first_row = np.r_[m, padding[n:]]

    H = linalg.toeplitz(first_col, first_row)
    # print(repr(H))
    return H


# convert image type to double (https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python)
def im2double(im):
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    # Divide all values by the largest possible value in the datatype
    return im.astype(np.float64) / info.max


# frobenius norm power by 2
def frobenius(x):
    A = blur_factor
    B = blurred_image
    return LA.norm(np.dot(A,x) - B, 'fro')**2


# Gradient of frobenius_p2
def gradientFrobenius(x):
    A = blur_factor
    B = blurred_image
    return 2*np.dot(np.dot(A.T,A),x) - 2*np.dot(A.T,B)
    # return 2*A.T@A@x - 2*A.T@B


# Lipschitz constant
def lipschitz():
    A = blur_factor
    eigen = LA.eigvals(np.dot(A.T,A))
    L = 2 * np.max(eigen)
    alpha = 1/L
    return alpha


def BFGS(x, g, d, tol=1e-4, max_loop=500):
    dim = len(x) # dimension of problem 
    H = np.eye(dim) # initial hessian

    loop = 0
    alpha = lipschitz()
    x_new = x + alpha * d
    g_new = gradientFrobenius(x_new)
    y = g_new - g
    s = x_new - x

    while LA.norm(g_new) >= tol and loop < max_loop:
        H = H + np.trace(y@y.T)/np.trace(y.T@s) - np.trace(H@s@s.T@H.T)/np.trace(s.T@H@s)
        d = -np.dot(H, g_new)
        x = x_new
        x_new = x + alpha * d
        g = g_new
        g_new = gradientFrobenius(x)
        loop += 1
    return x



# def bfgs(f, x0, max_iter=1000, tol=1e-6):
#     n = x0.size
#     Hk = np.eye(n)
#     xk = x0
#     gk = grad(f, xk)
#     fk = f(xk)
#     for k in range(max_iter):
#         pk = -np.dot(Hk, gk)
#         alpha = backtracking(f, xk, pk, gk)
#         xkp1 = xk + alpha * pk
#         sk = xkp1 - xk
#         gkp1 = grad(f, xkp1)
#         yk = gkp1 - gk
#         rhok = 1 / np.dot(yk, sk)
#         if np.isnan(rhok) or np.isinf(rhok):
#             return xk
#         Ak = np.eye(n) - rhok * np.outer(sk, yk)
#         Bk = np.eye(n) - rhok * np.outer(yk, sk)
#         Hkp1 = np.dot(np.dot(Ak, Hk), Bk) + rhok * np.outer(sk, sk)
#         if norm(gkp1) < tol:
#             return xkp1
#         Hk = Hkp1
#         xk = xkp1
#         gk = gkp1
#         fk = f(xk)
#     return xk


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



##############################################################################################################

# read in an image
img = cv2.imread("lena.tif", 0)
# must not convert the image to double to calculate PSNR and SSIM
img = im2double(img)
cv2.imshow("Original image", img)

# dragging effect
blur_factor = toeplitz_matrix(20)
blurred_image = np.dot(blur_factor, img)
cv2.imshow("Blurred image", blurred_image)


x0 = np.ones([512, 512])
dim = len(x0) # dimension of problem 
H = np.eye(dim) # initial hessian
g0 = gradientFrobenius(x0)
d0 = -np.dot(H, g0)
result = BFGS(x0, g0, d0, 1e-4, 500)
cv2.imshow("Result", result)
# print("The PSNR value is:", calculate_psnr(img, result)) # cv2.PSNR
# print("The SSIM value is:", calculate_ssim(img, result))

cv2.waitKey(0)