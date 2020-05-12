# Class: SCC0251 - Processamento de Imagens (2020 / 1 Semester)
# Student: Enrique Gabriel da Silva Teles - nUsp: 10724326

import numpy as np
import imageio


def gaussian_kernel(x, sig):
    return float(float(1/(2 * float(np.pi) * sig*sig)) * float(np.exp(-x*x/(2 * sig * sig))))


def bilateral_filter(img, n, s, r):
    N, M = img.shape

    spatial_g = np.zeros([n, n]).astype(float)

    # compute the spatial gaussian component
    for x in range(n):
        for y in range(n):
            x_center = float(x - int((n-1)/2))
            y_center = float(y - int((n-1)/2))
            dist = float(np.sqrt(float(x_center*x_center) +
                         float(y_center*y_center)))  # euclides distance
            spatial_g[x, y] = gaussian_kernel(dist, s)

    # apply the convolution
    a = int((n-1)/2)
    b = int((n-1)/2)

    # zero-padding
    img_padd = np.zeros([N+a+a, M+b+b]).astype(np.uint8)
    img_padd[a:-a, b:-b] = img

    # output image
    img_out = np.array(img_padd, copy=True).astype(float)

    # for every pixel in the image but the padding
    for x in range(a, N-a):
        for y in range(b, M-b):
            i_f = 0.
            w_p = 0.

            # for each neighbor
            for i in range(-a, a+1):
                for j in range(-b, b+1):
                    # range gaussian
                    diff = float(img_padd[x+i, y+j]) - float(img_padd[x, y])
                    rang_g = gaussian_kernel(float(diff), float(r))

                    # total value of the filter
                    w_i = rang_g * spatial_g[a + i, b + j]
                    w_p = w_p + w_i

                    i_f = i_f + (w_i * float(img_padd[x+i, y+j]))

            i_f = i_f / w_p
            img_out[x, y] = i_f

    # remove zero-padding and convert to uint8
    img_out = img_out[a:-a, b:-b]
    return img_out.astype(np.uint8)

def convolution(img, kernel):
    N, M = img.shape
    n, m = kernel.shape

    a = int((n-1)/2)
    b = int((m-1)/2)

    # flip kernel
    kernel_flip = np.flip(np.flip(kernel, 0), 1)

    # add zero-padding
    img_padd = np.zeros([N+a+a, M+b+b]).astype(np.uint8)
    img_padd[a:-a, b:-b] = img

    # output image
    img_conv = np.zeros([img_padd.shape[0], img_padd.shape[1]]).astype(float)

    # for every pixel in the image but the padding
    for x in range(a, N-a):
        for y in range(b, M-b):
            # for each neighbor
            for i in range(-a, a+1):
                for j in range(-b, b+1):
                    img_conv[x, y] += img_padd[x+i, y+j] * kernel[a + i, b + j]
                    

    # remove zero-padding and convert to uint8
    img_conv = img_conv[a:-a, b:-b]
    return img_conv


def normalization(img):
    i_min = img.min()
    i_max = img.max()
    img_norm = (((img - i_min) * 255) / (i_max - i_min))
    return img_norm

def unsharp(img, c, kernel):
    img_out = convolution(img, kernel) # convolution
    img_out = normalization(img_out) # scaling

    img_out = (c * img_out) + img # adding
    img_out = normalization(img_out) # scalling

    return img_out.astype(np.uint8)


def vignette(img, sig_row, sig_col):
    N,M = img.shape
    
    # compute kernels
    w_row = np.zeros([N]).astype(float)
    w_col = np.zeros([M]).astype(float)

    if(N % 2 == 0):
        a = int((N/2) -1)
    else:
        a = int((N-1)/2)

    if(M % 2 == 0):
        b = int((M/2) -1)
    else:
        b = int((M-1)/2)


    # for each row
    for r in range(N):
        w_row[r] = gaussian_kernel(r-a, sig_row)

    # for each col
    for c in range(M):
        w_col[c] = gaussian_kernel(r-b, sig_col)

    # transpose
    t_row = w_row.transpose()
    w_new = np.multiply(t_row, w_col)
    

    img_out = (w_new * img).astype(float) 
    img_out = normalization(img_out)

    return img_out.astype(np.uint8)



def compare(img_in, img_out):
    rse_matrix = np.power(img_in.astype(float) - img_out.astype(float), 2)
    rse = rse_matrix.sum()  # sum every element of the new matrix
    rse = np.sqrt(rse) # get the square root of the sum

    return round(rse, 4)

# read initial inputs
filename= str(input()).rstrip()
input_img = imageio.imread(filename)
method = int(input()) # 1, 2 or 3
save = int(input())


if method == 1:
    # read input
    n = int(input())
    s = float(input())
    r = float(input())

    # Bilateral filter
    output_img = bilateral_filter(input_img, n, s, r)

if method == 2:
    # read input
    c = float(input()) # >= 1
    kernel = int(input()) # 1 or 2

    kernel1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Unsharp mask using the Laplacian Filter
    if(kernel == 1):
        output_img = unsharp(input_img, c, kernel1)
    elif(kernel == 2):
        output_img = unsharp(input_img, c, kernel2)

if method == 3:
    sig_row = float(input())
    sig_col = float(input())

    # Vignette Filter
    output_img = vignette(input_img, sig_row, sig_col)

# check if is going to save output image or not
if save == 1:
    imageio.imwrite('output_img.png', output_img)

# compare input and output images and print the root squared error
rse = compare(input_img, output_img)
print(rse)
