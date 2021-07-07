import numpy as np
from numpy.lib import stride_tricks
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import scipy.ndimage as nd


# https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html?highlight=tensordot#numpy.tensordot
# https://jessicastringham.net/2017/12/31/stride-tricks/ 
# https://stackoverflow.com/questions/48097941/strided-convolution-of-2d-in-numpy
# https://setosa.io/ev/image-kernels/ 
# Indexing with a mask can be very useful to assign a
#  new value to a sub-array:
# a[a % 3 == 0] = -1
# # array([10, -1,  8, -1, 19, 10, 11, -1, 10, -1, -1, 20, -1,  7, 14])
class AdvancedFilter:
	
	def load(self, path):
		img = mpimg.imread(path)
		print(f'Image Type: {type(img)}')
		print(f"Image dtype: {img.dtype}")
		print(f"Image nb bytes: {img.nbytes}")
		print(f"Image strides: {img.strides}")
		print(f"Image Shape: {img.shape}")
		print(f"Image Height: {img.shape[0]} | Width: {img.shape[1]}")
		print(f"Image Dimensions: {img.ndim}")
		print(f"Image Size: {img.size}")
		print(f"Max RGB Value: {img.max()}")
		print(f"Min RGB Value: {img.min()}")
		print(f"RGB values for pixel (100th rows, 50th column): {img[100, 50]}\n")
		return (img)

	def display(self, img):
		plt.imshow(img)
		plt.show()
	
	def _dtype_to_int(self, array):
		if array.dtype == np.float32:
			array = (array * 255).astype(np.uint8)
		if array.dtype == np.float64:
			array = (array * 255 * 255).astype(np.uint16)
		return array

	def mean_blur(self, array, size):
		#size = 3 * stdv
		#assert size % 2 == 0, "size must be odd"
		# mean_kernel = np.ones((size, size), dtype="uint8")
		new_img = np.zeros(array.shape, dtype="uint8")
		kn_border = int((size - 1) / 2)
		for row in np.arange(kn_border, array.shape[0] - kn_border):
			for col in np.arange(kn_border, array.shape[1] - kn_border):
				for m in np.arange(0, array.shape[2]):
					new_img[row, col, m] = array[row-kn_border:row+kn_border+1, col-kn_border:col+kn_border+1, m].mean()
		return (new_img)

	def vec_mean_blur(self, array, kernel_size):
		shape = (array.shape[0] - kernel_size + 1, array.shape[1] - kernel_size + 1, kernel_size, kernel_size)
		new_shape = (array.shape[0] - kernel_size + 1, array.shape[1] - kernel_size + 1, array.shape[2])
		new_img = np.zeros(new_shape, dtype="uint8")
		strides = 2 * array.strides[:2]
		for m in np.arange(array.shape[2]):
			patches = stride_tricks.as_strided(array[:, :, m], shape=shape, strides=strides)
			print (patches.shape)
			# flattening the inner 2d arrays to length-100 vectors, 
			# and then computing the mean on the final axis:
			# veclen = kernel_size ** 2 
			# patches.reshape(*patches.shape[:2], veclen).mean(axis=-1).shape
			# OR : compute a mean over the last two axes, which should be more efficient than reshaping:
			# patches.mean(axis=(-1, -2)).shape -> 350 , 626
			# np.allclose(patch_means, strided_means) -> True
			strided_means = patches.mean(axis=(-1, -2))
			new_img[:, :, m] = strided_means
			print (new_img.shape)
		return new_img

	#def gaussian_blur(self, array, kernel_size):
		# specify the strides :  tuple of bytes to jump in each dimension when moving along the array
		# Each pixel in img is a uint8(1-byte), meaning the total image size is 354 x 630 x 3 x 1(byte) = 669,060 bytes.
		# strides is hence a sort of “metadata”-like attribute that tells us how many bytes
		# we need to jump ahead to move to the next position along each axis. 
		# Here we move in block of 1  bytes along the rows but need to traverse 1 x 354 bytes to move “down” 
		# from one row to another.
		        # shape = (191, 191, 10, 10)
        # img.strides = (2400, 12, 4) et strides =  (2400, 12, 2400, 12)
					# The rule of thumb for Gaussian filter design is to choose the filter kernel_kernel_size to be 
					#  about 3 times the standard deviation (sigma value) in each direction,
					#  for a total filter size of approximately 6*sigma rounded to an odd integer value. 

					# The rule of thumb for Gaussian filter design is to choose the filter size to
					#  be about 3 times the standard deviation (sigma value) in each direction, for a 
					#  total filter size of approximately 6*sigma rounded to an odd integer value. 
					# This goes along with what you mentioned about truncating the Gaussian at 3*sigma.

# edit: More explanation - sigma basically controls how "fat" your kernel 
# function is going to be; higher sigma values blur over a wider radius. 
# Since you're working with images, bigger sigma also forces you to use a larger 
# kernel matrix to capture enough of the function's energy.


# As a reference, in Mathematica the function GaussianMatrix features several ways
# to compute a Gaussian discrete matrix, e.g. using discrete Bessel approximation.
# By default, radius = 2 * sigma, which means that with sigma = 1, the matrix will be 5x5.

img = AdvancedFilter()
arr = img.load("../minion.jpg")
#arr = img._dtype_to_int(arr)

arr1 = arr.copy()
start = time.time()
arr1 = img.mean_blur(arr1, 5)
end = time.time()
print(f"mean blurr:\t\t[ exec-time = {end - start:.7f} ms ]\n")

arrc = arr.copy()
start = time.time()
arrc = img.vec_mean_blur(arrc, 5)
end = time.time()
print(f"mean blurrfff 1:\t\t[ exec-time = {end - start:.7f} ms ]\n")

# arrc = arr.copy()
# start = time.time()
# arrc = img.vec_mean_blur2(arrc, 5)
# end = time.time()
# print(f"mean blurrff 2:\t\t[ exec-time = {end - start:.7f} ms ]\n")
# img.display(arr1)
# plt.suptitle('mean blurr')
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(arr1),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()

# scipy
# start = time.time()
# dst = np.zeros(img.shape, img.dtype)
# for i in xrange(img.shape[2]):
#     dst[:, :, i] = nd.gaussian_filter(img[:, :, i], 5)
# end = time.time()
# print(f"scipy blurr:\t\t[ exec-time = {end - start:.7f} ms ]\n")


img = cv2.imread('../minion.jpg')
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
start = time.time()
blur = cv2.blur(RGB_img,(5,5))
end = time.time()
print(f"cv2 blurr:\t\t[ exec-time = {end - start:.7f} ms ]\n")
plt.subplot(221),plt.imshow(RGB_img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur),plt.title('Blurred cv2')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(arr1),plt.title('Blurred me')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(arrc),plt.title('Blurred me2')
plt.xticks([]), plt.yticks([])
plt.show()


# A, B, C = get_data(N, n_N, M, n_M)
# from numpy.fft  import fft2, ifft2
# def np_fftconvolve(A, B):
#     return np.real(ifft2(fft2(A)*fft2(B, s=A.shape)))
# def test_numpy_fft(A, B, C, prefetching=False):
#     if prefetching:
#         for i_N in np.arange(A.shape[0]):
#             for i_M in np.arange(B.shape[0]):
#                 C[i_N, i_M, :, :] = np_fftconvolve(A[i_N, :, :], B[i_M, :, :])
#     else:
#         for i_N in np.arange(A.shape[-1]):
#             for i_M in np.arange(B.shape[-1]):
#                 C[:, :, i_N, i_M] = np_fftconvolve(A[:, :, i_N], B[:, :, i_M])                            
# A, B, C = get_data(N, n_N, M, n_M)
# %%timeit
# test_numpy_fft(A, B, C)