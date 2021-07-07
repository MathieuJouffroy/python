import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImageProcessor():
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

class ScrapBooker():
	def crop(self, array: np.array, dimensions: tuple, position = (0, 0))-> np.array:# -> np.array:
		if (dimensions[0] > array.shape[0] or dimensions[1] > array.shape[1]):
			print (f"Error: Cannot crop with dimensions ({dimensions[0]} x {dimensions[1]}) bigger than the image ({array.shape[0]} x {array.shape[1]})")
			exit ()
		else:
			cropped_arr = array[position[0]: position[0] + dimensions[0],
								position[1]: position[1] + dimensions[1]]
		return cropped_arr

	def thin(self, array: np.array, n: int, axis=0) -> np.array:
		if axis == 0:
			return array[:, :n]
		elif axis == 1:
			return array[:n, :]

	def juxtapose(self, array: np.array, n: int, axis=0) -> np.array:
		if axis == 0:
			return np.tile(array, (n, 1))
		elif axis == 1:
			return np.tile(array, (1, n))

	def mosaic(self, array: np.array, dimensions: tuple) -> np.array:
		return np.tile(array, dimensions)
		
# slice : start:stop:step

## for color image : 3D ndarray of tuple (row (height) x column (width) x color (3)).

### TESTS ###
imp = ImageProcessor()
arr = imp.load("../boin.jpg")
imp.display(arr)
plt.show()
img = ScrapBooker()

start = time.time()
array_1 = img.thin(arr, 1800, 1)
end = time.time()
print(f"thin: [ exec-time = {end - start:.7f} ms ]")
plt.imshow(array_1)
plt.suptitle('thin')
plt.show()

start = time.time()
array = img.crop(arr, (1800, 1000), (900, 2510))
end = time.time()
print(f"crop: [ exec-time = {end - start:.7f} ms ]")
plt.imshow(array)
plt.suptitle('crop')
plt.show()

start = time.time()
arr2 = img.juxtapose(array, 4, 1)
end = time.time()
print(f"juxtapose: [ exec-time = {end - start:.7f} ms ]")
print (arr2.shape)
plt.suptitle('juxtapose')
plt.show()

start = time.time()
arr1 = img.mosaic(array, (3, 3, 1))
end = time.time()
print(f"mosaic : [ exec-time = {end - start:.7f} ms ]")
plt.imshow(arr1)
plt.suptitle('mosaic')
plt.show()

# numpy.s_  :  A nicer way to build up index tuples for arrays.
# np.s_[2::2] ->  slice(2, None, 2)
# slice(start, stop, step)
# >>> np.array([0, 1, 2, 3, 4])[np.s_[2::2]]   -> array([2, 4])