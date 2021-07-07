import numpy as np
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

### TEST ###
imp = ImageProcessor()
arr = imp.load("../42AI.png")
imp.display(arr)
plt.show()

arr = imp.load("../boin.jpg")
lum_img = arr[:, :, 0]
plt.imshow(lum_img)
plt.show()

plt.imshow(lum_img, cmap="hot")
plt.show()

imgplot = plt.imshow(lum_img)
imgplot.set_cmap('nipy_spectral')
plt.show()

imgplot = plt.imshow(lum_img)
plt.colorbar()
plt.show()
