import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ColorFilter():
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
    
    def invert(self, array: np.array):
        #  to invert the color of one pixel, we subtract the pixel's color values 
        #  from the maximum, 255 (8-bit number).
        print (array)
        return (255 - array[:, :, :3])

    def to_blue(self, array: np.array) -> np.array:
        # create zero matrix
        split_img = np.zeros(array.shape, dtype="uint8")
        # assign blue channel
        split_img[:, :, 2] = array[:, :, 2]
        if split_img.shape[-1] == 4:
            split_img[:, :, 3] = array[:, :, 3]
        return split_img

    def to_green(self, array: np.array) -> np.array:
        # using only * operator :
        # return array[:, :, :] * [0, 1, 0]
        # array[:, :, [0, 2]] = 0
        array[:, :, [0, 2]] = array[:, :, [0, 2]] * 0 # faster than line above
        return array

    def to_red(self, array: np.array) -> np.array:
        # array[:, :, [1, 2]] = array[:, :, [1, 2]] * 0
        # return array
        # using to to_green and to_blue :
        green = self.to_green(np.copy(array[:, :, :3]))
        blue = self.to_blue(np.copy(array[:, :, :3]))
        array[:, :, :3] = array[:, :, :3] - green - blue
        return array

    def celluloid(self, array, thresh=4):
        thresholds = np.linspace(0, 255, num=thresh + 1, dtype="uint8")
        #  np.logical_and([True, False], [False, False])
        for i in np.arange(thresh):
            array[(array >= thresholds[i]) & (array < thresholds[i+1])] = thresholds[i]
            array[(array >= thresholds[-2])] = thresholds[-1]
        return array

    def primary(self, array):
        # 255/2 = 127 
        array[(array > 127)] = 255
        array[(array <= 127)] = 0
        return array

    def color_reduction(self, array):
        # Cut off the remainder of the division using // and multiply again, 
        # the pixel values become discrete values and the number of colors can be reduced
        array_32 = array // 32 * 32
        #array_128 = array // 128 * 128
        return array_32

    def to_grayscale(self, array, filter='w'):
        # sum shape reshape, broadcast to
        assert filter in ["w", "weighted"] or filter in [
            "m", "mean"], "invalid filter"
        if filter in ['m', 'mean']:
            # using sum, broadcast_to, newaxis and shape
            # reduce to 2d -> shape(array rows, array columns)
            # reshape to 3d : add new axis -> shape()
            # reshape with original shape
            # x.shape[:-1] all except last dimension
            # (*mean.shape[:-1]) # -> array.shape[0] to [-1], here to [1]
            mean_reduce_2d = (np.sum(array[:, :, :3], axis=2) / 3).astype(int)
            mean = mean_reduce_2d[:, :, np.newaxis] # shape(rows, cols, 1)
            mean = np.broadcast_to(mean, (*mean.shape[:-1], 3))
            return mean
        elif filter in ['w', 'weighted']:
            # using dot reshape, shape and tile
            # avg = np.dot(array[..., :3],...
            avg = np.dot(array[:, :, :3], [0.299, 0.587, 0.114]).astype(int)
            avg = avg.reshape(*avg.shape, 1)
            avg = np.tile(avg, (1, 3))
            return (avg)
    
    def to_grayscale2(self, array, filter='w'):
        assert filter in ["w", "weighted"] or filter in [
            "m", "mean"], "invalid filter"
        if filter in ['m', 'mean']:
            # using mean, newaxis, shape and broadcast_to
            # gray = np.mean(array[:, :, :3], axis=2).astype(int)
            # gray = gray[:, :, np.newaxis]
            gray = np.mean(array[:, :, :3], axis=2, keepdims=True).astype(int)
            gray = np.broadcast_to(gray, (*gray.shape[:-1], 3))
        elif filter in ['w', 'weighted']: 
            # Using only sum and tile
            gray = (np.sum((array[..., :3] * [0.299, 0.587, 0.114]), axis=2, keepdims=True)).astype(int)
            gray = np.tile(gray, (1, 3))
        return (gray)

    def to_grayscale3(self, array, filter='w'):
        # Using only sum
        assert filter in ["w", "weighted"] or filter in [
            "m", "mean"], "invalid filter"
        if filter in ['m', 'mean']:
            mean = (np.sum(array[:, :, :3], axis=2, keepdims=True) / 3).astype(int)
            array[:, :, :3] = mean
            return array
        elif filter in ['w', 'weighted']:
            weighted = (np.sum((array[:, :, :3] * [0.299, 0.587, 0.114]), axis=2, keepdims=True)).astype(int)
            array[:, :, :3] = weighted
            return array

    def masking(self, array):
        # seperate the row and column values
        total_row , total_col , layers = array.shape
        # create vector
        x , y = np.ogrid[:total_row , :total_col]
        # get the center values of the image        
        cen_x , cen_y = total_row/2 , total_col/2     
        # Measure distance value from center to each border pixel.     
        # To make it easy, we can think it's like, we draw a line from center-     
        # to each edge pixel value --> s**2 = (Y-y)**2 + (X-x)**2    
        distance_from_the_center = np.sqrt((x-cen_x)**2 + (y-cen_y)**2)
        # Select convenient radius value    
        radius = total_row/2
        circular_pic = distance_from_the_center > radius
        # let assign value zero for all pixel value that outside the cirular disc.      
        # All the pixel value outside the circular disc, will be black now.        
        array[circular_pic] =  0
        return array

    def put_mask(self, array, mask = "circle"):
        # seperate the row and column values
        total_row , total_col , layers = array.shape
        # create vector
        x , y = np.ogrid[:total_row , :total_col]
        # get the center values of the image        
        cen_x , cen_y = total_row/2 , total_col/2     
        #create a circle mask which is centered in the middle of the image, and with radius 100
        circle_mask = (x-cen_x)**2 + (y-cen_y)**2 <= 200**2
        array[circle_mask] = [0,0,0]

        #square_mask = (x<200)&(x>100)&(y<600)&(y>500)
        return array

    def rotate(self, array):
        # seperate the row and column values
        total_row , total_col , layers = array.shape
        # create vector
        x , y = np.ogrid[:total_row , :total_col]
        #rotating an image 90 degrees CCW is like mapping the pixel at (x,y) to the pixel at (y,-x)
        rotate = array[y, -x]
        return rotate
    
    def halo(self, array):
        intensity = 255
        #create a noramlizing constant so the halo fully fades out at the corners
        denominator = intensity/((n/2)**2 + (m/2)**2)

img = ColorFilter()
arr = img.load("../boin.jpg")
print (arr)
arr = img._dtype_to_int(arr)
print (arr)

# Basic properties of Image : 
plt.suptitle('original')
img.display(arr)

# View Each Channel
plt.title('R channel')
plt.imshow(arr[ : , : , 0])
plt.show()

plt.title('G channel')
plt.imshow(arr[ : , : , 1])
plt.show()

plt.title('B channel')
plt.imshow(arr[ : , : , 2])
plt.show()

# Values to full intensity for following rows:
plt.suptitle('RGB full intensity for specific rows')
arr_r = arr.copy()
arr_r[100:150 , : , 0] = 255 #  R channel
arr_r[300:350 , : , 1] = 255 #  G channel
arr_r[500:550 , : , 2] = 255 #  B channel
plt.imshow(arr_r)
plt.show()

arr1 = arr.copy()
start = time.time()
arr1 = img.invert(arr1)
end = time.time()
print(f"invert:\t\t[ exec-time = {end - start:.7f} ms ]\n")
plt.suptitle('invert')
img.display(arr1)

arr_mask = arr.copy()
start = time.time()
arr_mask = img.masking(arr_mask)
end = time.time()
print(f"masking:\t\t[ exec-time = {end - start:.7f} ms ]\n")
plt.suptitle('masking')
img.display(arr_mask)

arr_mask = arr.copy()
start = time.time()
arr_mask = img.put_mask(arr_mask)
end = time.time()
print(f"put_mask:\t\t[ exec-time = {end - start:.7f} ms ]\n")
plt.suptitle('put_mask')
img.display(arr_mask)

arr2 = arr.copy()
start = time.time()
plt.suptitle('blue filter')
arr2 = img.to_blue(arr2)
end = time.time()
print(f"to_blue:\t[ exec-time = {end - start:.7f} ms ]")
img.display(arr2)

arr3 = arr.copy() 
start = time.time()
plt.suptitle('green filter')
arr3 = img.to_green(arr3)
end = time.time()
print(f"to_green:\t[ exec-time = {end - start:.7f} ms ]")
img.display(arr3)

arr4 = arr.copy()
start = time.time()
plt.suptitle('red filter')
arr4 = img.to_red(arr4)
end = time.time()
print(f"to_red:\t\t[ exec-time = {end - start:.7f} ms ]\n")
img.display(arr4)

arr5 = arr.copy()
start = time.time()
plt.suptitle('shade')
arr5 = img.celluloid(arr5, 5)
end = time.time()
print(f"cell shade:\t[ exec-time = {end - start:.7f} ms ]\n")
img.display(arr5)

arr_p= arr.copy()
start = time.time()
plt.suptitle('primary')
arr_p = img.primary(arr_p)
end = time.time()
print(f"primary:\t[ exec-time = {end - start:.7f} ms ]")
img.display(arr_p)

arr6 = arr.copy()
start = time.time()
plt.suptitle('gray scale1 : mean')
arr6 = img.to_grayscale(arr6, 'm')
end = time.time()
print(f"to_grayscale:\t[ exec-time = {end - start:.7f} ms ]")
img.display(arr6)

arr7 = arr.copy()
start = time.time()
plt.suptitle('gray scale1 : weighted')
arr7 = img.to_grayscale(arr7)
end = time.time()
print(f"to_grayscale:\t[ exec-time = {end - start:.7f} ms ]\n")
img.display(arr7)

arr6 = arr.copy()
start = time.time()
plt.suptitle('gray scale2 : mean')
arr6 = img.to_grayscale2(arr6, 'm')
end = time.time()
print(f"to_grayscale2:\t[ exec-time = {end - start:.7f} ms ]")
img.display(arr6)

arr7 = arr.copy()
start = time.time()
plt.suptitle('gray scale2 : weighted')
arr7 = img.to_grayscale2(arr7)
end = time.time()
print(f"to_grayscale2:\t[ exec-time = {end - start:.7f} ms ]\n")
img.display(arr7)


arr6 = arr.copy()
start = time.time()
plt.suptitle('gray scale3 : mean')
arr6 = img.to_grayscale3(arr6, 'm')
end = time.time()
print(f"to_grayscale3:\t[ exec-time = {end - start:.7f} ms ]")
img.display(arr6)

arr7 = arr.copy()
start = time.time()
plt.suptitle('gray scale3 : weighted')
arr7 = img.to_grayscale3(arr7)
end = time.time()
print(f"to_grayscale3:\t[ exec-time = {end - start:.7f} ms ]")
img.display(arr7)

# A 3D array is like a stack of matrices:

# plt.subplots
# plt.figure(figsize = (15,15))
# The first index, i, selects the matrix
# The second index, j, selects the row
# The third index, k, selects the column
