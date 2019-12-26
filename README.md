# python-opencv-course
Udemy Python for Computer Vision with OpenCV and Deep Learning Course

## NumPy and Matplotlib Recap
[Numpy Manual](https://docs.scipy.org/doc/numpy/reference/index.html)

```python
import numpy as np
```

_.arange_ : gives you set of values between a range with an optional space parameter

_.zeros_ : gives you a array of zeros

_.ones_ : gives you an array of ones

_.shape_ : gives you the shape of the array

_.reshape_ : takes an existing array and modifies the shape

_.max_ : gives you the maximum value in an array

_.min_ : gives you the minimum value in an array

_.argmax_ : returns the index of the max value of the array

_.argmin_ : returns the index of the min value of the array

_.mean_ : returns the mean of the array

_.copy_ : returns a copy of an array

_.asarray_ : converts an image to an array

*Slicing* : cutting out parts of array.  the min/max can be ommitted to send to start at the beginning/go to the end of an array
myarray[min:max]

Images are represented by a matrix of pixels consisting of a 3 dimensional array with: red, green, and blue whose values are between 0 and 255.  They can also be grayscale with a single dimensional matrix of values between 0 and 255 or normalized between 0 and 1.

In a numpy array.. this might look like (1280,720,3) where 1280 is the pixel width, 720 is the pixel height, and 3 is the color channels.  The 3 color channels are actually grayscale images to the computer.  Display devices such as LED monitors convert those values into the colors that are seen.

[RGB Color Model](https://en.wikipedia.org/wiki/RGB_color_model)

The [matplotlib](https://matplotlib.org/) module can be used to display images inside of a Python notebook.  

The Python Imaging Library, _PIL_ library allows us to import images.

_plt.imshow_ : Special function which shows a transformed image into an array

viridis, plasma, inferno, and magma are colormaps that are specially designed for people whom are color blind.

## OpenCV Basics

### Basic Commands

[OpenCV](https://opencv.org/) is a library of programming functions mainly aimed at real-time computer vision.  The original library was written in C++ by Intel, but it also usable through Python bindings.

It is useful to check the type returned from _cv2.imread_ to ensure that the correct file is being read.  If the wrong file is returned it will read _NoneType_.  If the correct type is read in, it will read _numpy.ndarray_.

OpenCV and Matplotlib expect different orders of the r,g,b channels.

_cv2.cvtColor_ allows you to convert a source image from one color space to another color space.  You can also pass in IMREAD constants to read in an image in a predefined color scheme.

_cv2.resize_ allows you to resize an image into new dimensions.  You can either pass in a new width and height or a width and height ratio.

The _cv2.flip_ method allows you to flip the image either along the vertical or horizontal axis.  This command is commonly used to generate more training data for machine learning algorithms.

The _cv2.imwrite_ method allows you to write an image file to a new file.

```python
# This code allows you to resize an image in a Jupyter Notebook frame.
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.imshow(fix_img)
```

### Drawing on Images

_cv2.rectangle_ allows you to draw a rectangle on the image.  This method is called in-place so the changes to the image will be permanent.  

_cv2.cirlce_ allows you to draw circles on the image.  You can fill in shapes by providing -1 in the thickness.

_cv2.line_ allows you to draw a line from one point to another point on an image.

_cv2.putText_ allows you to write text on an image.

_cv2.polylines allows you to draw a multi-vertice polygon on an image.  In order to use this method you must first declare a set of vertices and then cast them using the reshape command.

_Callbacks_ allow us to connect images to event functions with OpenCV..  This allows us to interact with images in real-time and later on videos.

#### Connecting Callback Functions

OpenCV has events that can be used to input actions into a cv2 image window.  Mouse clicks are a common event that you can call.  You can use multiple if statements to provide multiple inputs.

### Image Processing with OpenCV

#### ColorSpaces

RGB is the most common color space that we see, but there are also other color spaces that have been developed such as HSL (hue, saturation, lightness) and HSV (hue, saturation, value).  HSL and HSV more closely represent how humans see colors.

HSL is a cylindrical model which has hue wrapping around, saturation pushing out, and lightness pushing up and down.   HSV is another cylindrical model, but instead of lightness, we have a value.

### Blending and Pasting Images

Blending images can be done with the _addWeighted_ function in OpenCV.  This function only works if the images are the same size.

new_pixel = a * pixel_1 + b * pixel_2 + y

_Masking_ allows us to select parts of an image that we define.   Generally this is done through making things to come through as white and things to mask out in black.



