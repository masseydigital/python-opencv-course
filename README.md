# python-opencv-course
Udemy Python for Computer Vision with OpenCV and Deep Learning Course

Install the course content with:

```powershell
conda env create -f cvcourse_windows.yml
```

Activate this course with:

```powershell
conda activate python-cvcourse
```

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

## Image Processing with OpenCV

### ColorSpaces

RGB is the most common color space that we see, but there are also other color spaces that have been developed such as HSL (hue, saturation, lightness) and HSV (hue, saturation, value).  HSL and HSV more closely represent how humans see colors.

HSL is a cylindrical model which has hue wrapping around, saturation pushing out, and lightness pushing up and down.   HSV is another cylindrical model, but instead of lightness, we have a value.

### Blending and Pasting Images

Blending images can be done with the _addWeighted_ function in OpenCV.  This function only works if the images are the same size.

new_pixel = a * pixel_1 + b * pixel_2 + y

_Masking_ allows us to select parts of an image that we define.   Generally this is done through making things to come through as white and things to mask out in black.

### Thresholding, Blurring, and Smoothing

_Thresholding_ is fundamentally a very simple method of segmenting an image into different parts.  Thresholding will convert an image to black and white.

_Smoothing_ an image can help get rid of noise, or help an application focus on general details.  Blurring or smoothing are often used for edge detection.

_Gamma Correction_ can be applied to an image to make it appear brighter or darker.

_Kerneling_ can be applied over an image to produce a variety of effects.

[Interactive Visualizer](http://setosa.io/ev/image-kernels/)

An _image kernel_ is a small matrix used to apply effects like the ones you might find in Photoshop or Gimp, such as blurring, sharpening, outlining, or embossing.  They are also used in machine learning for feature extraction.

### Morphological Operators

_Morphological Operators_ are sets of Kernels that can achieve a variety of effects such as reducing noise.  I.e. reducing black points on a white background.  Other operators can also achieve an erosion or dilation effect (common for text data).

_Erosion_ is a morphological operator that can be used to separate the foreground from the background.

_Dilation_ adds more to the foreground.

_Opening_ is an erosion followed by a dilation.  This is helpful for removing background noise.

```python
opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
```

_Closing_ is a good way to get rid of noise in the foreground.

_Gradient_ is a crude form of edge detection that subtracts the opening from the closing to give you the edge.

An _Image Gradient_ is a directional change in the intensity or color of an image.   _Sobel Feldman Operators_ are an example of an image gradient edge detection algorithm.  Gradients can be calculated in a specific direction.

A _normalized gradient magnitude_ shows multi-directional edges in one image.

### Histograms

A _histogram_ is a visual representation of the distribution of a continuous feature.  Histograms are represented by bar charts and can include a distribution trend line.  For images, we can display the frequency of values for colors.  Each channel has values between 0-255 and there are 3 channels in an rgb image.

_Histogram Equalization_ is a method of contrast adjustment based on the image's histogram.

## Video Basics

**Don't have multiple kernels accessing video feeds at one time**

## Object Detection

### Template Matching

_Template Matching_ is looking for an exact copy of an image in another image.  It does this by scanning a larger image for a provided template by sliding it across the larger image.  The main option that can be adjusted is the comparison method used as the target template to slide across the larger image.  The methods are some sort of correlation based metric.

### Corner Detection

_Corner Detection_ is looking for corners in an image.  A _Corner_ is a point whose local neighborhood stands in two dominant and different edge directions.  Another way this could be said is that a corner is the junction of two edges, where an edge is a sudden change in image brightness.  Two common edge detection algorithms are _Harris Corner Detection_ and _Shi-Tomasi Corner Detection._

_Harris Corner Detection_ is based around corners can be detected by looking for significant changes in all directions.  Regions that have no change in any direction are called _Flat_.  _Edges_ won't have a major change in either edge.

The _Shi-Tomasi Corner Detection_ algorithm makes a small modification to the Harris Corner Detection algorithm to achieve better results.  This modification is changing the scoring algorithm for the change detection.

### Edge Detection

_Edge Detection_ is expanding to find general edges of objects.  One of the most popular detectors is the _Canny Edge_ detector.  The Canny Edge Detection algorithm is a multi-stage algorithm.  

1) Apply Gaussian Filter to smooth the image in order to remove any noise.
2) Find the intensity gradients of the image
3) Apply non-maximum suppression to get rid of spurious response to edge detection.
4) Apply double threshold to determine potential edges.
5) Track edges by hysteresis: Finalize detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

**For high resolution images, it is a good idea to apply your own custom blur**
**The Canny Algorithm also requires a user to decide on low and high threshold values.**

### Grid Detection

_Grid Detection_ combines both concepts to detect grids in images.  Cameras often create distortion in an image such as radial distortion and tangential distortion.  A good way to way to account for these distortions when performing operations like object tracking is to have a recognizable pattern attached ot the object being tracked.  Grid patterns are often used to calibrate cameras and track motion.

### Contour Detection

_Contour Detection_ is used to detect foreground vs background and allows for detection of external vs internal contours (eyes and smile from a cartoon face).  _Contours_ are defined as simply a curve joining all the continuous points (along the boundary), having the same color or intensity.  They are useful for shape analysis and object detection/recognition.  _External Contours_ are contours that occur at the outside edge of a shape.  _Internal Contours_ are contours that occur in the inside of a shape.s

### Feature Matching

_Feature Matching_ are advanced methods of detecting matching objects in another image, even if the target image is not shown.  Feature matching extracts defining key features from an input image, then using a distance calculation, finds all the matches in a secondary image - meaning we no longer need an exact copy of the image.

### Watershed Algorithm

The _watershed algorithm_ is an advanced algorithm that allows us to segment images into foreground and background.  It also allows us to manually set seeds to choose segments of an image.  The term watershed comes from geography - a land area that channels rainfall and snowmelt to creeks, streams, and rivers, and eventually to outflow points such as reservoirs, bays, and the ocean.  In image processing this is represented by a grayscale topographic surface where _high intensity_ denotes peaks and hills while _low intensity_ denotes valleys.  The algorithm can then fill every isolated valleys with different colored labels.

### Haar Cascades and Face Detection

_Haar Cascades_ can be used to detect faces and images.  This is not considered facial recognition (need deep learning).  Haar Cascades are a key component of the Viola Jones object detection framework.  It is also important to note that face detection is not the same as face recognition.  This algorithm is useful to detect if there is a face in an image and locate it, but not who it belongs too.

The main features in the Viola-Jones algorithm are edge features (white against black) in horizontal and vertical directions, line features which are surrounded by white or black and four-rectangle features.  Our features are not binary so we utilize means to determine these features.  

Calculating the sums for the entire image would be computationally expensive, but the viola-jones algorithm solves this by using the integral image (summed area table) O(1) running time.  The algorithm also saves time by going through a cascade of classifiers.  Once an image fails a classifier, we can stop attempting to detect a face.

The downside to this algorithm is the very large data sets needed to create your own features.  Luckily though, many pre-training sets of features already exist!

## Object Tracking

### Optical Flow

Optical Flow is the pattern of apparent motion of images between two consecutive frames caused by movement of object or camera.  

Optical flow makes a few assumptions:

1) The pixel intensities of an object do not change between consecutive frames
2) Neighboring pixels have similar motion

In OpenCV, optical flow methods do the following (Lucas-Kanade function)...

* Take in a given set of points and a frame
* Attempt to find those same points in the next frame
* Uses supplies points to track

**The techniques that are given do not help differentiate if camera is moving or if the object is moving (in opposite directions)

The _Lucas Kanade_ algorithm only computes optical flow for a _sparse_ feature set (only the points that we define)

The _Gunner Farneback's_ algorithm is used to calculate _dense_ optical flow (all points).

[Image Pyramids](https://en.wikipedia.org/wiki/Pyramid_(image_processing)) can be used with the Lucas Kanade algorithm to find optical flow at multiple resolutions.

### MeanShift and CamShift

### Built-in Tracking Apis