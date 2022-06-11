# Arabic-Handwritten-OCR
## Step 1: Line Segmentation of Parargraph images
In step 1, you input an image of a handwritten body of Arabic text (i.e. paragraph with multiple lines) and you receive an output folder containing a number of images equal to the number of lines written in the input paragraph image. Each output image contains a handwritten line.

I have provided an example of a paragraph image taken from [KHATT dataset](http://khatt.ideas2serve.net/index.php) below. Notice that the chosen example has lines which are almost parallel with the horizontal edge of the image frame and the lines have uniform spacings. This is crucial for the algorithm to extract the lines correctly. If your image does not meet these requirements, you may consider preprocessing steps such as [deskewing the image](https://pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/).
![AHTD3A0158_Para1](https://user-images.githubusercontent.com/47701869/173181877-221cc557-4426-44d1-b5bc-2aa4ff18b7bc.jpg)
The output of the algorithm (for the first line) should look like this. 
![example](https://user-images.githubusercontent.com/47701869/173182156-4052f5e1-e22b-44d7-8da5-49ce93fd0d4d.jpg)
Because the paragraphs are handwritten in freestyle, some letters with descenders and/or ascenders may overlap with the lines above and/or below the current line. The algorithm uses the gradients of the intensity histograms of thresholded images to find the best cut position for each individual line that will maximize the line content.

## Step 2: Word and subword segmentation of Line images
Given an input image of a handwritten Arabic line (i.e. a single line with multiple words), you receive an output folder containing a number of images equal to the number of words written in the input line image. Each output image contains a handwritten word/subword For the example of this step, we will use the output line we received from the previous line segmentation step.

Using the word segmentation algorithm, we either receive a complete word like the one shown below, if the word consists entirely of fully connected letters:

![example_word](https://user-images.githubusercontent.com/47701869/173183253-6aca4420-a019-4981-afe0-eff2fbefa70f.jpg)

Or we receive a subword (an image of a partial word) if the word consists of a group of connected and disconnected letters like so:

![disconnedted_letter1](https://user-images.githubusercontent.com/47701869/173183383-387d5526-3a43-4966-b06d-a204f2a19c8b.jpg)

![disconnected_letter_2](https://user-images.githubusercontent.com/47701869/173183403-67607619-9041-4d22-9d2c-5eb4c21b2d67.jpg)

The reason why it is favorable to perform this double segmentation step is that out of a limited dataset we can receive a substantial amount of samples to train on.
However, the [KHATT dataset](http://khatt.ideas2serve.net/index.php) does not provide a ground truth on word and subword levels. For this project we have labeled a total of 5160 images found in the dataset folder.

## Step 3: Word images preprocessing
Given a zipped folder containing all unpreprocessed word images, the output is a zipped folder containing preprocessed word images. The preoprocessing is described below:
1) We first take the word image, crop any un-needed black space vertically and horizontally
2) We then resize the word image (without distortion) to be 64 pixels in width and 32 pixels in height. If distortionless resizing is not possible without padding, we check and add the required padding to the image.
3) An additional pre-processing step would be to skeletonize the word to get a sharper and thinner outline of the letters. This is required in order to emphasize the morphological similarity of the letters despite the differences in handwriting.

## Step 4: Training the model (DCNNs + Bidirectional LSTMs with CTC Loss)
### 1) Deep Convolutional Neural Networks (DCNNs):
### Introduction:
In classical computer vision techniques, an image is convolved with a number of high-frequency (e.g. Sobel Filters) and low-frequency (e.g. Gaussian Blur Filter) to extract useful information (features) about the objects in an image such as edges and corners. These filters (kernels) have matrices of well-known properties and values. The convolution output is a new image which is called a "feature map" that highlights edges and corners of a targeted object and discards other irrelevant information. Below is an the feature map of an image after applying the Sobel Edge detector:

![edge_detection_sobel](https://user-images.githubusercontent.com/47701869/173190755-6be335b5-6e25-4a02-915e-972200beaf47.png)

However, in computer vision with Deep learning, we treat the filters as learnable parameters that can be optimized for a given dataset over a number of epochs with gradient descent. With deep learning, we do not know the form of the kernel that best suits our problem statement and gives us the best feature map, it is an unknown that we solve for. 
Also, with deep learning, we can treat feature maps as new images to find even more detailed features within them. This is where deep learning gets its name, as a hierarchical structure of great depth can be constructed to extract more complicated features about an object in an image with each layer.

![bb6149549a54b5737420749e95b23de0](https://user-images.githubusercontent.com/47701869/173191020-b0c88714-fb87-4eda-9415-ce8be4c1793b.png)

Both approaches are used in the literature of handwritten OCR to extract the features of the words and characters in a dataset. For example, [this](https://www.researchgate.net/publication/325804874_Recognition_of_Handwritten_Arabic_Characters_using_Histograms_of_Oriented_Gradient_HOG) is a paper using HOG (Histogram of Oriented Gradients) which is a classical learning free technique to extract the features of Arabic characters. Another example is [this paper](https://www.inderscienceonline.com/doi/abs/10.1504/IJISTA.2016.080103), which uses CNNs to extract the features of the characters. You can notice that both papers use an SVM (support Vector Machines) classifier for the transcription step, which means that the transcription phase is learnable in both papers but feature extraction can be learning-free.

### Architecture Used in this project:
Batch size is set to 64 images per batch and each image is 64 pixels wide and 32 pixels tall.
Each batch passes through two CNN layers for feature extraction.

The first CNN layer consists of 32 kernels of size (3,3) and a Max pooling layer of size (2,2). The output is 32 feature maps with dimensions 32 pixels wide by 16 pixels tall. The feature maps are then passed to a ReLu activation function and a batch normalization layer.

The second CNN layer consists of 64 kernels of size (3,3) and a Max pooling layer of size (2,2). The output is 64 feature maps with dimensions 16 pixels wide by 8 pixels tall. The feature maps are then passed to a ReLu activation function and a batch normalization layer.

[Batch normalization](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/) is crucial for accelerating the learning process while reducing the internal covariance shift that occurs when the dataset is not fed in its entirety to the network.

Finally a fully connected dense layer is used to flatten the feature maps into a sequence of numbers optimized for detection by the recurrent neural network (RNN). The dense layer is used with 16 nodes at the output. It receives a flattened input vector of dimensions [1, 64*8] and outputs a flattend vector of dimensions [1,16].
The low number of output nodes in the dense layer is used to avoid overfitting.

### 2) Birdirectional Long-Short Term Memory Recurrent Neural Networks (Bi-LSTM RNNs):

![image](https://user-images.githubusercontent.com/47701869/173202092-8918b188-87f6-4c49-b321-06aaae3edfcc.png)

