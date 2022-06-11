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
4) 
## Step 4: Training the model (DCNNs + Bidirectional LSTMs with CTC Loss)
