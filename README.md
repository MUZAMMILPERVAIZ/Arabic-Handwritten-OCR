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
In classical computer vision techniques, an image is convolved with a number of high-frequency (e.g. Sobel Filters) and low-frequency (e.g. Gaussian Blur Filter) to extract useful information (features) about the objects in an image such as edges and corners. These filters (kernels) have matrices of well-known properties and values. The convolution output is a new image which is called a "feature map" that highlights edges and corners of a targeted object and discards other irrelevant information.

However, in computer vision with Deep learning, we treat the filters as learnable parameters that can be optimized for a given dataset over a number of epochs with gradient descent. With deep learning, we do not know the form of the kernel that best suits our problem statement and gives us the best feature map, it is an unknown that we solve for. 
Also, with deep learning, we can treat feature maps as new images to find even more detailed features within them. This is where deep learning gets its name, as a hierarchical structure of great depth can be constructed to extract more complicated features about an object in an image with each layer.

Both approaches are used in the literature of handwritten OCR to extract the features of the words and characters in a dataset. For example, [this](https://www.researchgate.net/publication/325804874_Recognition_of_Handwritten_Arabic_Characters_using_Histograms_of_Oriented_Gradient_HOG) is a paper using HOG (Histogram of Oriented Gradients) which is a classical learning free technique to extract the features of Arabic characters. Another example is [this paper](https://www.inderscienceonline.com/doi/abs/10.1504/IJISTA.2016.080103), which uses CNNs to extract the features of the characters. You can notice that both papers use an SVM (support Vector Machines) classifier for the transcription step, which means that the transcription phase is learnable in both papers but feature extraction can be learning-free.

### Architecture Used in this project:
Batch size is set to 64 images per batch and each image is 64 pixels wide and 32 pixels tall.
Each batch passes through two CNN layers for feature extraction.

The first CNN layer consists of 32 kernels of size (3,3) and a Max pooling layer of size (2,2). The output is 32 feature maps with dimensions 32 pixels wide by 16 pixels tall. The feature maps are then passed to a ReLu activation function and a batch normalization layer.

The second CNN layer consists of 64 kernels of size (3,3) and a Max pooling layer of size (2,2). The output is 64 feature maps with dimensions 16 pixels wide by 8 pixels tall. The feature maps are then passed to a ReLu activation function and a batch normalization layer.

[Batch normalization](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/) is crucial for accelerating the learning process while reducing the internal covariance shift that occurs when the dataset is not fed in its entirety to the network.

Finally a fully connected dense layer is used to flatten the feature maps into a sequence of numbers optimized for detection by the recurrent neural network (RNN). The dense layer is used with 16 nodes at the output. It receives a flattened input vector of dimensions [1, 64*8] and outputs a flattend vector of dimensions [1,16].
The low number of output nodes in the dense layer is used to avoid overfitting.
A regularization technique is used by adding a Dropout layer after the fully connected layer.

### 2) Birdirectional Long-Short Term Memory Recurrent Neural Networks (Bi-LSTM RNNs):

### Introduction:
Any sequence based data has the property that current data entries depend on the data entries prior to them in time, space or any other measured dimension. In some special cases, the current data entries also depend on the upcoming entries in the sequence. 

For sequence based detection problems, the normal feed-forward neural network fails to capture the backward and forward dependencies of the data entries, because they focus on the current data entry only. Therefore, feed-forward NNs perform poorly in such problems.

Recurrent Neural Networks (RNNs) provide pathways of dependence between the current data entries and the the entries prior to and after it.

RNNs are particularly prone to exploding/vanishing gradients, due to the fact that unrolling the network to perform Back Propagation Through Time (BPTT) means that many partial derivatives are multiplied together which increases the probability of vanishing gradients especially with saturating activation functions.
The problem of vanishing gradients is mitigated by using Long Short Term Memory (LSTM) gated RNN which is shown in the diagram below:

![LSTM](https://user-images.githubusercontent.com/47701869/173226842-8128e0b3-cb8c-44d2-a185-b4f4ccf8272e.png)

In LSTM networks, there is an input gate (denoted by i(t)), a forget gate (denoted by f(t)), and an output gate (denoted by o(t)) as well as a cell state (denoted by c(t)).

At the forget gate, the sigmoid activation function is applied to the current inputs x(t) concatinated with the hidden layer vectors from a previous step in the sequence h(t-1) (multiplied by the weights).

If the result of the activation function is close to zero, the current information is considered unimportant, and information with sigmoid values closer to 1 is considered important.

Similarly, if we trace the equation for the current cell state we will find that it is a function that depends on both i(t) and f(t) and decides whether to take the previous information into consideration or not. Finally, the current cell state is transformed into the range [-1,1] by the tanh function affecting the output of the cell o(t).

This gated behavior limits long chains of dependencies on previous data entries and therefore reduces the vanishing gradients problem.


![image](https://user-images.githubusercontent.com/47701869/173202092-8918b188-87f6-4c49-b321-06aaae3edfcc.png)

### Architecture Used in this project:
The output of the fully connected dense layer, delivering the output of the previous CNN layers, is fed to a batch normalization layer and then to a single Bidirectional-LSTM with 256 LSTM Units in both directions (i.e. 128 LSTM unit in either direction). 

To add a regularization effect to the network, a dropout ratio of 0.35 is chosen.


### 3) Connectionist Temporal Classification (CTC) Loss Function:

A neural network with CTC loss has a softmax output layer with one more unit than there are characters in the vocabulary (L).

The activations of the first L units are interpreted as the probabilities of observing the corresponding characters at particular times in th sequence. The activation of the extra unit is the probability of observing a ‘blank’, or no character.

Together, these outputs units define the probabilities of all possible ways of aligning all possible character sequences with the input sequence.

The total probability of any one character sequence can then be found by summing the probabilities of its different alignments.

Let y be the output of an RNN given to the softmax layer, the softmax outputs can be denoted by $ y^t_k $ which is interpreted as the probability of observing character k at time t,  which defines a distribution over the set $L^{'T}$ of length T sequences over the alphabet $ L'= L U blank$ is:

$$ y^t_k = p( \pi | x ) = \prod_{t=1}^{T} y^t_{\pi}, \forall \pi \in  L^{'T}  $$

The next step is to define a many characters-to-one character map $ B:  L^{'T} \rightarrow L^{\leq T} $ such that $L^{\leq T}$ are the set of final characters predicted with length equal to or less than the input sequence length T.

This is done simply by removing all blanks and repeated characters from the predictions (e.g. if the blank character is denoted by $ \eta$ then $ B(a \eta ab \eta) = B(\eta aa \eta \eta abb) = aab $ ).

Finally, the mapping B is used to define the conditional probability of a given character $ l \in L^{\leq T} $ as the sum of all probabilities of the possible labels associated with this character

$$ p(l | x) = \sum_{\pi \in B^{-1}(l)} p( \pi | x ) $$

![CTC](https://user-images.githubusercontent.com/47701869/173327644-bbf5ca8e-0d35-4610-aa5f-17e9e3b8243f.PNG)

Given the above formulation, the output of the classifier should be the most probable labelling for the input sequence:
$$ h(x)=max(p(l | X))$$

The task of finding this most probable labelling is known as decoding. The most efficient decoding algorithm is the prefix/beam search decoding algorithm shown in the figure below:

![beam_search](https://user-images.githubusercontent.com/47701869/173332378-055e6adb-6d9e-45ba-a6f5-66dbb8f1a862.PNG)

Each node either ends (‘e’) or extends the prefix at its parent node. The number above an extending node is the total probability of all labellings beginning with that
prefix.
The number above an end node is the probability of the single labelling ending at its parent.
At every iteration the extensions of the most probable remaining prefix are explored.
Search ends when a single labelling (here ‘XY’) is more probable than any remaining prefix.
For futher reading, check this [link](https://distill.pub/2017/ctc/) and this [paper](https://dl.acm.org/doi/10.1145/1143844.1143891) out.

### Architecture Used in this project:
A softmax layer with output nodes equal to the characters in the vocabulary+1 is defined after the Bi-LSTM layer.
The output of this softmax layer is used as an input to a CTC loss function keras.backend.ctc_batch_cost 
The CTC cost function is the maximization of $ p(l | x) $ or conversely the minimization of $log(p(l | x))$
The most likely sequence of character predictions is decoded by using beam search algorithm with a beam width of 100 nodes.

## Step 5: Evaluating the model

A dataset of 5160 images was augmented to obtain a dataset with double the size (15480 images).

The training-testing split was performed so that 80% of the dataset is used for the training and the testing and validation utilizes 20%.

The total number of epochs was set to 450 epochs which is directly dependent on the choice of the optimizer and the learning rate.

ADAM optimizer is used with an exponential decay scheduler for the learning rate starting at a maximum value of 0.0001. A small value of the learning rate is chosen beause larger starting values lead to an earlier onset of overfitting before reaching an acceptable accuracy.

At the final epoch, the training loss reached 1.5730 whereas the validation loss was 1.1795.

![download (7)](https://user-images.githubusercontent.com/47701869/173602635-580006ed-97c7-4d69-ab27-4be28f19e304.jpg)

We use the Character Error Rate (CER) metric to evaluate the model over the test dataset.

The Character Error Rate (CER) compares, for a given transcription, the total number of characters (n), including spaces, to the minimum number of insertions (i), substitutions (s) and deletions (d) of characters that are required to obtain the Ground Truth result. The formula to calculate CER is as follows: CER = [ (i + s + d) / n ]*100

Insertions: Extra characters that the model predicts which are not in the ground truth label (example of a label ='cat', example of insertion='caat')

Deletions: Missing characters that the model doesn't predict (correctly or otherwise) which are in the ground truth label (example of a label ='cat', example of deletion='at')

substitutions: characters which are wrongly predicted by the model (example of a label ='cat', example of substitution='bat')

By evaluating the CER for the model over the testing dataset, we found that CER is is around 13% meaning that the model successfully predicted 87% of the characters in all ground truth labels of the testing set (including punctuation marks and spaces).

Here is a visualization of the model's predictions versus the image samples for reference:

![download (5)](https://user-images.githubusercontent.com/47701869/173499677-c99d44f6-3b5b-4625-be05-326e8168a526.jpg)


## Future Works Regarding this project:

### 1) Increasing the Dataset size:

So far, we have used a very small portion of the paragraph images provided by the [KHATT dataset](http://khatt.ideas2serve.net/index.php) and therefore our dataset is limited.

KHATT dataset provides 1400 paragraph images for training and 300 paragraph image for testing and validation (total 2000 images). However, we used less than 100 images to obtain our dataset because labels for words and subwords were unavailable.

We assume that the model will perform better by an increased and diversified dataset, given that it is already showing promising results with a small dataset

### 2) Tuning the model's hyperparameters

An example of the model's hyperparameters are the number of CNN layers, the number of kernels within eah CNN layer, the number of the nodes in the intermediate dense layer and the dropout ratio, the number of bidirectional RNNs used and the number of LSTM units inside each RNN.

All of these are parameters which control the architecture and complexity of the model. We found that for highly complicated models, overfitting is achieved almost on the onset of training whereas less complicated models take longer times to converge. 

It is worth investigating (systematically) the relationship between these hyperparameters to find the optimum combination of values

### 3) Enhancing segmentation technique to obtain words and subwords

The currently used segmentation technique cannot handle a paragraph image where the distance between each 2 consecutive lines is not consistent across the liness themselves. An example of this case is shown below:

![2-Figure1-1 (1)](https://user-images.githubusercontent.com/47701869/173219295-ad223b48-06fd-433f-9088-0e2b43c3f600.png)

Also the algorithm struggles with highly noisy and/or overlapping paragraph images, for example:

![A-sample-handwritten-Arabic-document-image-including-rule-lines](https://user-images.githubusercontent.com/47701869/173219333-de7b5d18-a7de-4da9-957e-fc0f41ffeaee.png)

Therefore, a more general and robust segmentation technique must be investigated for these cases.


