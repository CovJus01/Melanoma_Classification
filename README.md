# Melanoma_Classification
A model trained to classify various skin blemishes. This readme will explain some of the process and design intentions. It will also have an update of the progress and decisions made as I learn to modify and optimize the learning model to work better. The project is an ongoing project as I continue to learn more about Tensorflow, and about neural networks.

## Data retrieval and alteration
The dataset is one from kaggle and consists of the following structure:
#### Training set
2000 images(1372 nevus, 374 melanoma, 254 seborrheic)
#### Validation set
150 images(78 nevus, 30 melanoma, 42 seborrheic)
#### Test set
600 images(393 nevus, 117 melanoma, 90 seborrheic)\

To retrieve the data we resize the images to be 200x200 and convert them to greyscale. The result is a (2000, 40 000) shaped array of features, and a (2000, 1) set of labels.
The labels are also each set their own value as we want to use a sparse categorical cross entropy to evaluate the cost of the error in the optimzation. These values range from 0-2

The data is then also regularized using a standard Z-score normilization function. We also shuffle the dataset so that the model is exposed to different examples throughout its training cycle.

## Model Design
Currently I am using a regular Dense neural network. It initially had the following structure:
-  Layer 1: 250 units with ReLU activation
-  Layer 2: 200 units with ReLU activation
-  Layer 3: 150 units with ReLU activation
-  Layer 4: 50 units with ReLU activation
-  Layer 5: 25 units with ReLU activation
-  Layer 6: 10 units with ReLU activation
-  Layer 7: 3 units with linear activation\
This model used the Sparse Categorical Cross Entropy to calculate it's loss and used the data from the logits.
This model ran into the problem of settling on always predicting the most common value, 0 leading to no change in the resulting prediction

The second iteration was simplified and takes on this structure (Current Model):
-  Layer 1: 25 units with ReLU activation
-  Layer 2: 15 units with ReLU activation
-  Layer 3: 7 units with ReLU activation
-  Layer 4: 3 units with linear activation\
This is the current model that I am currently using to analyze the data and verify that it is working as intended. I am currently updating the process so that I can analyze the training vs validation error and many other attributes that will help me in optimizing the way I create the neural network.

Training Acc (Blue) vs Validation Acc (Red):
<img title="Iteration2Chart" alt="Chart supposed to be here" src="./images/1.png">
# To be continued...
