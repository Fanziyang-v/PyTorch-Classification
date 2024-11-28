# Classification in PyTorch



## Implementations



### AlexNet



#### Authors

Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton

#### Abstract

We trained a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 39.7\% and 18.9\% which is considerably better than the previous state-of-the-art results. The neural network, which has 60 million parameters and 500,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two globally connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of convolutional nets. To reduce overfitting in the globally connected layers we employed a new regularization method that proved to be very effective.

[[Paper]](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)[[Code]](./models/alexnet.py)

Since the GTX 580 GPU has only 3GB of memory, a single GPU could not store the entire AlexNet. The implementation here is a simplified version of the original.

![AlexNet](./assets/AlexNet.png)

