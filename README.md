# Person Re-Identification Project

## Table of Contents
1. [Introduction](#introduction)
2. [Project Description](#project-description)
3. [Prerequisites](#prerequisites)
4. [Step 1: Data Collection and Preprocessing](#step-1-data-collection-and-preprocessing)
5. [Step 2: Person Detection and Tracking](#step-2-person-detection-and-tracking)
6. [Step 3: Feature Extraction](#step-3-feature-extraction)
7. [Step 4: Person Re-Identification Model](#step-4-person-re-identification-model)
8. [Step 5: Visualization and Demonstration](#step-5-visualization-and-demonstration)


## Introduction

Identifying and tracking individuals is a complex task for machines, involving:

1. Detecting people in camera frames.
2. Recognizing them.
3. Efficiently searching and comparing with a database.
4. Continuous tracking.
5. Balancing speed and accuracy in the system.

Taking into Consideration each factor - yolo and facenet based pipeline along with clustering for efficient search.

![image](https://github.com/Dharssini/LiveSitter_PersonReid/assets/73426538/9d968016-40aa-422d-952d-97ef8f0c2663)

## Project Description

The project consists of five main steps, each contributing to the development of a person re-identification system:

### Prerequisites

Before getting started with the project, make sure you have the following prerequisites:

- Python 3.6+
- PyTorch
- OpenCV
- YOLO or Faster R-CNN (for person detection)
- Relevant datasets of publicly available CCTV footage

## Step 1: Data Collection and Preprocessing (10 points)

1. Collect a dataset of publicly available CCTV footage that includes multiple camera views capturing people walking.
2. Preprocess the video data into a format suitable for model training.
3. Document the data collection and preprocessing process in your README.md file.

## Step 2: Person Detection and Tracking (20 points)

YOLOv3 is very fast, flexible and can run in real time (~30-40 fps) and this third version is quite accurate in
detection. Based on the Speed-Accuracy tradeoff of all the methods we decided to use YOLOv3 for person
detection as we need real time performance for surveillance with considerable accuracy.

Broadly speaking, Face detection involves the detection of faces in an image as well as localisation of it by
determining the bounding box coordinates. A face cannot be recognised if it is not detected first. Facial data
show a high degree of variability, and it is difficult to detect one using handcrafted filters. So machine learning
methods need to be applied to detect various patterns.
I have used MTCNN -Multi-task Cascaded Convolutional Networks for this.

Several deep learning methods have been developed and demonstrated for face detection or, in general,
object detection like RCNN, fast-RCNN or YOLO. “Multi-task Cascaded Convolutional Neural Network”, or MTCNN for short, is one of the most popular approaches and current state of the art model.The MTCNN model is also capable of locating facial
landmarks such as eyes and mouth. The Convolutional Neural Networks (CNNs) have achieved a remarkable performance in many computer vision tasks. The network is a cascaded structure of three networks.
1. Proposal Network (P-Net)
2. Refine Network (R-Net)
3. Output Network or O-Net
Before providing the image to the network, it is rescaled to different scales to build an image pyramid.

 KCF Tracking Algorithm implemented in OpenCV [Paper - https://arxiv.org/abs/1404.7584]. It used multiple techniques including the concepts of histogram along with kernelized correlation to track the required entity between subsequent frames. It is a good technique which handles speed-Accuracy tradeoff well.


## Step 3: Feature Extraction (15 points)

we want to create different copies of the same image in different sizes to search for different sized faces within the image.
We have a 12 x 12 stage 1 kernel for each scaled copy that will go through every part of the image, scanning for faces. It starts in the top left corner, a section of the image from (0,0) to (12,12). This portion of the image is passed to P-Net, which returns the coordinates of a bounding box if it notices a face. Then, it would repeat that process with sections (0+2a,0+2b) to (12+2a, 12+2b), shifting the 12 x 12 kernel at a stride of 2.
The weights and biases of P-Net have been trained so that it outputs a relatively accurate bounding box for every 12 x 12 kernel. The P-Net output list is parsed to get confidence levels for each bounding box and delete the boxes with lower confidence. After selecting the boxes with higher confidence, the coordinate system is standardised, converting all the coordinate systems to that of the actual, “un-scaled” image. However, there are still many bounding boxes left, and a lot of them overlap. Non-Maximum Suppression, or NMS, is a method that reduces the number of bounding boxes. NMS is conducted by first sorting the bounding boxes by their confidence or score. Calculate each kernel area and the overlapping area between each kernel and the kernel with the highest score. The kernels that overlap a lot with the high-scoring kernel get deleted.

For every bounding box, create an array of the same size and copy the pixel values to the new array, resize them to 24 x 24 pixels and normalise them between -1 and 1. All these 24x24 normalised arrays are then fed into the R-Net. The output of R-Net is the coordinates of the new, more accurate bounding boxes and the confidence level of each of these bounding boxes.
The final stage is the O-Net the bounding boxes obtained from R-Net are passed into the O-Net after resizing to 48x48; the O-Net computes the location of 5 facial landmarks - eyes, nose, and endpoints of the mouth.
The loss functions used with this network are the cross-entropy loss function for the face binary classification.
The bounding box regression and the facial landmark localisation Euclidean loss have been utilised.
Undoubtedly deep learning has a significant advantage over shallow learning algorithms. This network can detect face even with some occlusions and lighting inconsistencies that other algorithms like haar cascades could not. However, this accuracy comes with the cost that this algorithm is relatively slow on individual frames; MTCNN barely runs at around ten frames per second. However, it can be fastened with the use of GPUs.
** I used a pre-trained MTCNN model implemented in pytorch.** This model was trained on a dataset of millions of images.

## Step 4: Person Re-Identification Model (20 points)

![image](https://github.com/Dharssini/LiveSitter_PersonReid/assets/73426538/6bf4ce66-1221-40c5-b9d5-c86d850edec4)

Trained a Face Recognition Siamese Model from Scratch on LFW Dataset it was able to perform extremely well on this data but not able to generalize well. For performance comparison we used the CMC(Rankwise comparison) plot to show the deterioration in generalization.

Face recognition has been an active area of research in recent years in deep learning in computer vision.
Deep learning methods can leverage enormous datasets of faces and learn rich and compact representations
of faces, allowing modern models to perform as well and later to outperform the face recognition capabilities of
humans. There are perhaps four milestone systems on deep learning for face recognition that drove these
innovations: DeepFace, the DeepID series of systems, VGGFace, and FaceNet.
I used a ResNet-34 based siamese network. ResNets are specially designed networks with many convolutional layers; these solve vanishing gradients problem in deep learning. For specific problems like face recognition and signature verification, we cannot always rely on getting more data to solve this kind of tasks, a new type of neural network architecture called Siamese Networks. It uses only a few numbers of images to get better predictions. The ability to learn from very little data made Siamese networks more popular in recent years.
A Siamese Neural Network is a class of neural network architectures that contain two or more identical subnetworks. They have the same configuration with the same parameters and weights. Parameter updating is mirrored across both sub-networks. It is used to find the similarity of the inputs by comparing their feature vectors. Traditionally, a neural network learns to predict multiple classes. So it poses a problem when we need to add/remove new classes to the data. In this case, we have to update the neural network and retrain it on the whole dataset. Also, deep neural networks need a large volume of data to train. Siamese Networks, on the other hand, learn a similarity function. This way, it can classify new classes of data without training the network again.
There are several types of loss functions used in siamese networks:
1. Contrastive Loss: distance-based loss as opposed to more conventional error-prediction losses. This
loss is used to learn embeddings in which two similar points have a low Euclidean distance and two
different points have a sizeable Euclidean distance.
2. Triplet Loss: It is a loss function where an anchor point is compared to a similar point and a different
point. The distance from the anchor to the similar input is minimised, and the distance from the anchor
to the different point is maximised.

The ResNet-34 based model we used was pre-trained on more than 3 million faces. This dataset is derived
from several datasets: the face scrub dataset and the Visual Geometry Group dataset. The total number of
individual people were 7485. The model was also tested against the LFW benchmark dataset and gave an
excellent accuracy.

## Step 5: Visualization and Demonstration (10 points)

Object Detection:

![image](https://github.com/Dharssini/LiveSitter_PersonReid/assets/73426538/dca4d4ea-a04a-4268-a690-b9f02c6811bc)

Face Detection:

![image](https://github.com/Dharssini/LiveSitter_PersonReid/assets/73426538/6b038abb-b4e7-4b9e-95e0-02caf6c1bea2)

Siamese Face detection:

![__results___7_0](https://github.com/Dharssini/LiveSitter_PersonReid/assets/73426538/6b640ca2-04a2-48a5-9d7a-afe9dfaaa0a8)


LiveSitter_PersonReid
