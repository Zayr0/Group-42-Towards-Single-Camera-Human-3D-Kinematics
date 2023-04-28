# Group-42-Towards-Single-Camera-Human-3D-Kinematics
In this blog, direct 3D human kinematic estimation (D3KE) proposed in [[1]](#1), as well as a reproduction of the paper we conducted with a slight modification to the proposed network are explained.


# Introduction
**Towards Single Camera Human 3D-Kinematics**, the paper published in 2022, proposed a method - Direct 3D Human Kinematic Estimation (D3KE) by applying a pretrained ResNeXt-50 convolutional network as a backbone and a sequential network to do temporal smoothing and to lift the pose from 2D to 3D. In our reproduction project, we replaced the convolutional ResNeXt-50 by its variant, ResNeXt101, to see whether the performance of the estimation can be improved.
Before diving into the details of the network structures of D3KE and how the backbone network is used in the network, we will briefly explain about ResNet-50 and ResNet-101 which are the basis of ResNeXt-50 and ResNeXt-101. After which, we also explain the motivations of our project by highlighting the difference between ResNeXt-50 and ResNeXt-101.


ResNet-50 was first introduced in the paper **Deep Residual Learning for Image Recognition** published in 2015 by H. Kaiming, X. Shangyu, S. Ren, and S. Jian [[2]](#2). ResNet-50 refers to a specific type of convolutional neural network (CNN) called residual network with 50 layers and weights, consisting of the first convolutional layer followed by a max pooling layer, 48 convolutional layers and an average pooling layer followed by a fully connected layer. Every 3 convolutional layers in the 48 convolutional layers form a residual block where a residual connnection, also known as skip connection is used. Skip connections are the main key feature of ResNet, with which the vanishing gradient problem (VGP) is mitigated by allowing feature maps to bypass intermediate layers, which in turn enables the deeper layers to learn more complex representations without suffering from the VGP. The 48 convolutional layers consists of 4 stages: the first, second, third and the last stages contain 3, 4, 6 and 3 residual blocks, respectively. Similarlly, ResNet-101 is comprised of 101 layers and weights. The structual difference between ResNet-50 and ResNet-101 is the number of residual blocks in the third stage, i.e., 23 blocks instead of 6. As the latter has more layers, it could capture more details of features and data representations. In fact, [[2]](#2) shoewd ResNet-101 has lower error rates on both 10-crop and single-mode tests on ImageNet than ResNet-50 and other networks such as VGG-16 and PReLU-net. However, it should be noted that as a consequence of the increased number of layeres, ResNet-101 may require more computational resources and longer time to train. Furthermore, due to the dame reasons, it might potentially be prone to overfitting.


While ResNet-50 and ResNet-101 certaintly outperform the other network on the image classifications, ResNeXt, introduced in the paper **Aggregated Residual Transformations for Deep Neural Networks** even outperforms the former two networks. The difference between ResNet and ResNeXt is that while ResNet uses residual blocks to learn more diverse features and data representations, ResNeXt is equipped with *cardinality*, which basically means that each residual block is repeated in the new dimention with the cardinality size and grouped convolution is used, which eventually enables ResNeXt to learn even more features and data representations than ResNet. In line with the difference in performance between ResNet-50 and ResNeXt-101, the results given in [[3]](#3) showed that ResNeXt-101 has lower error rates than ResNeXt-50. Since [[1]](#1) did not explore the use of ResNeXt-101 for D3KE, which could potentially lead to better estimation performance as mentioned above. Hence, it is our mission in this reproduction proect to equip D3KE with ResNeXt-101 and test it.

With our mouthful explanations for ResNet, ResNeXt and our motivations having been explained, it is time to dive into the details of the network structures of D3KE and how the backbone network is used in the network.

# How does D3KE work?
![image](https://user-images.githubusercontent.com/104576899/234027838-11b4e92e-5fe1-4a5f-abb9-3866960eaa8a.png)

Traditional pose estimation methods tend to adopt a two-stepped approach to estimating joint angles. First, a network is used to identify keypoints of the human body in theimage like hands, elbows, knees, joints, etc. These keypoints are then used to regress the pose of the body as well as the kinematics. This method seems to draw inspiration from the traditional method of Optical Motion Capture (OMC), where markers placed on the body are used to reconstruct the pose. However, this method has its drawbacks. Aside from the obvious problem that labelling all the keypoints in every image is a laborious and time-consuming process, there is also the issue that the people performing labelling tend to misidentfy the exact location of the keypoints in the image by virtue of not being experts. This can be a problem since the resulting errors in marker predictions can propogate to the next step and lead to significant deviation from the ground-truth pose.

D3KE gets around this issue by avoiding decoupling the two steps, instead regressing the pose directly from the input image. The authors justify this choice by pointing out that deep learning systems have managed to outperform multi-step approaches by implicitly learning the individual steps through end-to-end learning. This means that the network can make more accurate pose predictions by training the whole network, unlike in the previous method where the difference between the predicted and the ground-truth pose could not be used to tune the network parameters.

# Method

## Network structure

## Loss function

$$
L = \lambda_1 L_{joint} + \lambda_2 L_{marker} + \lambda_3 L_{body} + \lambda_4 L_{angle}
$$

# Results

# Conclusion

# Running and commands

# Reference


## References
<a id="1">[1]</a> 
Bittner M, Yang WT, Zhang X, Seth A, van Gemert J, van der Helm FC. Towards Single Camera Human 3D-Kinematics. Sensors. 2022 Dec 28;23(1):341. https://doi.org/10.3390/s23010341

<a id="2">[2]</a> 
He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
https://doi.org/10.48550/arXiv.1512.03385

<a id="3">[3]</a> 
Xie S, Girshick R, Dollár P, Tu Z, He K. Aggregated residual transformations for deep neural networks. InProceedings of the IEEE conference on computer vision and pattern recognition 2017 (pp. 1492-1500).
https://doi.org/10.48550/arXiv.1611.05431

