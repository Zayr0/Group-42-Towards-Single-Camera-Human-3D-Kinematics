# Group-42-Towards-Single-Camera-Human-3D-Kinematics
In this blog, direct 3D human kinematic estimation (D3KE) proposed in [[1]](#1), as well as a reproduction of the paper we conducted with a slight modification to the proposed network are explained. As for the criteria of the reproduction project, we chose *New algorithm variant*.


# Introduction
**Towards Single Camera Human 3D-Kinematics**, the paper published in 2022, proposed a method - Direct 3D Human Kinematic Estimation (D3KE) by applying a pretrained ResNeXt-50 convolutional network as the convolutional backbone and a sequential network to do temporal smoothing and to lift the pose from 2D to 3D. In our reproduction project, we replaced the convolutional ResNeXt-50 by its variant, ResNeXt101, to see whether the performance of the estimation can be improved.
Before diving into the details of the network structures of D3KE and how the backbone network is used in the network, we will briefly explain about ResNet-50 and ResNet-101 which are the basis of ResNeXt-50 and ResNeXt-101. After which, we also explain the motivations of our project by highlighting the difference between ResNeXt-50 and ResNeXt-101.


ResNet-50 was first introduced in the paper **Deep Residual Learning for Image Recognition** published in 2015 by H. Kaiming, X. Shangyu, S. Ren, and S. Jian [[2]](#2). ResNet-50 refers to a specific type of convolutional neural network (CNN) called residual network with 50 layers and weights, consisting of the first convolutional layer followed by a max pooling layer, 48 convolutional layers and an average pooling layer followed by a fully connected layer. Every 3 convolutional layers in the 48 convolutional layers form a residual block where a residual connection, also known as skip connection is used. Skip connections are the main key feature of ResNet, with which the vanishing gradient problem (VGP) is mitigated by allowing feature maps to bypass intermediate layers, which in turn enables the deeper layers to learn more complex representations without suffering from the VGP. The 48 convolutional layers consist of 4 stages: the first, second, third and the last stages contain 3, 4, 6 and 3 residual blocks, respectively. Similarly, ResNet-101 is comprised of 101 layers and weights. The structural difference between ResNet-50 and ResNet-101 is the number of residual blocks in the third stage, i.e., 23 blocks instead of 6. As the latter has more layers, it could capture more details of features and data representations. In fact, [[2]](#2)showed ResNet-101 has lower error rates on both 10-crop and single-mode tests on ImageNet than ResNet-50 and other networks such as VGG-16 and PReLU-net. However, it should be noted that as a consequence of the increased number of layers, ResNet-101 may require more computational resources and longer time to train. Furthermore, due to the same reasons, it might potentially be prone to overfitting.

While ResNet-50 and ResNet-101 certainly outperform the other network on the image classifications, ResNeXt, introduced in the paper **Aggregated Residual Transformations for Deep Neural Networks** even outperforms the former two networks. The difference between ResNet and ResNeXt is that while ResNet uses residual blocks to learn more diverse features and data representations, ResNeXt is equipped with *cardinality*, which basically means that each residual block is repeated in the new dimension with the cardinality size and grouped convolution is used, which eventually enables ResNeXt to learn even more features and data representations than ResNet. In line with the difference in performance between ResNet-50 and ResNeXt-101, the results given in [[3]](#3) showed that ResNeXt-101 has lower error rates than ResNeXt-50. Since [[1]](#1) did not explore the use of ResNeXt-101 for D3KE, which could potentially lead to better estimation performance as mentioned above. Hence, it is our mission in this reproduction project to equip D3KE with ResNeXt-101 and test it.

With our mouthful explanations for ResNet, ResNeXt and our motivations having been explained, it is time to dive into the details of the network structures of D3KE and how the backbone network is used in the network.

# How does D3KE work?
![['Overview of the proposed direct 3D human kinematics estimation’ (D3KE)'[[1]](#1)]](https://user-images.githubusercontent.com/104576899/234027838-11b4e92e-5fe1-4a5f-abb9-3866960eaa8a.png)
**Figure. 1** *"Overview of the proposed direct 3D human kinematics estimation (D3KE)"[[1]](#1)* <br>

Traditional pose estimation methods tend to adopt a two-stepped approach to estimating joint angles. First, a network is used to identify key points (centers of major joints) of the human body in the image such as the shoulders, hips and knees. These keypoints are then used to regress the pose of the body as well as the kinematics using a musculoskeletal model. This method seems to draw inspiration from the traditional method of Optical Motion Capture (OMC), where markers placed on the body are used to reconstruct the pose. However, this method has its drawbacks. First, since the maker position estimation and the pose estimation are decoupled, the errors in the marker position estimates propagate to the pose estimates. The propagated errors in the pose estimates cannot be corrected by using the ground-truth error of the pose estimates since the pose estimation stage does not consider/include the maker position estimation. These uncorrectable errors across the stages can lower the accuracy of the pose and thus other subsequent estimation stages such as the kinematics estimation. In addition to the obvious problem that labeling all the key points in every image is a laborious and time-consuming process, there is also the issue that people performing labeling of the ground-truth for the key points tend to misidentify the exact location of the key points in the image due to lack of expertise. This can be a problem as the bias in the ground-truth of marker position causes errors in the maker position predictions and such errors can again, propagate to the next step and lead to significant deviation from the ground-truth pose.

D3KE gets around this issue by avoiding decoupling the two steps, i.e., instead of estimating the pose of the body in the multi-steps, the pose is estimated by using a single network that learns the underlying relation between the marker position and the musculoskeletal model (body scaling factors, and joint angles and positions). The authors justify this choice by pointing out that deep learning systems have managed to outperform multi-step approaches by implicitly learning the individual steps through end-to-end learning. This means that instead of having explicit representations of data such as the marker position estimates during the pose estimations, the single deep network only considers the end-to-end loss. The consolidation of the stages enables the network to correct errors within the network, thereby no propagation error as there are no decoupled stages. This of course, means that the network can make more accurate pose predictions by training the whole network, unlike in the ‘multi-steps’ method where the difference between the predicted and the ground-truth pose could not be used to tune the network parameters.

# Method

## Network structure
The network used in the papers is composed of two main parts. First, a convolutional neural network with ResNeXt-50 as a backbone is used to predict the markers and the joint angles. A sequential network is then used to exploit the temporal differences across the data. This allows for the predictions to be “lifted” into 3D. Three different sequential networks were investigated for pose and angle refinement: a LSTM, a Temporal Convolutional Network (TCN) and a Transformer. Both networks also contain a layer which allows the networks to perform the kinematic transformations of the musculoskeletal model. This allows the resulting pose predictions to be taken into account during training.

## Loss function
The loss function is composed of four terms as seen below:

$$L = \lambda_1 L_{joint} + \lambda_2 L_{marker} + \lambda_3 L_{body} + \lambda_4 L_{angle}$$

where $\lambda_1$, $\lambda_2$, $\lambda_3$ and $\lambda_4$ are weights of losses, $L_{joint}$ is the loss of joint position, $L_{marker}$ is the loss of maker position, $L_{body}$ is the loss of body scales and $L_{angle}$ is the loss of joint angles.

As the marker loss generates loss based on the distance of the marker to the estimated position of the marker, it does an indirect loss estimation of the angles as well, however they are now weighted based on the length of the body parts. As the combination of these lengths and angles of the body eventually determine the marker position. This means that the marker loss represents the cumulative loss of the angles loss.

The joint and marker losses are calculated as L1 losses relative to the root position

$$L = ||(\hat(y) - (\hat y)_{root}) - (y - y_{root})||_1$$

Here, the root position is the position of the pelvis.
In order to impose the underlying relations and constraints of the musculoskeletal model, a skeletal-model layer is added after the network during training. The network converts the predicted joint angles into marker positions and contains no learnable parameters.

# Results and Conclusion
The first major problem with trying to replicate or adapt this project is that the data referenced and needed to generate the dataset is very big, about 50-60 GB. And the data generated from it is around 500 GB. The google cloud storage has 100GB available, and thus the same results as in the paper can not be replicated this way. Because of this it would be very hard to compare the actual accuracy of the model running on google cloud to the results in the paper.
Because of this training a new reasonable model was not possible.

The second major issue is that the checkpoints provided by Marian Bittner did not match up with the OpenSim TreeLayer models. This means that the run_inference.py script (which checks the models accuracy) did not run. Because of this no comparison can be made between the pretrained ResNeXt50 and ResNeXt101.

All in all, due to the issues described above, we could not manage to reproduce D3KE with ResNeXt 50 that was originally used in [[1]](#1), let alone with ResNeXt-101 to investigate whether the performance of the estimation improves or not, which is the aim of our project. With regard to the difference in number of layers between the two pretrained backbone networks, some might argue that the increase of the layers might potentially lead the network to overfit. Nevertheless, under an assumption that the overfitting is less likely to occur with ResNeXt-101, we believe that the kinematic estimation by D3KE with ResNeXt-101 could have higher accuracy in comparison to the one with ResNeXt-50. 

As can be seen in the table below from [[3]](#3)’s github page, the errors of the different ResNet structures can be observed. We expect that these errors will propagate in the final result of the D3KE model. The error of ResNet-101 is not that much lower than the error of ResNet-50, but will take a lot longer to run due to the layer depth difference between the two. This is why we suspect that the switch to a 101 layer will not have a great positive effect on the overall D3KE model, depending on the application of course. For a live high fps operation, it might even be worth to downgrade to a 34 or 18 layer depth ResNet structure.

ImageNet 1-crop error rates (224x224)

| Network       | Top-1 error | Top-5 error |
| ------------- | ----------- | ----------- |
| ResNet-18     | 30.43       | 10.76       |
| ResNet-34     | 26.73       | 8.74        |
| ResNet-50     | 24.01       | 7.02        |
| ResNet-101    | 22.44       | 6.21        |
| ResNet-152    | 22.16       | 6.16        |
| ResNet-200    | 21.66       | 5.79        |


## References
<a id="1">[1]</a> 
Bittner M, Yang WT, Zhang X, Seth A, van Gemert J, van der Helm FC. Towards Single Camera Human 3D-Kinematics. Sensors. 2022 Dec 28;23(1):341. https://doi.org/10.3390/s23010341

<a id="2">[2]</a> 
He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
https://doi.org/10.48550/arXiv.1512.03385

<a id="3">[3]</a> 
Xie S, Girshick R, Dollár P, Tu Z, He K. Aggregated residual transformations for deep neural networks. InProceedings of the IEEE conference on computer vision and pattern recognition 2017 (pp. 1492-1500).
https://doi.org/10.48550/arXiv.1611.05431
https://github.com/facebookarchive/fb.resnet.torch/tree/master/pretrained
