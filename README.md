# Group-42-Towards-Single-Camera-Human-3D-Kinematics
In this blog, direct 3D human kinematic estimation (D3KE) proposed in [reference of paper], as well as a reproduction of the paper we conducted with a slight modification to the proposed network are explained.


# Introduction
**Towards Single Camera Human 3D-Kinematics**, the paper published in 2022, proposed a method - Direct 3D Human Kinematic Estimation (D3KE) by applying a pretrained ResNeXt-50 convolutional network as a backbone and a sequential network to do temporal smoothing and to lift the pose from 2D to 3D. In our reproduction project, we replaced the convolutional ResNeXt-50 by its variant, ResNeXt101, to see whether the performance of the estimation can be improved.
Before diving into the details of the network structures of D3KE and how the backbone network is used in the network, we will briefly explain about ResNet-50 and ResNet-101 which are the basis of ResNeXt-50 and ResNeXt-101. After which, we also explain the motivations of our project by highlighting the difference between ResNeXt-50 and ResNeXt-101.
ResNet-50 was first introduced in the paper **Deep Residual Learning for Image Recognition** published in 2015 by H. Kaiming, X. Shangyu, S. Ren, and S. Jian. ResNet-50 refers to a specific type of convolutional neural network (CNN) called residual network with 50 layers and weights, consisting of the first convolutional layer followed by a max pooling layer, 48 convolutional layers and an average pooling layer followed by a fully connected layer. Every 3 convolutional layers in the 48 convolutional layers form a residual block where a residual connnection, also known as skip connection is used. Skip connections are the main key feature of ResNet, with which the vanishing gradient problem (VGP) is mitigated by allowing feature maps to bypass intermediate layers, which in turn enables the deeper layers to learn more complex representations without suffering from the VGP. The 48 convolutional layers consists of 4 stages: the first, second, third and the last stages contain 3, 4, 6 and 3 residual blocks, respectively. Similarlly, ResNet-101 is comprised of 101 layers and weights. The structual difference between ResNet-50 and ResNet-101 is the number of residual blocks in the third stage, i.e., 23 blocks instead of 6. As the latter has more layers, it could capture more details of features and data representations. In fact, [paper for resnet] shoewd ResNet-101 has lower error rates on both 10-crop and single-mode tests on ImageNet than ResNet-50 and other networks such as VGG-16 and PReLU-net. However, it should be noted that as a consequence of the increased number of layeres, ResNet-101 may require more computational resources and longer time to train. Furthermore, due to the dame reasons, it might potentially be prone to overfitting.
While ResNet-50 and ResNet-101 certaintly outperform the other network on the image classifications, ResNeXt, introduced in the paper **Aggregated Residual Transformations for Deep Neural Networks** even outperforms the former two networks. The difference between ResNet and ResNeXt is that while ResNet uses residual blocks to learn more diverse features and data representations, ResNeXt is equipped with *cardinality*, which basically means that each residual block is repeated in the new dimention with the cardinality size and grouped convolution is used, which eventually enables ResNeXt to learn even more features and data representations than ResNet. In line with the difference in performance between ResNet-50 and ResNeXt-101, the results given in [reference for resnest] showed that ResNeXt-101 has lower error rates than ResNeXt-50. Since [reference for D3KE] did not explore the use of ResNeXt-101 for D3KE, which could potentially lead to better estimation performance as mentioned above. Hence, it is our mission in this reproduction proect to equip D3KE with ResNeXt-101 and test it.

With our mouthful explanations for ResNet, ResNeXt and our motivations having been explained, it is time to dive into the details of the network structures of D3KE and how the backbone network is used in the network.
# Method

## Network structure

## Loss function

$$
L = \lambda_1 L_{joint} + \lambda_2 L_{marker} + \lambda_3 L_{body} + \lambda_4 L_{angle}
$$

# Results

# Conclusion

# Running and commands

# Resources

