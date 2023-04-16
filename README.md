# Group-42-Towards-Single-Camera-Human-3D-Kinematics
In this blog, direct 3D human kinematic estimation (D3KE) proposed in [reference of paper], as well as a reproduction of the paper we conducted with a slight modification to the proposed network are explained.


# Introduction
\textit{Towards Single Camera Human 3D-Kinematics}, the paper published in 2022, proposed a method - Direct 3D Human Kinematic Estimation (D3KE) by applying a pretrained ResNet-50 convolutional network as a backbone and a sequential network to do temporal smoothing and to lift the pose from 2D to 3D. In our reproduction project, we replaced the convolutional ResNet-50 by its variant, ResNeXt101, to see whether the performance of the estimation can be improved.
Before diving into the details of the network structures of D3KE and how the backbone network is used in the network, we will briefly explain about ResNet-50 as well as ResNet-101 and the motivations of our project.
ResNet-50 was first introduced in the paper \textit{Deep Residual Learning for Image Recognition} published in 2015 by H. Kaiming, X. Shangyu, S. Ren, and S. Jian. ResNet-50 refers to a specific type of convolutional neural network (CNN) called residual network with 50 layers and weights, consisting of the first convolutional layer followed by a max pooling layer, 48 convolutional layers and an average pooling layer followed by a fully connected layer. In every 3 convolutional layers in the 48 convolutional layers, a skip connection, also known as residual connnection is used. Skip connections are the main key feature of ResNet, with which the vanishing gradient problem (VGP) is mitigated by allowing feature maps to bypass intermediate layers, which in turn enables the deeper layers to learn more complex representations without suffering from the VGP.
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

