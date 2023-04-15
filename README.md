# Group-42-Towards-Single-Camera-Human-3D-Kinematics
A repo for the practical of the TU Delft Deeplearning course, about the recreation of 3D human pose angles from a single 2D image


# Introduction
The Towards Single Camera Human 3D-Kinematics paper uses Direct 3D Human Kinematic Estimation (D3KE) by applying a pretrained ResNeXt50 convolutional network as a backbone and a sequential network to do temporal smoothing and to lift the pose from 2D to 3D. In this project group 42 is trying to swap the convolutional ResNexXt50 with other convolutional networks of the same input and output dimesnions, (TODO be more specific) like the ResNeXt101, to see how this affects performance of the overall estimation. 

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

