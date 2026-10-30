# SPEC: SimPLe world model (Pong)

# 0. Scope
This repo builds the deterministic world model for a reproduction of SimPLe (see docs/paper.md and reference/tensor2tensor for more info) for Pong. Its final implementation will be in a shared Colab Notebook: work done previously by teammates can be found in the "teammates" directory. OOP should be a core design philosophy: the model class definition and pertinent unit tests will be separate cells in a Colab notebook. This implementation is for a final project in a reinforcement learning class, so legibility and informative comments are particularly imporant. It should be implemented in PyTorch.

# 1. Model Architecture
## 1.1 Note on differing architectural descriptions in source material
Fig 2 and Appendix C in the paper (docs/paper.md) differ in certain metrics describing the world model architecture. This implementation will focus initially on the architecture described in Appendix C of paper.md.

## 1.2 Core Architecture
The deterministic world model can roughly be descirbed as a CNN encoder/decoder, which takes 4 frames (3x105x80) and an input action from a wrapper/task environment (specified in teammates) to predict the next frame and the obatined reward. 

### Architecture
0. Input (Bx4x3x105x80) (B = batches)

| Layer | Output | Details | 
| --- | --- | --- |
| 1. Input frame dense | 96  | - |
| Downscale convolution 1 | 192 | kernel 4x4, stride 2x2 |
| Downscale convolution 2 | 384 | kernel 4x4, stride 2x2 |
| Downscale convolution 3 | 768 | kernel 4x4, stride 2x2 |
| Downscale convolution 4 | 768 | kernel 4x4, stride 2x2 |
| Downscale convolution 5 | 768 | kernel 4x4, stride 2x2 |
| Downscale convolution 6 | 768 | kernel 4x4, stride 2x2 |
| Action embedding | 768 | - | 
| Reward predictor hidden| 128 | - |
| Reward predictor output dense| 3 | - | 
| Middle convolution 1| 768 | kernel 3x3, stride 1x1 | 
| Middle convolution 2| 768 | kernel 3x3, stride 1x1 | 
| Upscale convolution 1| 768 | kernel 4x4, stride 2x2|
| Upscale convolution 2| 768 | kernel 4x4, stride 2x2|
| Upscale convolution 3| 768 | kernel 4x4, stride 2x2|
| Upscale convolution 4| 384 | kernel 4x4, stride 2x2|
| Upscale convolution 5| 192 | kernel 4x4, stride 2x2|
| Upscale convolution 6| 96 | kernel 4x4, stride 2x2|
| Output frame dense | 768 | - | 

From the paper: "All activation functions are ReLU, except for the layers marked as "output", which have softmax activations... In the frame prediction network, the downscale layers are connected to the corresponding upscale layers with residual connections. All convolution and transposed convolution layers are preceded by dropout 0.15 and followed by layer normalization." Please adhere to these guidelines when building the model. 

