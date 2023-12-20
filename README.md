# Style_transfer
Adaptive instance normalization style transfer algorithm  

# Overview
The StyleTransfer project leverages the AdaIN approach to blend the content of one image with the style of another. This is achieved through a model that consists of an encoder, a style transfer module (AdaIN), and a decoder. The encoder captures the content and style features of the input images, the AdaIN module adapts the style features onto the content, and the decoder reconstructs the final image.

# Example Results
Below are some examples of style transfers using our model:

# Content Image
![image](https://github.com/ZanePrance/Style_transfer/assets/141082203/8ccfefde-8292-4a35-868f-8c8a304eb7a3)

# Style Image
![image](https://github.com/ZanePrance/Style_transfer/assets/141082203/6a9b8262-7ba4-4ce0-9322-c05255fb7c6f)

# Style Transferred Image
![image](https://github.com/ZanePrance/Style_transfer/assets/141082203/7d908c4b-8458-43ae-8ab6-de77cfb64e6e)

# Model Architecture
The model is composed of three main parts:

Encoder: Captures the content and style features from the input images.

AdaIN Layer: Transfers the style features onto the content features.

Decoder: Reconstructs the final image from the combined features.

# Encoder
The encoder is based on the first few layers of a pre-trained VGG-19 network. It extracts feature maps from the input images.

# AdaIN Layer
The AdaIN layer aligns the mean and variance of the content features to those of the style features, effectively transferring the style.

# Decoder
The decoder is a series of upsampling and convolution layers that reconstructs the final stylized image from the combined features.

# Customization
You can customize various parameters including the layers used for content and style representation, the weight of the style loss, and more.
