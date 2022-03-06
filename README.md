# Image denoising
As you can figure out from the topic, this project tries to remove noises from images.

For the sake of generating a new image, I used a model which is made by 2 parts:

1- Encoder part

2- Decoder part

The used model is a mixture of Unet and Runet and it's kind of be customized. The whole architecture is shown below:


![Denoising model architecture](https://github.com/MehranZdi/Image_Inpainting/blob/main/model_architecture.jpg "model architecture").

Let's cut to the chase.

## Encoder part

In encoder part, the goal is mining information of an image. This task will be done by Convolutional layers and Pooling layers. I have to mention that for feeding the model I used colored images in FFHQ dataset and I changed the images' sizes to 512 * 512 * 3.
Take a look at the bottom right part of the image, there is a guide for marks I used for explaining what every single mark in the architecture is for.

After passing the input through two convolutional layers, it's pooling layer's turn. Pooling layer can store more and vital information of an image. It reduces the height and width of the image and increases the number of channels. These steps will be done till the size of the matrix will be 16 * 16 * 512.

And this is where decoder part comes and helps us.

## Decoder part

In this part, we should generate a matrix with the size of 512 * 512 * 3 from a matrix with the size of 16 * 16 * 512 which is made by encoder part.

What we're going to do is called upsampling. There are many methods to do upsampling, but in this project, I used Pixel shuffle method.

### Pixel shuffle
