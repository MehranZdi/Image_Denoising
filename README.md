# Image denoising
As you can figure out from the topic, this project tries to remove noises from images.

For the sake of generating a new image, I used a model which is made by 2 parts:

1- Encoder part

2- Decoder part

The used model is a mixture of Unet and Runet and it's kind of be customized. The whole architecture is shown below:


![Denoising model architecture](https://github.com/MehranZdi/Image_Inpainting/blob/main/model_architecture.jpg "model architecture").

Let's cut to the chase.

## Encoder part

In encoder part, the goal is mining information of an image. This task will be done by Convolutional layers and Pooling layers. I should mention that for feeding the model I used colored images in FFHQ dataset and I changed the images' sizes to 512 * 512 * 3.
Take a look at the bottom right part of the image, there is a guide for marks I used for explaining what every single mark in the architecture is for.

After passing the input through two convolutional layers, it's pooling layer's turn. Pooling layer can store more and vital information of an image. It reduces the height and width of the image and increases the number of channels. These steps will be done till the size of the matrix will be 16 * 16 * 512.

And this is where decoder part comes and helps us.

## Decoder part

In this part, we should generate a matrix with the size of 512 * 512 * 3 from a matrix with the size of 16 * 16 * 512 which is made by encoder part.

What we're going to do is called **upsampling**. There are many methods for upsampling, but in this project, I used **Pixel shuffle** method.

### Pixel shuffle:

First of all, we have to be familiar with sub-pixel concept. As all of us know, a digital image is made of many pixels which are related to each other. in microscopic world, there are many tiny pixels between every two pixel. These tiny pixels are called sub-pixel. take a look at the below image to get a better intuition.
Piiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiic---- Sub pixel

In pixel shuffle method, we multiply the number of channels of the next layer(Actually the number of channels that we want in the next layer) by **block size** squared and consider the result as the number of filters of the next convolutional layer.
For instance, the size of result matrix in encoder layer is 16*16*512, if we consider the block size as 2 and the number of channles of the next layer as 256, after doing mentioned computations, new matrix will be the size of 16*16*1024.

We just do sub-pixeling, so pixel shuffling is not finished yet. For doing pixel shuffle, we should divide the number of channels of the result matrix by block size squared. But there is a point. For not losing the information of the image, we multiply Height and witdth of the image by block size. In this case, we keep all informatinos of an image. So as you can see in the Model Architecture image, the matrix in the first part of decoder is 32*32*256.
Don't you have any questions? (Hint: the gray arrows!)
for keeping more information of an image, we can concatenate the corresponding matrices of the encoder part and decoder part. This is called **skip connections**.





