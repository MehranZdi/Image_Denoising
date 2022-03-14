# Image denoising

## Introduction
As you can figure out from the topic, this project tries to remove noises from images by deep learning methods.

## Table of Contents
- [Explanation](#Explanation)
    - [Encoder part](#Encoder_part)
- Code
- Streamlit
- Result
- References
- Contributing

## Explanation

For the sake of generating a new image, I used a model which is made by 2 parts:

1- Encoder part

2- Decoder part

The used model is a mixture of Unet and Runet and it's kind of be customized. The whole architecture is shown below:


![Denoising model architecture](https://github.com/MehranZdi/Image_Inpainting/blob/main/model_architecture.jpg "model architecture").

Let's cut to the chase.

### Encoder part

In encoder part, the goal is mining information of an image. This task will be done by Convolutional layers and Pooling layers. I should mention that for feeding the model I used colored images in FFHQ dataset and I changed the images' sizes to 512 * 512 * 3.
Take a look at the bottom right part of the image, there is a guide for marks I used for explaining what every single mark in the architecture is for.

After passing the input through two convolutional layers, it's pooling layer's turn. Pooling layer can store more and vital information of an image. It reduces the height and width of the image and increases the number of channels. These steps will be done till the size of the matrix will be 16 * 16 * 512.

And this is where decoder part comes and helps us.

### Decoder part 

In this part, we should generate a matrix with the size of 512 * 512 * 3 from a matrix with the size of 16 * 16 * 512 which is made by encoder part.

What we're going to do is called **upsampling**. There are many methods for upsampling, but in this project, I used **Pixel shuffle** method.

### Pixel shuffle:

First of all, we have to be familiar with sub-pixel concept. As all of us know, a digital image is made of many pixels which are related to each other. in microscopic world, there are many tiny pixels between every two pixel. These tiny pixels are called sub-pixel. take a look at the below image to get a better intuition.
Piiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiic---- Sub pixel

In pixel shuffle method, we multiply the number of channels of the next layer(Actually the number of channels that we want in the next layer) by **block size** squared and consider the result as the number of filters of the next convolutional layer.
For instance, the size of result matrix in encoder layer is 16*16*512, if we consider the block size as 2 and the number of channles of the next layer as 256, after doing mentioned computations, new matrix will be the size of 16*16*1024.

We just do sub-pixeling, so pixel shuffling is not finished yet. For doing pixel shuffle, we should divide the number of channels of the result matrix by block size squared. But there is a point; for not losing the information of the image, we multiply height and witdth of the image by block size. In this case, we keep all informatinos of an image. So as you can see in the Model Architecture image, dimensions of the matrix in the first part of decoder is 32*32*256.
Don't you have any questions? (Hint: the gray arrows!)


Based on Unet paper, for keeping more information of an image, we can concatenate the corresponding matrices of the encoder part and decoder part. This will be done by **skip connections**.

After doing mentioned concatenation, the output matrix pass through two convolutional layers and then a ReLU activation function. These steps will be done till the dimensions of the matrix be 512*512*16. This matrix will passed through a convolutional layer with 1 filter and a sigmoid activation function.



## Code
### A function for creating convolutional layers

the below function helps us to implement convolutional layer blocks which are in the encoder part and the decoder part(Blue arrows).
I trained this project on Kaggle, FFHQ dataset is available on Kaggle so you can use that.

```python 
def conv_blocks_maker(inputs=None, n_filters=32, kernel_size=(3,3), padding='same'):
    
    '''First layer'''
    
    conv = tkl.Conv2D(filters = n_filters,
                      kernel_size = kernel_size,
                      padding = padding,
                      kernel_initializer = 'he_normal')(inputs)
    
    conv = tkl.Activation('relu')(conv)
    
    
    '''Second layer'''
    
    conv = tkl.Conv2D(filters = n_filters,
                     kernel_size = kernel_size,
                     padding = padding,
                     kernel_initializer = 'he_normal')(conv)
    
    conv = tkl.Activation('relu')(conv)
    
    return conv
```

### A function for creating pool layers
As you can figure out from function's name, it's for making pool layers(red arrows).

```python
def pool_maker(skip, pool_size=(2,2), dropout_prob=0.1):
    conv = tkl.MaxPooling2D(pool_size)(skip)
    conv = tkl.Dropout(dropout_prob)(conv)
    
    return conv
```

### Pixel shuffling
Following functions give us a hand to handle pixle shuffling part in decoder part(dark green arrows).
As I mentioned before, at the end of the pixel shuffling, we divide the number of filters by block size squared. And then multiply height and width by block size. This is tensorflow's duty. There is a function in tensorflow named **depth_to_space** that takes care of that.
```python 
def upsampler(conv, block_size, num_filters):  
    
    """Sub-pixel convolution"""
    conv = Conv2D(num_filters * (block_size ** 2), (3,3), padding='same')(conv)
    
    """Pixel shuffle"""
    conv =  pixel_shuffle(block_size)(conv)    
    
    return conv


def pixel_shuffle(block_size):
    return lambda conv: tf.nn.depth_to_space(conv, block_size)
```
### A function for decoder part
This function does whatever I explained above about decoder part. above functions is called by the following function.
If you have paid attention, you figured out that in the last row in decoder part, there is just one convolutional layer. I took care of that with a flag named conv_blocks which is True or False. So conv_blocks variable in decoder function is for what I said.

```python
def decoder(conv, skip, block_size, n_filters, kernel_size, conv_blocks):
    pixel_shuffle = upsampler(conv, block_size, n_filters)
    concatenated = concatenate([skip, pixel_shuffle])
        
    if (conv_blocks == True):
        conv = conv_blocks_maker(concatenated, n_filters, kernel_size)
    else:
        conv = Conv2D(n_filters, (3,3), padding='same',
                      activation = 'relu')(concatenated)
    
    return conv
```    

### A function for creating the architecture
In the following function, all above functions gather and create the whole architecture.
This part is obvious, so you won't have a serious problem.

```python
def unet_model_creator(n_filters=32, dropout_prob=0.1):
    
    '''Encoder part:'''
    
    input_size = (WIDTH, HEIGHT, n_channels)
    
    input_img = tf.keras.Input(input_size, name = 'image' )
    skip_1 = conv_blocks_maker(input_img, n_filters / 2, kernel_size = 3)
    
    conv = pool_maker(skip_1)
    skip_2 = conv_blocks_maker(conv, n_filters, kernel_size = 3)
    
    conv = pool_maker(skip_2)
    skip_3 = conv_blocks_maker(conv, n_filters * 2, kernel_size = 3)
    
    conv = pool_maker(skip_3)
    skip_4 = conv_blocks_maker(conv, n_filters * 4, kernel_size = 3)
    
    conv = pool_maker(skip_4)
    skip_5 = conv_blocks_maker(conv, n_filters * 8 , kernel_size = 3)
    
    conv = pool_maker(skip_5)
    conv = conv_blocks_maker(conv, n_filters * 16 , kernel_size = 3)
    

    '''Decoder part'''        
    
    decoded_layer = decoder(conv, skip_5, block_size=2,
                            n_filters=256, kernel_size=(3,3),
                            conv_blocks=True)
    
    decoded_layer = decoder(decoded_layer, skip_4, block_size=2,
                            n_filters=128, kernel_size=(3,3),
                            conv_blocks=True)
    
    decoded_layer = decoder(decoded_layer, skip_3, block_size=2,
                            n_filters=64, kernel_size=(3,3),
                            conv_blocks=True)
    
    decoded_layer = decoder(decoded_layer, skip_2, block_size=2,
                            n_filters=32, kernel_size=(3,3),
                            conv_blocks=True)
    
    decoded_layer = decoder(decoded_layer, skip_1, block_size=2,
                            n_filters=16, kernel_size=(3,3),
                            conv_blocks=False)

    output = Conv2D(3, (1,1), activation='sigmoid')(decoded_layer)

    model = tf.keras.Model(inputs = [input_img], outputs = [output])
    
    return model
```
In last decoded_layer variable, I set conv_blocks to False. This is exactly what I said before. 

### Training the model
First of all, for handling the dataset, I used DataGenerator class which is a suitable way to deal with huge datasets. Check [this](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) out.

For training the model, I used Adam optimizer and MSE as loss functin. for choosing the best value for learning rate, I did Try and Error methode and finally used the learning rate with the value of 0.003. I trained my model in 20 epochs.

I hope this projects can help you with anything you want.

## Streamlit

There are many ways to deploy a ML or DL model. Streamlit is one of the fastest and easiest. There is no need to get involoved with frontend and backend,because streamlit take cares of everything. Check [Streamlit](https://streamlit.io/).

I wrote a program to make a webpage in order to work with the model. Source codes are available in _main.py_ file and _helper.py_ file.

## Result
Take a look at a sample result below:

![result](https://github.com/MehranZdi/Image_Inpainting/blob/main/result.png "Denoised image").

## References
1- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

2- [RUNet: A Robust UNet Architecture for Image Super-Resolution](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WiCV/Hu_RUNet_A_Robust_UNet_Architecture_for_Image_Super-Resolution_CVPRW_2019_paper.pdf)


## Contributing

I'd appreciate if you contribute and make the project better and more robust.

