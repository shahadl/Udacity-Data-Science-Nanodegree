# Dog Breeds Classifier

## Table of Contents

1. [Project Motivation](#motivation)
2. [Overview](#over)
3. [Description of the contents](#files)
4. [Results](#results)
5. [Required Libraries](#libraries)


## Project Motivation <a name="motivation"></a>
This is a Udacity DSND project to classify Dog breeds using pre-trained CNN model.

## Overview <a name="over"></a>
I built an image classification model.

The model accepts images as input. 
If a dog is spotted in the photo, it will provide an estimate of the dog's breed.
And if a human face is detected, it will provide an estimate of the most similar dog breed.

The image below displays a sample output of the finished project.
![Sample Dog Output](https://github.com/shahadl/PROJECT-2/blob/master/image/1.PNG)


## Description of the contents <a name="files"></a>
- dog_app.ipynb is a Jupyter notebooks, it has the whole code
- The image folder, it contains some pictures that i used in this project.
- The test folder, init contains some pictures that i used to test the result of my model.

## Results <a name="results"></a>
I tested my model by entering 6 samples as pictures, 4 dogs, 1 human face, and 1 other. 
The model was able to successfully identify: dog pictures, the human face,and other
- The model successfully predicted the breed of 3 out of 4 dogs.

here are some pictures i used to test the results.
![Sample Bernese Mountain Dog Output](https://github.com/shahadl/PROJECT-2/blob/master/image/2.PNG)
- Here's when i entered an image of a Bernese Mountain Dog
![Sample human Output](https://github.com/shahadl/PROJECT-2/blob/master/image/3.PNG)
- When I entered Jake Gyllenhaal face, it resembled him with "English springer spaniel" dog breed.

![Sample "others" Output](https://github.com/shahadl/PROJECT-2/blob/master/image/4.PNG)
- And when I entered a toy image, it did not recognize it either as a dog or as a human.

In this blog post, I wrote more details about the results of this classification 
[Dog Breed Classification](https://medium.com/@Lzcv2/dog-breed-classification-41abe5e01c32?postPublishedType=initial)

## Required Libraries <a name="libraries"></a>
- Pandas, NumPy, Scikit-learn (Machine Learning Libraries)
- Matplotlib 
- Keras
