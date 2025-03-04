# Projec2: Image Classification
## Introduction
In this project, you'll classify images from the CIFAR-10 dataset. The dataset consists of airplanes, dogs, cats, and other objects. The dataset will need to be preprocessed, then train a convolutional neural network on all the samples. You'll normalize the images, one-hot encode the labels, build a convolutional layer, max pool layer, and fully connected layer. At then end, you'll see their predictions on the sample images.

## Final Results: 
<img src="assets/predictions.JPG" width=500px>



## Re-Submission
Overall great job :+1: implementing the network and tuning the hyperparameters. Just a small change in the **print_stats** function and you will be good to go.

- tasks: upload the changes back in FloydHub and resubmit project =) 

## Re-Submission feedback: 
Meets Specifications

Good job, you nailed this assignment. I've given you some extra pointers, so have a look through my notes.

Here are a few extra resources about building / tweaking Tensorflow models:

Great read about building neural networks and usage of epochs / hyperparameters:
http://cs231n.github.io/neural-networks-3/

Here's an article that talks about weight initializations:
http://cs231n.github.io/neural-networks-2/#init

Great read about all the backgrounds of stddev:
http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network/186351#186351

Link with multiple (best scoring) different models + approaches + settings and all based on research: http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d313



## Submission
the project was executed in Floydhub 

1. Ensure you've passed all the unit tests in the notebook.
2. Ensure you pass all points on the rubric.
3. save the notebook as an HTML file. 
4. Submit	
	1. Package the "dlnd_image_classification.ipynb", 
	2. push the files from your GitHub repo.
5. Hit Submit Project below!

## Running the Udacity Deep Learning Foundations image classification project on floydhub.com

1. Create an account on [floydhub.com](https://www.floydhub.com) (don't forget to confirm your email). You will automatically receive 100 free GPU hours. 

2. Install the `floyd` command on your computer:

        pip install -U floyd-cli
        
    Do this even if you already installed `floyd-cli` before, just to make sure you have the most recent version (Its pace of development is fast!).

3. Associate the command with your Floyd account:

        floyd login

    (a page with authentication token will open; you will need to copy the token into your terminal)

2. Clone this repository:

        git clone https://github.com/ludwiktrammer/deep-learning.git

    Note: There are couple minor differences between this repository and the original Udacity repository. You can read about them [in README](https://github.com/ludwiktrammer/deep-learning/tree/master/image-classification#how-is-this-repository-different-from-the-original). To follow this instructions you need to use this repository.

3. Enter the folder for the image classification project:

        cd image-classification

4. Initiate a Floyd project:

        floyd init dlnd_image_classificationdvt

5. Run the project:

        floyd run --gpu --env tensorflow --mode jupyter --data diSgciLH4WA7HpcHNasP9j

    It will be run on a machine with GPU (`--gpu`), using a Tenserflow environment (`--env tensorflow`), as a Jupyter notebook (`--mode jupyter`), with Floyd's built-in cifar-10 dataset  available (`--data diSgciLH4WA7HpcHNasP9j`).
    
6. Wait for the Jupyter notebook to become available and then access the URL displayed in the terminal (described as "path to jupyter notebook"). You will see the notebook.

7. Remember to explicitly stop the experiment when you are not using the notebook. As long as it runs (even in the background) it will cost GPU hours. You can stop an experiment in the ["Experiments" section on floyd.com](https://www.floydhub.com/experiments) or using the `floyd stop` command:

        floyd stop ID
 
    (where ID is the "RUN ID" displayed in the terminal when you run the project; if you lost it you can also find it in the ["Experiments" section on floyd.com](https://www.floydhub.com/experiments))
    
**Important:** When you run a project it will always start from scratch (i.e. from the state present *locally* on your computer). If you made changes in the remote jupiter notebook during a previous run, the changes will **not** be present in subsequent runs. To make them permanent you need to add the changes to your local project folder. When running the notebook you can download them directly from Jupyter - *File / Download / Notebook*. After downloading it, just replace your local `dlnd_image_classification.ipynb` file with the newly downloaded one.

Alternatively, If you already stoped the experiment, you can still download the file using the `floyd output` command:

    floyd output ID

(where ID is the "RUN ID" displayed in the terminal when you run the project; if you lost it you can also find it in the ["Experiments" section on floyd.com](https://www.floydhub.com/experiments))
    
Just run the command above, download `dlnd_image_classification.ipynb` and replace your local version with the newly downloaded one.

## How is this repository different from [the original](https://github.com/udacity/deep-learning)?

1. I added support for Floyds built-in cifar-10 dataset. If its presence is detected, it will be used, without a need to download anything. ([see the commit](https://github.com/ludwiktrammer/deep-learning/commit/2e84ff7852905f154f1692f67ca15da28ac43149), [learn more abut datasets provided by Floyd](http://docs.floydhub.com/guides/datasets/))

2. I added a `floyd_requirements.txt` file, so an additional dependency is automatically taken care of. ([see the commit](https://github.com/ludwiktrammer/deep-learning/commit/80b459411d4395dacf8f46be0b028c81858bd97a), [learn more about `.floyd_requirements.txt` files](http://docs.floydhub.com/home/installing_dependencies/))

3. I added a `.floydignore` file to stop local data from being uploaded to Floyd - which wastes time and may even result in a timeout ([see the commit](https://github.com/ludwiktrammer/deep-learning/commit/30d4b536b67366feef38425ce1406e969452717e), [learn more about `.floydignore` files](http://docs.floydhub.com/home/floyd_ignore/))

3. I added this README
