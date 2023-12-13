# Eventfulness for Interactive Video Alignment
This is a barebone repository for code of *Eventfulness for Interactive Video Alignment*

# Installation 
You can first clone this directory with the command
```
git@github.com:jiatiansun/Eventfulness.git
```
We set up a virtual environment using [conda](https://docs.anaconda.com/free/anaconda/install/index.html) for network training/testing. With the assumption that it is being installed for your bash script beforehand, you can create the eventfulness by simply using the commands below:
```
cd setup
bash createCondaEnv.sh
```

# Training Network On Synthetic Data
To train a network on the synthetic data we generated, download first the training and testing dataSets from [eventfulness website](https://www.cs.cornell.edu/abe/projects/eventfulness/) and store the extracted `dataSets` directory under the root of this repository.

Then to run the scripts related to network training or prediction, navigate to the `scripts` directory with the command
`cd scripts`. In this directory, you can train a network with the command
```
python train.py [ARGUMENTS]
```
You can learn about the appliable arguments with the command:

```
python train.py --help
```
An example commmand of training a network with 2 CUDA GPUS on our synthetic data is stored in `./scripts/train.sh`. Run the script to start the training process. The trained checkpoints of the network would be stored at `scripts/lossAccuracyReport/START_TRAIN_TIME/prediction` and the training/validation loss data would be stored at `scripts/runs/START_TIME_MACHINE/`. You can visualize the loss plot by running tensorboard in the `scripts` directory.

# Predict Eventfulness with our Trained Network

Download our trained model from  [eventfulness website](https://www.cs.cornell.edu/abe/projects/eventfulness/) and then extract the `checkpoints` and place it under the eventfulness repository. Then, you can use the model to predict eventfulness for all videos stored in a dataset in the format of 
```
target_dir/
    val/
        vidType1/
            vid01.mp4
            vid02.mp4
            ...
        vidType2/
            vid11.mp4
            vid12.mp4
            ...
        ...
```
You take `bouncingBall` dataset in your downloaded file as a reference for the dataset structure. Similarly in `scripts`, you can use 
```
python predict.py [ARGUMENTS]
```
to predict eventfulness for a dataset and for the detail instructions of using different arguments, please run the command `python predict.py --help` to find out. 

The eventfulness prediction would be stored as `.json` files in the `result` subdirectory under the `target_dir` that you would like to make prediction for.

For more questions regarding training or predicting eventfulness, please contact Caroline Sun by <js3623@cornell.edu> or Abe Davis by <abedavis@cornell.edu>.