# tf_cat_dog
Vision. Image Classifier Cat/Dog. Tensorflow 1.2

# Prepare dataset

Put your train, test dataset in `./data/train` and `./data/test` respectly. 
The directory should contain image files. 

Run script to prepare tfrecords files for train and valid dataset. 
It will use 80% of data for train, and 20% of data for valid. 

`./create_cat_dog_tfrecords.py --splitname train`

Run this to prepare tfrecord file for test. 

`./create_cat_dog_tfrecords.py --splitname train`

Now you can see tfrecords files in `./data/records`. 
You may have to create the directory first. 

# Train

Train from the start, or from the recent check point. 

`python train.py`

It will output checkpoints in `./train_output`

Optionally you can start at some checkpoint with 

`python train.py --load_step=10200`

# Eval

This script will load checkpoint from `./train_output` and create submission file at `./eval_output`. You may have to create eval_output dir first. Also, you can specify checkpoint with --load_step option. 

`python train.py --mode=test`


# TODO
[ ] ratio could be import for spatial feature, try zeropadding.
[ ] start from small dataset
[ ] Binary softmax vs sigmoid. logloss 
[ ] Test submission on Kaggle. See public leaderboard result
[ ] Make evaluator. Evaluate a given image dir with inference graph
[ ] Make notebook so we can see inference result from known images.
[ ] Use Adam optimizer. It's my favorite
[ ] Try multiple model ensemble and see changes on public leader board
[ ] VGG Net is known as slow. Is there better model for image classification?
[ ] Consider using image normalization with mean value from train_dataset

[v] Consider batch-normalization on every layer
[v] Add image augmentatio (horizontal flip, image jitter, random crop)
[v] Pull out global_step var in train.py. Use it for save/load and logging.
[v] Make validation set containing same ratio of dog and cat classes
[v] Enable tensorboard draw graph for train loss, train acc, val loss, val acc
[v] Loader/Saver implementation. 
