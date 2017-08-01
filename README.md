# tf_cat_dog
Vision. Image Classifier Cat/Dog. Tensorflow 1.2

# TODO
[ ] Make evaluator. Evaluate a given image dir with inference graph
[ ] Make notebook so we can see inference result from known images.
[ ] Use Adam optimizer. It's my favorite
[ ] Test submission on Kaggle. See public leaderboard result
[ ] Try multiple model ensemble and see changes on public leader board
[ ] Consider batch-normalization on every layer
[ ] Consider using image normalization with mean value from train_dataset

[v] Add image augmentatio (horizontal flip, image jitter, random crop)
[v] Pull out global_step var in train.py. Use it for save/load and logging.
[.] Make validation set containing same ratio of dog and cat classes
[v] Enable tensorboard draw graph for train loss, train acc, val loss, val acc
[v] Loader/Saver implementation. 
