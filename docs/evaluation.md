# Model Evaluation 

Our model evaluation script can be found in `evaluate.py`. We also include a checkpoint of our model
trained on ScanNet in `models/pretrained_weights`. You can evaluate the model by running the
following command:

```
python evaluate.py mine --checkpoint unsupervisedRR/models/pretrained_weights/unsuprr_scannet.pkl --progress_bar --boost_alignment
```
