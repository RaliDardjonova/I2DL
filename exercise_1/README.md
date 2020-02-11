# Create Submission

You can use `create_submission.sh` to create your zip file, it will create a zip of only required files and folders.  In order to create the zip, just execute `./create_submission.sh`. For submission, this zip file should be uploaded to https://dvl.in.tum.de/teaching/submission/.


If for some reason the zip file is created manually, please make sure that the zip has the following structure. Never include datasets in the zip!

```
exercise_1.zip
├── exercise_code
	├── classifiers
		├── *.py
    ├── *.py
├── models
	├── softmax_classifier.p
	├── two_layer_net.p
	├── feature_neural_net.p
├── 1_softmax.ipynb
└── 2_two_layer_net.ipynb
└── 3_features.ipynb
```

