# MaskID

During the pandemic, unlocking phones that uses facial recognition is difficult with a mask on. This creates frustration and unintentionally encourages people to take off their masks in public to unlock their phones, creating a health concern. Therefore, we derived a facial recognition model that recognizes people both with and without masks on, which creates a more efficient biometric passcode hopefully ensures that people won't remove their masks in public to unlock their phone. 

## Installation

Simply clone this repository to get all the files and datsets to run our MaskID.

```bash
git clone https://github.com/mirandamo/maskID.git
```

## Usage
To train the mask neural network, run the following command. You will need to have `tensorflow`, `imutils`, `sklearn`, amongst others installed. Run `pip install <module>` if needed. Activate the cs1430 environment as necessary.

```python
(cs1430_env) \maskID\face-recognition\has-mask> python model_train.py
```


To perform identification of a user, collect data by running `face_capture.py`. Will capture 30 images of your face live. Like you would with Apple's iPhone FaceID, slightly rotate your face between images. Perform this with and without masks, naming the captures `<your name> <(mask)|(nomask)>` when prompted in the command line. Make sure your face is in frame, otherwise it will error when no face is detected.

```python
(cs1430_env) \maskID\face-recognition> python face_capture.py
```

To perform face recognition, feature's need to be extracted by running `extract_embeddings.py` with the proper arguments. This script outputs a `.pickle` file of feature vectors in the outputs folder. Next, fit the SVM by running `train_model.py` with the appropriate arguments. You can change the kernel by change the line `kernel_type = 'rbf'` with `linear` or `polynomial`. Lastly, run the live video feed by calling `recognize_video.py` with the appropriate arguments, including specifying the kernel used when training.

```python
# extract feature vectors
(cs1430_env) \maskID\face-recognition> python .\extract_embeddings.py 
--dataset dataset/nomask 
--embeddings output/embeddings_nomask.pickle 
--detector face_detection_model 
--embedding-model .\openface.nn4.small2.v1.t7

# fit SVM
(cs1430_env) \maskID\face-recognition> python .\train_model.py 
--embeddings .\output\embeddings_nomask.pickle
--recognizer output/recognizer_nomask_rbf.pickle 
--le output/le_nomask_rbf.pickle

# run live video feed
(cs1430_env) \maskID\face-recognition> python .\recognize_video.py 
--detector .\face_detection_model\ 
--embedding-model .\openface.nn4.small2.v1.t7 
-k rbf

```


Alternatively, if instead you want to run the classifier on a single image rather than a live feed, run `recognize.py` and pass in the image using the `-i` arg.
```python
# run on single image
(cs1430_env) \maskID\face-recognition> python .\recognize.py 
--detector .\face_detection_model\ 
--embedding-model .\openface.nn4.small2.v1.t7 
-k rbf 
-i images/michael/mike_nomask.jpg
```

## Contributing
Any questions, please direct to the contributors for this project, {michael_chen5, dev_ramesh, matteo_lunghi, man_hei_miranda_mo}@brown.edu


## License
04/23/2021\
Brown University CS1430 Computer Vision - Spring 2021