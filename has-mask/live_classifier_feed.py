import cv2
import numpy as np
from keras.models import load_model
import os
import sys
sys.path.append(os.path.abspath('..')) 
from maskModel.model import mask_classifier
from nomaskModel.model import nomask_classifier

model=load_model("./model2-010.model")

labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        # print(reshaped.shape)
        
        label=np.argmax(result,axis=1)[0]
        mask_text = labels_dict[label]
        has_mask = mask_text == "mask"
        name = "?"
        if has_mask:
            name = mask_classifier(face_img)
        # else:
        #     name = nomask_classifier(face_img)
        # print(name)
        color = (0,0,255) if name == "?" else (0,255,0)
        cv2.rectangle(im,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color,-1)
        cv2.rectangle(im,(x,y+h),(x+w,y+h+40),color,-1)
        cv2.putText(im, name, (x+(w//2)-len(name)*10, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(im, mask_text, (x+(w//2)-len(mask_text)*8, y+h+30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()