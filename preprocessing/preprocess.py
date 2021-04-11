import sklearn
from sklearn import datasets
import cv2
import os
import sys
import random
import argparse
import numpy as np
from PIL import Image, ImageFile
import face_recognition

__version__ = '0.3.0'

def load_olivetti(path):
    # load olivetti face dataset
    return sklearn.datasets.fetch_olivetti_faces(path, shuffle=False, random_state=0, download_if_missing=True, return_X_y=False)


def generate_masks_on_images(data_path, path_to_mask_image, pic_id):

    # iniates FaceMask object with histogram of oriented gradients
    masker = FaceMask(data_path,path_to_mask_image, pic_id)
    # creates the masked images
    masker.mask()
    return

# Class of FaceMask object
class FaceMask:
    facial_features = ("nose_bridge","chin")

    def __init__(self, image_path, mask_path, pic_id, show=False, model='hog'):
        self.image_path = image_path
        self.mask_path = mask_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None
        self.pic_id = pic_id
    
    # function that determines key details via HOG
    def mask(self):
        face_array = face_recognition.load_image_file(self.image_path)
        face_locations = face_recognition.face_locations(face_array, model=self.model)
        face_poi = face_recognition.face_landmarks(face_array, face_locations)
        self._face_img = Image.fromarray(face_array)
        self._mask_img = Image.open(self.mask_path)

        # finds mask in dataset
        flag = False
        for poi in face_poi:
            skip = False
            for feature in self.facial_features:
                if feature not in poi:
                    skip = True
                    break
            if skip: continue

            flag = True
            self.mask_img(poi)

        # if face is found
        if flag:
            if self.show:
                self._face_img.show()
            # save image
            self.save_img()
        else:
            print("ERROR: no face found")

    # function to place the mask on images
    def mask_img(self, face_landmark:dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)


    def save_img(self):
        path_splits = os.path.splitext(self.image_path)
        new_face_path = os.getcwd() + "/results/" + str(self.pic_id) + '-with-mask' + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Saved masked image to {new_face_path}')


    # compute image point distances
    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                        (line_point1[0] - line_point2[0]) * point[1] +
                        (line_point2[0] - line_point1[0]) * line_point1[1] +
                        (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                        (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)




def loop_over_data(path, path_to_mask_image):
    images = images = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for i in range(len(images)):
        image = images[i]
        print("Fetching image at path " + str(image))
        
        #handles exceptions in case of weird file types
        # generate_masks_on_images(image, path_to_mask_image, i)
        try:
            generate_masks_on_images(image, path_to_mask_image, i)
        except:
            print("invalid file type")
            continue
    return

def main():

    # choose dataset path
    # olivetti option
    # path_data = "without_mask_data/olivetti/images" #arg parse this
    path_data = "without_mask_data/test2"
    # blue mask option
    path_to_mask_image = "mask_images/blue-mask.png"

    # load dataset of choice amongst options
    saved_data = load_olivetti(path_data)
    
    # loop over data and generate mask images
    loop_over_data(path_data, path_to_mask_image)


# call to main
main()