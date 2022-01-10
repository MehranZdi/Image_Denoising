import cv2
from skimage.util import random_noise
import os
import glob
from itertools import chain

class File_Maker():

    def __init__(self, path):
        self.path = path
        self.file_maker(self.path)
        self.dicti = {}

    def file_maker(self, path):
        foldernames = os.listdir(path)
        filenames = []
        for folder in foldernames:
            filenames.append(glob.glob(os.path.join(path, f'{folder}/*.jpg')))
        filenames = list(chain.from_iterable(filenames))
        return filenames

    def list_to_dict(self, filenames):
        img_files = []
        for file in filenames:
            img = cv2.imread(file)
            img = cv2.resize(img, (256 ,256))
            img = random_noise(img, mode='gaussian')
            img_files.append(img)
        keys = list(range(31))
        self.dicti = dict(zip(keys, img_files))
        return self.dicti

    def __str__(self):
        return '"dicti" is returned.'

if __name__ == '__main__':

    path = '/home/mehranz/Documents/Datasets/Denoising_face/FileMakerModule_images'
    obj = File_Maker(path)
    # filenames = file_maker(path)
    # files_dict = list_to_dict(filenames)
    print(obj)
    print('Done.')
# else:
#     path = '/home/mehranz/Documents/Datasets/Denoising_face/FileMakerModule_images'
#     filenames = file_maker(path)
#     files_dict = list_to_dict(filenames)


