import os
import cv2

def split_video(path):
    """
    Splits the video at path :path: into images and places them into a folder with the video's filename. 
    All images are named with the video filename, an underscore, and then the number of the frame.
    """
    pass

if __name__ == "__main__":
    filename = "pleural_effusion_clip.mp4"
    split_video(filename)
