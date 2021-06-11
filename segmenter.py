import os, cv2
import numpy as np
from filter_beam import filter_beam 

def split_video(mp4_path):
    '''
    Converts an mp4 video to a series of images and saves the images in the same directory.
    Calculates the area of the largest contour for every image and relays the image with 
    the largest area to mask_all_images. 
    :param mp4_path: File name of the mp4 file to convert to series of images.
    :return: Returns the mask used, else mp4_path and number of frames if failed to find contour.
    '''
    vc = cv2.VideoCapture(mp4_path)
    vid_dir, mp4_filename = os.path.split(mp4_path)      # Get folder and filename of mp4 file respectively
    mp4_filename = mp4_filename.split('.')[0]       # Strip file extension

    idx = 0
    while (True):
        ret, frame = vc.read()
        if not ret:
            break   # End of frames reached
        img_path = vid_dir + '/' + mp4_filename + '_' + str(idx) + '.png'
        cv2.imwrite(img_path, frame) # Save all the images out
        idx +=1 

def change_contrast(image, factor):
    '''
    Changes contrast for a black and white image by the specified factor.
    :image: cv2 image object
    :factor: numerical multiplicative factor
    ''' 
    for x in range(len(image)):
        for y in range(len(image[x])):
            val = image[x,y] * factor
            image[x,y] = val if val < 255 else 255
    
    return image

def create_masks(img_path, frame_num):
    """
    Creates masks based on the segmented frame found in seg_frame_path.
    """
    #import the images
    key_frame = cv2.imread(img_path + "_" + str(frame_num) + ".png")
    beam_mask = filter_beam(key_frame)
    key_frame = cv2.cvtColor(cv2.bitwise_and(beam_mask,key_frame), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path + "_" + str(frame_num) + "_beamed.png",key_frame)
    key_frame = change_contrast(key_frame, 4.0)

    #key_mask = cv2.imread(img_path + "_mask_" + str(frame_num) + ".png",0)
    #masked_key = cv2.bitwise_and(key_frame,key_mask)
    new_frame = cv2.imread(img_path + "_" + str(frame_num + 1) + ".png")
    new_frame = cv2.cvtColor(cv2.bitwise_and(beam_mask,new_frame), cv2.COLOR_BGR2GRAY)
    new_frame = change_contrast(new_frame, 4.0)

    #trying with a couple methods here:
    #SIFT method
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(key_frame,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(new_frame,None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    for x in keypoints_1:
        print(x.pt)

    img3 = cv2.drawMatches(key_frame, keypoints_1, new_frame, keypoints_2, matches, new_frame, flags=2)
    cv2.imshow("matched",img3)
    cv2.waitKey(0)

    #use the SIFT paradigm but do it semi-manually

    #active contouring method
#def next_SIFT(sift, keypoints_1, descriptors_1, img_path, frame_num, )

if __name__ == "__main__":
    #filename = "video\pleural_effusion_clip.mp4"
    #split_video(filename)
    create_masks("video\pleural_effusion_clip",70)

#filename = "video\pleural_effusion_clip_mask_70.png"
#mask = cv2.imread(filename)
#for x in range(len(mask)):
  #  for y in range(len(mask[x])):
   #     if not(np.all(mask[x,y] == [0,0,0]) or np.all(mask[x,y] == [255,255,255])):
    #        mask[x,y] = [0,0,255]
        #if np.all(mask[x,y] < [150,150,150]):
            #mask[x,y] = [0,0,0]
        #else:
            #mask[x,y] = [255,255,255]

#mask[300,500] = [0,0,255]
#cv2.imshow("blarg",mask)
#cv2.waitKey(0)
#cv2.imwrite("video\pleural_effusion_clip_mask_70.png",mask)


