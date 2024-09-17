## Team: Freris Leonardos (2696) - Marios Chrisostomos Askitis (2760)
## Below is the complete commented code for image stiching 2 or 3 images. 
## All you have to do is to change the path for the images in the row 165 of the code to "InputImages/three_images" for 3-images stitching or "InputImages/two_images"
## for 2-images stiching. The result wiil be automated saved in the main folder of the project with the name "Stitched_Panorama.png".
## While program runs, plots about the key points, inital matches and good matches between the source and destination pictures will be displayed.

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read all images from a folder and storing them into a list (Images).
def ReadImage(ImageFolderPath):
    Images = []									# Input Images will be stored in this list.
    ImageNames = os.listdir(ImageFolderPath)
        
    for i in range(len(ImageNames)):             # Getting all image's name present inside the folder.
        ImageName = ImageNames[i]
        InputImage = cv2.imread(ImageFolderPath + "/" + ImageName)  # Reading images one by one.

        Images.append(cv2.resize(InputImage, (0,0), fx=1, fy=1)) # Storing images and resize if the images are too big or too small

    #If folder contains less than 2 images terminate the program.    
    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        exit(1)
    
    return Images

# Function which draw the Key points, initial matches and the good matches side by side of the two images 
def plot_features(BaseImage, SecImage, BaseImage_kp, SecImage_kp, InitialMatches, GoodMatches):

    # Using drawKeypoints to plot the keypoints in the image.
    BaseImage_kp_plot = cv2.drawKeypoints(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), BaseImage_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    SecImage_kp_plot = cv2.drawKeypoints(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), SecImage_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the results side by side.
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(BaseImage_kp_plot)
    ax[1].imshow(SecImage_kp_plot)
    fig.suptitle('Key Points', fontsize=20)
    plt.show()

    # Using drawMatchesKnn to detect and display first the inital matches and then the good matches in the pictures.
    InitialMatches_img = cv2.drawMatchesKnn(BaseImage, BaseImage_kp, SecImage, SecImage_kp, InitialMatches, None, flags=2)
    GoodMatches_img = cv2.drawMatchesKnn(BaseImage, BaseImage_kp, SecImage, SecImage_kp, GoodMatches, None, flags=2)
    plt.imshow(InitialMatches_img)
    plt.title('Initial Matches', fontsize = 20)
    plt.show()
    plt.imshow(GoodMatches_img)
    plt.suptitle('Good Matches', fontsize=20)
    plt.show()

    return None
    
# Function extracting interest points and filtering them into good matches between two images.
def FindMatches(BaseImage, SecImage):

    # Using SIFT to find the keypoints and decriptors in the images
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)
    
    # Using Brute Force matcher to find matches.
    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

    # Applytng ratio test and filtering out the good matches.
    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.4 * n.distance:
            GoodMatches.append([m])

    # Plot the previous outcomes.
    plot_features(BaseImage, SecImage, BaseImage_kp, SecImage_kp, InitialMatches, GoodMatches)

    return GoodMatches, BaseImage_kp, SecImage_kp

#Function to determine outlies with RANSAC and the perspective matrix.
def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    # If less than 4 matches found, exit the code.
    if len(Matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)

    # Storing coordinates of points corresponding to the matches found in both the images.
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

    # Changing the datatype to "float32" for finding homography.
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    # Finding the homography matrix(transformation matrix).
    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    return HomographyMatrix, Status

 #Function to warp and blend two images into one stitched.  
def warpImages(BaseImage, SecImage, HomographyMatrix):

    # Finding width and height of the left (BaseImage) and right (SecImage) images.
    height_l, width_l = BaseImage.shape[:2]
    height_r, width_r = SecImage.shape[:2]
    
    # Creating list (list_of_points_l) containing corner cordinations of the left image.
    list_of_points_l = np.float32([
        [0,0], 
        [0,height_l],
        [width_l,height_l], 
        [width_l,0]
    ])
    list_of_points_l = list_of_points_l.reshape(-1,1,2)

    # Creating list (list_of_points_r) containing perspective conrer cordinations of the right in the plane of the left image
    temp_points = np.float32([
        [0,0], 
        [0,height_r], 
        [width_r,height_r],
        [width_r,0]
    ])
    temp_points = temp_points.reshape(-1,1,2)
    
    list_of_points_r = cv2.perspectiveTransform(temp_points, HomographyMatrix)
    
    # Defining boundaries of the new combined image.
    list_of_points = np.concatenate((list_of_points_l, list_of_points_r), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel())
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel())
    
    # Creating the translation matrix.
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0,0,1]])
    
    # Warping the right image to fit in the left.
    output_img = cv2.warpPerspective(SecImage, 
                                     H_translation.dot(HomographyMatrix), 
                                     (x_max - x_min, y_max - y_min))

    # Pasting the right image into the left.
    output_img[-y_min:height_l-y_min, 
               -x_min:width_l-x_min] = BaseImage
    
    return output_img


#Function which make use of the previous ones to stitch images.
def StitchImages(BaseImage, SecImage):
    # Finding matches between the 2 images and their keypoints
    Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage)
    
    # Finding homography matrix.
    HomographyMatrix, Status = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    
    # Warping and blending 2 images.
    StitchedImage = warpImages(BaseImage, SecImage, HomographyMatrix)

    return StitchedImage

if __name__ == "__main__":
    # Reading images with given path.
    Images = ReadImage("InputImages/three_images")
    
    # Recursive stitching per 2 images from left to right.
    BaseImage = Images[0]
    for i in range(1, len(Images)):
        StitchedImage = StitchImages(BaseImage, Images[i])

        BaseImage = StitchedImage.copy()    

    # Storing stitched image in the same file as given input path with the name "Stitched_Panorama.png".
    cv2.imwrite("Stitched_Panorama.png", BaseImage)