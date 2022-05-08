# IMPORT NECCESSARY LIBRARIES:
import numpy as np
import cv2
import glob
#import matplotlib.pyplot as plt


# FUNCTION DEFINITIONS:
def stereoCalibrate(imgPath_left, imgPath_right, nb_vertical=6, nb_horizontal=9, verbose=0):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp_left = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
    objp_left[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)
    objp_right = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
    objp_right[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints_left = [] # 3d point in real world space
    objpoints_right = [] # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = [] # 2d points in image plane.
    objpoints = []  # 3d point in real world space
    
    images_left = glob.glob(imgPath_left)
    images_right = glob.glob(imgPath_right)
    assert images_left
    assert images_right
    
    
    for fname_left in images_left:
        img_left = cv2.imread(fname_left)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        
        #Implement findChessboardCorners here
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (nb_vertical, nb_horizontal), flags=cv2.CALIB_CB_ADAPTIVE_THRESH )   
        
        objpoints.append(objp_left)
        # If found, add object points, image points (after refining them)
        if ret_left == True:
            objpoints_left.append(objp_left)
            imgpoints_left.append(corners_left)

            if(verbose):
                # Draw and display the corners
                img_left = cv2.drawChessboardCorners(img_left, (nb_vertical,nb_horizontal), corners_left,ret_left)
                cv2.imshow('img_left',img_left)
                cv2.waitKey(100)
            
    cv2.destroyAllWindows()
    
    for fname_right in images_right:
        img_right = cv2.imread(fname_right)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        #Implement findChessboardCorners here
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (nb_vertical, nb_horizontal), flags=cv2.CALIB_CB_ADAPTIVE_THRESH )
        
        #objpoints.append(objp_right)
        # If found, add object points, image points (after refining them)
        if ret_right == True:
            objpoints_right.append(objp_right)
    
            imgpoints_right.append(corners_right)

            if(verbose):    
                # Draw and display the corners
                img_right = cv2.drawChessboardCorners(img_right, (nb_vertical,nb_horizontal), corners_right,ret_right)
                cv2.imshow('img_right',img_right)
                cv2.waitKey(100)
            
    cv2.destroyAllWindows()     
    
    print("Rectifying images...")
    # Calibrate camera
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints_left, imgpoints_left, gray_left.shape[::-1], None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints_right, imgpoints_right, gray_left.shape[::-1], None, None)
    
    dims = gray_left.shape[::-1]
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right,dist_right, dims,
                criteria=stereocalib_criteria, flags=flags)
    
    R1 = np.zeros(shape=(3,3))
    R2 = np.zeros(shape=(3,3))
    P1 = np.zeros(shape=(3,3))
    P2 = np.zeros(shape=(3,3))
        
    cv2.stereoRectify(M1, d1, M2, d2,dims, R, T, R1, R2, P1, P2, Q=None, alpha=-1, newImageSize=(0,0))
    maplx, maply = cv2.initUndistortRectifyMap(M1, d1, R1, mtx_left, dims, cv2.CV_32FC1)
    maprx, mapry = cv2.initUndistortRectifyMap(M2, d2, R2, mtx_right, dims, cv2.CV_32FC1)
    
    print(maply.shape)
    print(maply)
    
    return maplx, maply, maprx, mapry
