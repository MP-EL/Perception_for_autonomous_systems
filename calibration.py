import cv2
import numpy as np
import glob

class StereoCalibration:
    def __init__(self):

        self.img_dir_l = r'Stereo_calibration_images/left-*' # images for calibration
        self.img_dir_r = r'Stereo_calibration_images/right-*' # images for calibration

        self.img_set_l = glob.glob(self.img_dir_l)
        self.img_set_r = glob.glob(self.img_dir_r)

        assert (len(self.img_set_l) != 0) and (len(self.img_set_r) != 0), "No images found in directory"

        self.cb_vert = 9 # number of vertical corners
        self.cb_horiz = 6 # number of horizontal corners

        self.num_corner_found = 0
        self.obj_points = np.zeros((self.cb_vert*self.cb_horiz, 3), np.float32)
        self.obj_points[:,:2] = np.mgrid[0:self.cb_vert, 0:self.cb_horiz].T.reshape(-1,2)

        self.obj_points_3d = []
        self.img_points_r = []
        self.img_points_l = []

        self.show_imgs = False
        self.initial_calibration()

    def initial_calibration(self):
        # 2 for loops for both left and right images
        for calib_img_l in self.img_set_l:
            img_l = cv2.imread(calib_img_l)
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (self.cb_vert, self.cb_horiz), None)
            if ret_l:
                self.img_points_l.append(corners_l)
                self.num_corner_found += 1
                self.obj_points_3d.append(self.obj_points)

                if self.show_imgs:
                    cv2.imshow('img_l', cv2.drawChessboardCorners(img_l, (self.cb_vert, self.cb_horiz), corners_l, ret_l))
                    cv2.waitKey(100)

        for calib_img_r in self.img_set_r:
            img_r = cv2.imread(calib_img_r)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (self.cb_vert, self.cb_horiz), None)
            if ret_r:
                self.img_points_r.append(corners_r)
                self.num_corner_found += 1
                # obj_points_3d.append(obj_points)
                
                if self.show_imgs:
                    cv2.imshow('img_r', cv2.drawChessboardCorners(img_r, (self.cb_vert, self.cb_horiz), corners_r, ret_r))
                    cv2.waitKey(100)

        cv2.destroyAllWindows()
        assert self.num_corner_found != 0
        print(f"Found {self.num_corner_found} corners")

        self.left_ret, self.left_mtx, self.left_dist, self.left_rvecs, self.left_tvecs = cv2.calibrateCamera(self.obj_points_3d, self.img_points_l, gray_l.shape[::-1], None, None)

        self.right_ret, self.right_mtx, self.right_dist, self.right_rvecs, self.right_tvecs = cv2.calibrateCamera(self.obj_points_3d, self.img_points_r, gray_r.shape[::-1], None, None)

        # stereo calibration

        self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(self.obj_points_3d,
                                                                                                                                                      self.img_points_l, 
                                                                                                                                                      self.img_points_r, 
                                                                                                                                                      self.left_mtx, 
                                                                                                                                                      self.left_dist, 
                                                                                                                                                      self.right_mtx, 
                                                                                                                                                      self.right_dist, 
                                                                                                                                                      gray_l.shape[::-1], 
                                                                                                                                                      flags=cv2.CALIB_FIX_INTRINSIC)
        #Rectification

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(self.left_mtx, self.left_dist, self.right_mtx, self.right_dist, gray_l.shape[::-1], self.R, self.T)

        mx_l, my_l = cv2.initUndistortRectifyMap(self.left_mtx, self.left_dist, self.R1, self.P1, gray_l.shape[::-1], cv2.CV_32FC1)
        mx_r, my_r = cv2.initUndistortRectifyMap(self.right_mtx, self.right_dist, self.R2, self.P2, gray_r.shape[::-1], cv2.CV_32FC1)


    def undistort_images(self, img_l, img_r):
        #undistort:
        img_left_dist = cv2.undistort(img_l, self.left_mtx, self.left_dist, self.P1, self.left_mtx)
        img_right_dist = cv2.undistort(img_r, self.right_mtx, self.right_dist, self.P2, self.right_mtx)

        return img_left_dist, img_right_dist

    def rectify_images(self, img_l, img_r):
        # rectify:
        img_left_rect = cv2.remap(img_l, self.mx_l, self.my_l, cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_r, self.mx_r, self.my_r, cv2.INTER_LINEAR)

        return img_left_rect, img_right_rect
