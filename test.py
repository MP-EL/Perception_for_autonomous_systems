import numpy as np
import cv2
import matplotlib.pyplot as plt
from calibration import *
from kalmann import *
from machine_learning import *
import glob
import time

if __name__ == '__main__':
    dir_to_folders = r'Stereo_conveyor_with_occlusions/'

    img_dir_left = dir_to_folders + 'left/*'
    img_dir_right = dir_to_folders + 'right/*'

    img_set_left = sorted(glob.glob(img_dir_left))
    img_set_right = sorted(glob.glob(img_dir_right))

    img_sets = zip(img_set_left, img_set_right)

    #setup calibration
    print("Setting up calibration...")
    calib = StereoCalibration()
    print("Calibration done!")

    #setup kalman filter
    print("Setting up kalman filter...")
    kalman = KalmanFilter()
    print("Kalman filter done!")

    print("Setting up inference model...")
    inference = InferenceModel()
    print("Inference model done!")
    prev_time = time.time()
    old_center_point = np.array([[-1], [-1]])

    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280,720))

    #temp points
    for img_left, img_right in img_sets:
        start_time = time.time()

        img_left = cv2.cvtColor(cv2.imread(img_left), cv2.COLOR_BGR2RGB)
        img_right = cv2.cvtColor(cv2.imread(img_right), cv2.COLOR_BGR2RGB)

        img_left_calib, img_right_calib = calib.undistort_images(img_left, img_right)

        #Perform inference on the images and get the center point return none if no point is found
        print("Performing inference...")
        center_point = inference.perform_inference(img_left_calib, img_right_calib)
        print("Inference done!")

        #Perform kalman filter update
        print("Performing kalman filter update...")
        if center_point[0] != old_center_point[0] and center_point[1] != old_center_point[1]:
            print("TRUE")
            kalman.update(center_point)
            old_center_point = center_point
        kalman.predict()
        print("Kalman filter update done!")
        x, P = kalman.get_state()

        size_of_tracked_obj = 10
        cv2.circle(img_left, (int(x[0]), int(x[3])), size_of_tracked_obj, (0, 255, 0), 2)
        #draw frame:
        out.write(np.flip(img_left, axis=-1))

        # Run at 30 fps:
        curr_time = time.time()
        diff = curr_time - prev_time
        prev_time = curr_time
        delay = max(1/30 - diff, 0)
        print(f"frame delay: {delay}, actual diff: {diff}")
        time.sleep(delay)
    
    out.release()

