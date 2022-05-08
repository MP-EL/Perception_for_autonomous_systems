# IMPORT NECCESSARY LIBRARIES:
import numpy as np
import cv2
import sys
import glob
import os
import time
# IMPORT OTHER PY FILES:
sys.path.append('External Functions')
import stereoCamera
import kalmanFilter


#FUNCTION DEFINITIONS:

    
# SETUP:
print("Calibrating stereo camera...") #debug.
if os.path.isfile('Config Files/mapry.txt'):
    maplx = np.float32(np.loadtxt('Config Files/maplx.txt'))
    maply = np.float32(np.loadtxt('Config Files/maply.txt'))
    maprx = np.float32(np.loadtxt('Config Files/maprx.txt'))
    mapry = np.float32(np.loadtxt('Config Files/mapry.txt'))
else:
    maplx, maply, maprx, mapry = stereoCamera.stereoCalibrate('Images/Stereo_calibration_images/right*.png', 'Images/Stereo_calibration_images/left*.png')
    np.savetxt('Config Files/maplx.txt', maplx)
    np.savetxt('Config Files/maply.txt', maply)
    np.savetxt('Config Files/maprx.txt', maprx)
    np.savetxt('Config Files/mapry.txt', mapry)

x, P, u, F, H, R, I = kalmanFilter.init()
FPS = 1000/60 #[ms]
frameIdx = 0
startROI = [1065, 255, 1246, 431] # startX, startY, endX, endY.
endROI = [377, 359, 713, 672] # startX, startY, endX, endY.
xlimL = 420
xlimU = 1100
# Load images:
bandMask = cv2.imread('Images/maskL.png',0)
_, mask = cv2.threshold(bandMask, 20, 255, cv2.THRESH_BINARY)
images_left = glob.glob('Images/sample_Stereo_conveyor_without_occlusions/left/*.png')
images_left = glob.glob('Images/Stereo_conveyor_with_occlusions/left/*.png')
#images_left = glob.glob('Images/Stereo_conveyor_without_occlusions/left/*.png')
assert images_left


# MAIN LOOP:
if __name__ == "__main__":
    try:
        while True:
            print("Camera Calibration done.. Running main program ...")
            BG = images_left[0]
            # Go through every image in folder:
            for fnamel in images_left:
                # Display and undistort each frame:
                imgL = cv2.imread(fnamel)
                imgL = cv2.remap(imgL, maplx, maply,interpolation=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=(0, 0, 0, 0))
                fnamer = fnamel
                fnamer.replace("left", "right").replace("Left", "Right")
                fnamer.replace("Left", "Right")
                imgR = cv2.imread(fnamer)
                imgR = cv2.remap(imgR, maprx, mapry,interpolation=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=(0, 0, 0, 0))
                
                # Frame differencing:
                frame = imgR.copy()
                if frameIdx == 0:
                    BG=frame
                    out = cv2.VideoWriter('Images/Output/output.mp4', -1, 20.0, (BG.shape[1], BG.shape[0]))
                    #cv2.imwrite('Images/Output/remap.png',BG) #debug.
                frame = imgL.copy()
                diff = cv2.absdiff(frame, BG)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                eroded = cv2.erode(thresh, None, iterations=4)
                dilated = cv2.dilate(eroded, None, iterations=4)
                dilated = cv2.bitwise_and(dilated, mask)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Sort and filter contours by area:
                areaThresh = 3000
                if len(contours) > 0:
                    cntsSorted = sorted(contours, key=cv2.contourArea) # index -1 to get last element.
                    cnt = max(cntsSorted, key = cv2.contourArea)
                    if cv2.contourArea(cnt) >= areaThresh:
                        (xbb, ybb, wbb, hbb) = cv2.boundingRect(cnt) # bb = bounding box.
                        if xbb > xlimU or xbb < xlimL:
                            cv2.rectangle(frame, (xbb, ybb), (xbb+wbb, ybb+hbb), (0, 255, 0), 2)
                            # Update Kalman Filter observation:
                            Z = np.array([[xbb], [ybb]])
                            x, P = kalmanFilter.update(x, P, Z, H, R)
                # Update kalman filter prediction:
                x, P = kalmanFilter.predict(x, P, F, u)
                cv2.circle(frame, (int(x[0][0]), int(x[3][0])), radius=0, color=(255, 0, 0), thickness=10)
                xpos = x[0][0]
                ypos = x[3][0]
                zpos = xpos/(BG.shape[1]/2)+0
                if int(xpos) < 370 or int(xpos) > 1250 or int(ypos) < 160 or int(ypos) > 580: # Reset kalman filter when prediction moves off screen. (also include > Xres and Yres?)
                       x, P, u, F, H, R, I = kalmanFilter.init()
                       
                # Draw object class and position on frame:
                color = (0, 0, 255)
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                cv2.rectangle(frame, (10,10), (250,200),color, thickness)
                #cv2.line(frame, (xlimL,0), (xlimL,BG.shape[0]), (0,255,0), thickness) #debug.
                #cv2.line(frame, (xlimU,0), (xlimU,BG.shape[0]), color, thickness) #debug.
                cv2.putText(frame, 'Object: Book', (30,50), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, 'X = ' + str(int(xpos)), (30,90), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, 'Y = ' + str(int(ypos)), (30,130), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, 'z = ' + str(round(zpos, 1)) + ' m', (30,170), font, fontScale, color, thickness, cv2.LINE_AA)

                # Display resulting frames:
                out.write(frame.astype('uint8'))
                cv2.imshow("frame", frame)
                cv2.imshow("ObjectDetector", dilated)
                cropStart = imgL[startROI[1]:startROI[1]+(startROI[3]-startROI[1]), startROI[0]:startROI[0]+(startROI[2]-startROI[0])]
                cropEnd = imgL[endROI[1]:endROI[1]+(endROI[3]-endROI[1]), endROI[0]:endROI[0]+(endROI[2]-endROI[0])]
                cv2.imshow("cropStart", cropStart)
                #cv2.imshow("cropEnd", cropEnd)
                frameIdx = frameIdx + 1
                
                cv2.waitKey(int(FPS)) #Run at 60 FPS-ish.
            cv2.destroyAllWindows()
            break
    except KeyboardInterrupt:
        print("n\Exit")

out.release()
print("End of program.")
# END
