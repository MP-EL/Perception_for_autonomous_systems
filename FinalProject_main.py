# IMPORT NECCESSARY LIBRARIES:
import numpy as np
import cv2
import sys
import glob
import os
import time
import math
# IMPORT OTHER PY FILES:
sys.path.append('External Functions')
import stereoCamera
import kalmanFilter


#FUNCTION DEFINITIONS:
def calcOpticalFlow(lastFrame, currentFrame):
    # Make copy of current frame:
    img = currentFrame.copy()
    # Convert frames to grayscale for feature extraction:
    gray1 = cv2.cvtColor(lastFrame, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(currentFrame, cv2.COLOR_RGB2GRAY)
    # Find features in last frame:
    feat1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7)
    # Find features in current frame:
    if feat1 is not None:
        feat2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat1, None)
        # Draw lines between similar features in current and last frame:
        tolerance = 1
        maxLen = 10
        x = []
        y = []
        DX =[]
        DY = []
        for i in range(len(feat1)):
            f10=int(feat1[i][0][0])
            f11=int(feat1[i][0][1])
            f20=int(feat2[i][0][0])
            f21=int(feat2[i][0][1])
            dx = f20-f10
            dy = f21-f11
            lineLen = math.sqrt((dx)**2 + (dy)**2)
            if (f10 < f20-tolerance or f10 > f20+tolerance) and (f11 < f21-tolerance or f11 > f21+tolerance) and lineLen < maxLen:
                cv2.line(img, (f10,f11), (f20, f21), (0, 255, 0), 2)
                cv2.circle(img, (f10, f11), 5, (0, 255, 0), -1)
                x.append(f20)
                y.append(f21)
                DX.append(dx)
                DY.append(dy)
        if len(x) and len(y) > 0:
            cv2.circle(img, (int(sum(x)/len(x)), int(sum(y)/len(y))), 5, (0, 0, 255), -1)
            avgDX = round(sum(DX)/len(DX))
            avgDY = round(sum(DY)/len(DY))
        else:
            avgDX = 0
            avgDY = 0
        #cv2.imshow('img',img)
        return img,avgDX,avgDY
    else:
        return img,0,0

def siftDetector(sift, lastFrame, currentFrame):
    img1 = cv2.cvtColor(lastFrame, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(currentFrame, cv2.COLOR_RGB2GRAY)
    # Initiate SIFT detector
    #sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.18 * n.distance: #0.75
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Display image
    #cv2.imshow('5',img3)
    return img3

def accelLimiter(dx, dy, lastdx, lastdy):
    # This function limits the rate of change observed through optical flow, based on 3 assumptions.
    # 1. Object cannot accelerate rapidly.
    if abs(dx > 3):
        dx = 3
    if abs(dy > 2):
        dy = 1
    # 2. Object cannot stand still on conveyor.
    if abs(dx == 0):
        dx = 2
        dx = lastdx
    if abs(dy == 0):
        dy = 2
        dy = lastdy
    if dy>dx+1 or dy<dx-1:
        dy = abs(dx)-1
    # 3. Object cannot change direction.
    ndx = abs(dx)*-1
    ndy = abs(dy)
    return ndx, ndy
    

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
sift = cv2.SIFT_create()
FPS = 1000/60 #[ms]
frameIdx = 0
startROI = [1065, 255, 1246, 431] # startX, startY, endX, endY.
endROI = [377, 359, 713, 672] # startX, startY, endX, endY.
xlimL = 420
xlimU = 1100 #1070
startX = 0
startY = 0
lastdx = 2
lastdy = 2
timerLimit = 2
startTime = time.time()
# Load images:
bandMask = cv2.imread('Images/maskR.png',0)
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
                OG = imgR.copy()
                frame = imgR.copy()
                if frameIdx == 0:
                    BG=frame
                    lastFrame = np.zeros((BG.shape[0], BG.shape[1], 3), dtype = "uint8")
                    out = cv2.VideoWriter('Images/Output/output.mp4', -1, 20.0, (BG.shape[1], BG.shape[0]))
                    #cv2.imwrite('Images/Output/remap.png',BG) #debug.
                frame = imgL.copy()
                diff = cv2.absdiff(frame, BG)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                # Blur image to reduce noise:
                #gray = cv2.GaussianBlur(gray, (5,5), 0)
                _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                eroded = cv2.erode(thresh, None, iterations=6)
                dilated = cv2.dilate(eroded, None, iterations=6)
                dilated = cv2.bitwise_and(dilated, mask)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Sort and filter contours by area:
                areaThresh = 3000
                if len(contours) > 0:
                    cntsSorted = sorted(contours, key=cv2.contourArea) # index -1 to get last element.
                    cnt = max(cntsSorted, key = cv2.contourArea)
                    if cv2.contourArea(cnt) >= areaThresh:
                        (xbb, ybb, wbb, hbb) = cv2.boundingRect(cnt) # bb = bounding box.
                        if True:#xbb > xlimU or xbb < xlimL:
                            if time.time()-startTime > timerLimit:
                                startX = 0
                                startY = 0   
                            startTime = time.time()
                            cv2.rectangle(frame, (xbb, ybb), (xbb+wbb, ybb+hbb), (0, 255, 0), 2)
                            # sift and optical flow:
                            ROImask = np.zeros(frame.shape[:2], dtype="uint8")
                            cv2.rectangle(ROImask, (xbb, ybb), (xbb+wbb, ybb+hbb), 255, -1)
                            ROImask = cv2.bitwise_and(ROImask, dilated)
                            maskedObject = cv2.bitwise_and(OG,OG, mask=ROImask)
                            #cv2.imshow('sift', siftDetector(sift, lastFrame, maskedObject)) #debug.
                            OF,avgDX,avgDY = calcOpticalFlow(lastFrame, maskedObject)
                            cv2.imshow('OF', OF) #debug.
                            lastFrame = maskedObject
                            # Update Kalman Filter observation:
                            #Z = np.array([[xbb+(wbb)], [ybb+(hbb)]])
                            #x, P = kalmanFilter.update(x, P, Z, H, R)
                            #cv2.circle(frame, (int(xbb+(wbb)), int(ybb+(hbb))), radius=0, color=(0, 0, 255), thickness=10)
                            
                            if startX == 0 and startY == 0: 
                                startX = xbb
                                startY = ybb
                                #Z = np.array([[xbb+(wbb/2)], [ybb+(hbb/2)]])
                                Z = np.array([[startX], [startY]])
                            else:
                                print('avgDX=' + str(avgDX))
                                print('avgDY=' + str(avgDY))
                                ndx, ndy = accelLimiter(avgDX, avgDY, lastdx, lastdy)
                                ndx = -4
                                ndy = 1
                                startX = startX + ndx
                                startY = startY + ndy
                                lastdx = ndx
                                lastdy = ndy
                                Z = np.array([[startX], [startY]])
                            x, P = kalmanFilter.update(x, P, Z, H, R)
                            
                # Update kalman filter prediction:
                x, P = kalmanFilter.predict(x, P, F, u)
                xpos = x[0][0]
                ypos = x[3][0]
                zpos = xpos/(BG.shape[1]/2)+0
                cv2.circle(frame, (int(xpos), int(ypos)), radius=0, color=(255, 0, 0), thickness=10)
                cv2.circle(frame, (int(startX), int(startY)), radius=0, color=(0, 0, 255), thickness=10)
                if int(xpos) < 370 or int(xpos) > 1250 or int(ypos) < 160 or int(ypos) > 580 or xbb<xlimL: # Reset kalman filter when prediction moves off screen. (also include > Xres and Yres?)
                       x, P, u, F, H, R, I = kalmanFilter.init()
                       startX = 0
                       startY = 0
                       
                # Draw object class and position on frame:
                color = (0, 0, 255)
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                cv2.rectangle(frame, (10,10), (250,200),color, thickness)
                cv2.line(frame, (xlimL,0), (xlimL,BG.shape[0]), (0,255,0), thickness) #debug.
                cv2.line(frame, (xlimU,0), (xlimU,BG.shape[0]), color, thickness) #debug.
                cv2.putText(frame, 'Object: Book', (30,50), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, 'X = ' + str(int(xpos)), (30,90), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, 'Y = ' + str(int(ypos)), (30,130), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, 'z = ' + str(round(zpos, 1)) + ' m', (30,170), font, fontScale, color, thickness, cv2.LINE_AA)

                # Display resulting frames:
                out.write(frame.astype('uint8'))
                cv2.imshow("frame", frame)
                cv2.imshow("diff", dilated)
                cropStart = imgL[startROI[1]:startROI[1]+(startROI[3]-startROI[1]), startROI[0]:startROI[0]+(startROI[2]-startROI[0])]
                cropEnd = imgL[endROI[1]:endROI[1]+(endROI[3]-endROI[1]), endROI[0]:endROI[0]+(endROI[2]-endROI[0])]
                #cv2.imshow("cropStart", cropStart)
                #cv2.imshow("cropEnd", cropEnd)
                #cv2.imshow('MaskSift', lastFrame)
                frameIdx = frameIdx + 1
                
                cv2.waitKey(int(FPS)) #Run at 60 FPS-ish.
            cv2.destroyAllWindows()
            break
    except KeyboardInterrupt:
        print("n\Exit")

out.release()
print("End of program.")
# END
