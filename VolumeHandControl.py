from turtle import clear
import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math


#pycaw library for getting volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCamera, hCamera = 600, 600

cap=cv2.VideoCapture(0)
cap.set(3, wCamera)#propid number 3 is width
cap.set(4, hCamera)#propid number 4 is height

ctime=0
ptime=0
cv2.destroyAllWindows()
handDetetctor = htm.handDetector(maxHands = 1, detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
# print(minVol, maxVol)

while True:
    success, img = cap.read()
    handDetetctor.findHands(img)
    lmlist= handDetetctor.findPosition(img, draw=False)
    # print(lmlist)
    if len(lmlist)!=0:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1+x2)//2 , (y1+y2)//2
        length = math.hypot(x2-x1, y2-y1)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 10, cv2.FILLED)
        cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        if length<50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

        #hand length is 50 to 300
        #volume range is -65 to zero
        vol = np.interp(length, [50,300], [minVol,maxVol])
        #converting the range 50 to 300 TO minVol to maxVol
        # cv2.putText(img, f'{str(int(vol))}%', (800,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
        volume.SetMasterVolumeLevel(vol, None)

    ctime=time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {str(int(fps))}', (40,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
