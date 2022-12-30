import cv2
#importing open cv
import mediapipe as mp
#for hand recognition
import time

cap = cv2.VideoCapture(0)
#capturing the video through webcam number zero

mpHands = mp.solutions.hands #hands object
hands = mpHands.Hands(max_num_hands=3)


mpDraw = mp.solutions.drawing_utils
#this line is drawing the hand

while True:
    success, img = cap.read()
    #reading the current frame of the image
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #coverting image to rgb
    results=hands.process(imgRGB)
    #processing the hand out of image

    if results.multi_hand_landmarks:
        #if we find a hand landmark (hand bascially) in results
        for hand in results.multi_hand_landmarks:
            #all the hands in this frame
            for id,lm in enumerate(hand.landmark):
                #each hand consisits of 21 landmarks with a given id
                # and lm object consisiting of x y z position
                h,w,c= img.shape
                #hwight width
                cx,cy = int(lm.x*w),int(lm.y*h)
                #correct positon in spcae
                #bit of tweaking
                if id==12:
                    cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
            #finally drawing the hand
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    #displaying the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

