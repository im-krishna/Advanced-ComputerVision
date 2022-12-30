import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, mode=False, complexity = 1, sland = True, eseg =False, smseg = True, minDconf = 0.5, minTconf = 0.5):

        self.mode=mode
        self.complexity=complexity
        self.sland=sland
        self.eseg=eseg
        self.smseg=smseg
        self.minDconf=minDconf
        self.minTconf=minTconf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(mode,complexity,sland,eseg,smseg,minDconf,minTconf)
        self.mpDraw = mp.solutions.drawing_utils


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks , self.mpPose.POSE_CONNECTIONS)

    def findPosition(self, img, draw = True):
        lmList=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c=img.shape
                cx, cy=int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw and id==14:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        return lmList





def main():
    cap = cv2.VideoCapture(0)
    ptime=0
    ctime=0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        detector.findPose(img)
        list = detector.findPosition(img)
        if len(list)!=0:
            print(list[10])
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(20,40),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
