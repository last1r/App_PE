import cv2
import numpy as np
import math
import mediapipe as mp
from tkinter import *
import time

#----------------------------------------------------POSEDETECTOR------------------------------------------------------------
class poseDetector():

    def __init__(self, static_image_mode=False, model_complexity=1,
                 smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity,
                                     self.smooth_landmarks, self.enable_segmentation,
                                     self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Получаем точки маркеров

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # калькулятор Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # рисуем
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 0), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 0), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 0), 2)

        return angle
#----------------------------------------------------------------------------------------------------------------------------

cap = cv2.VideoCapture('PoseVidios/Jumping jack.mp4')
detector = poseDetector()

#---------------------------------------------------Jumping_jack-------------------------------------------------------------
def Jumping_jack(cap):
    count = 0
    dir = 0

    while True:
        success, img = cap.read()
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            leg1 = detector.findAngle(img, 12, 24, 28)
            leg2 = detector.findAngle(img, 11, 23, 27)

            arm1 = detector.findAngle(img, 24, 12, 16)
            arm2 = detector.findAngle(img, 23, 11, 15)

            leg1_per = np.interp(leg1, (185, 200), (0, 100))

            leg2_per = np.interp(leg2, (155, 170), (100, 0))

            arm1_per = np.interp(arm1, (20, 165), (0, 100))

            arm2_per = np.interp(arm2, (340, 185), (100, 0))

            if leg1_per == 100 and leg2_per == 100 and arm1_per == 100 and arm2_per == 100:
                if dir == 0:
                    count += 0.5
                    dir += 1

            if leg1_per == 0 and leg2_per == 0 and arm1_per == 0 and arm2_per == 0:
                if dir == 1:
                    count += 0.5
                    dir = 0

        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), 15)
        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(img, (25, 158), (150, 100), (0, 255, 127), -1)
        cv2.putText(img, str(int(count)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 10)
        cv2.putText(img, 'Jumping jack', (37, 138), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)
#----------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------biceps_exercise_left------------------------------------------------------
def biceps_exercise_left(cap):
    count = 0
    dir = 0
    while True:
        success, img = cap.read()

        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 11, 13, 15)

            per = np.interp(angle, (210, 300), (0, 100))
            bar = np.interp(angle, (220, 300), (460, 100))

            if per == 100:
                if dir == 0:
                    count += 0.5
                    dir += 1
            if per == 0:
                if dir == 1:
                    count += 0.5
                    dir = 0

        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), 15)
        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(img, (25, 158), (150, 100), (0, 255, 127), -1)
        cv2.putText(img, str(int(count)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 10)
        cv2.putText(img, 'biceps', (60, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(img, 'exercise', (55, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)
#----------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------biceps_exercise_right-----------------------------------------------------
def biceps_exercise_right(cap):
    count = 0
    dir = 0
    while True:
        success, img = cap.read()

        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 12, 14, 16)

            per = np.interp(angle, (210, 300), (0, 100))
            bar = np.interp(angle, (220, 300), (460, 100))

            if per == 100:
                if dir == 0:
                    count += 0.5
                    dir += 1
            if per == 0:
                if dir == 1:
                    count += 0.5
                    dir = 0

        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), 15)
        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(img, (25, 158), (150, 100), (0, 255, 127), -1)
        cv2.putText(img, str(int(count)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 10)
        cv2.putText(img, 'biceps', (60, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(img, 'exercise', (55, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)
#----------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------squats---------------------------------------------------------------
def squats(cap):
    count = 0
    dir = 0
    while True:
        success, img = cap.read()
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            angle = detector.findAngle(img, 24, 26, 28)
            angle1 = detector.findAngle(img, 23, 25, 27)

            per = np.interp(angle, (200, 295), (0, 100))
            per1 = np.interp(angle1, (200, 295), (0, 100))

            if per == 100 and per1 == 100:
                if dir == 0:
                    count += 0.5
                    dir += 1
            if per == 0 and per1 == 0:
                if dir == 1:
                    count += 0.5
                    dir = 0

        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), 15)
        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(img, (25, 158), (150, 100), (0, 255, 127), -1)
        cv2.putText(img, str(int(count)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 10)
        cv2.putText(img, 'squats', (60, 138), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)
#------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------push_ups----------------------------------------------------------------
def push_ups(cap):
    count = 0
    dir = 0
    while True:
        success, img = cap.read()
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            angle = detector.findAngle(img, 11, 13, 15)
            angle1 = detector.findAngle(img, 12, 14, 16)

            per = np.interp(angle, (190, 280), (100, 0))

            print(angle, per)

            if per == 100:
                if dir == 0:
                    count += 0.5
                    dir += 1
            if per == 0:
                if dir == 1:
                    count += 0.5
                    dir = 0

        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), 15)
        cv2.rectangle(img, (25, 15), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(img, (25, 158), (150, 100), (0, 255, 127), -1)
        cv2.putText(img, str(int(count)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 10)
        cv2.putText(img, 'push ups', (48, 138), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)
#------------------------------------------------------------------------------------------------------------------------------


