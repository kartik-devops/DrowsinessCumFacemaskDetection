
import numpy as np
import cv2
import dlib
import face_recognition
import imutils
from imutils import face_utils
import distance
from playsound import playsound
from scipy.spatial import distance
def eye_aspect_ratio(eye):
    a=distance.euclidean(eye[1],eye[5])
    b=distance.euclidean(eye[2],eye[4])
    c=distance.euclidean(eye[0],eye[3])
    EAR = (a+b)/(2*c)
    return EAR
def mouth_aspect_ratio(mouth):
    a=distance.euclidean(mouth[5],mouth[8])
    b=distance.euclidean(mouth[1],mouth[10])
    c=distance.euclidean(mouth[0],mouth[6]);
    MAR=(a+b)/(2*c)
    return MAR
cap=cv2.VideoCapture(0)
detector= dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(left_eye_start , left_eye_end)=face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(right_eye_start , right_eye_end)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouth_start , mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
EAR_threshold=0.22
MAR_threshold = 0.8
eye_consec_frames=25
yawn_status=False
yawn_cnt=0
eye_cnt=0
while True:
    ret,frame = cap.read()
    frame = imutils.resize(frame, width=680)
    pic=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawn_status
    var=detector(pic,0)
    for i in var :
        shape=predictor(pic,i)
        shape=face_utils.shape_to_np(shape)
        left_eye=shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]
        mouth=shape[mouth_start:mouth_end]
        leftear = eye_aspect_ratio(left_eye)
        rightear=eye_aspect_ratio(right_eye)
        ear=(leftear+rightear)/2
        mouear=mouth_aspect_ratio(mouth)
        left_hull=cv2.convexHull(left_eye)
        right_hull=cv2.convexHull(right_eye)
        mouth_hull=cv2.convexHull(mouth)
        cv2.drawContours(frame,[left_hull],-1,(0,255,255),1)
        cv2.drawContours(frame,[right_hull],-1,(0,255,255),1)
        cv2.drawContours(frame,[mouth_hull],-1,(0,255,0),1)
        if ear < EAR_threshold:
            #eye_cnt+=1
            cv2.putText(frame, "Eyes blinking / Feeling drowsy", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 328), 2)
            eye_cnt+=1
            if eye_cnt >=eye_consec_frames:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                playsound("scary.mp3")

        else:
            eye_cnt=0

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if mouear > MAR_threshold:
            cv2.putText(frame, "Yawning ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawnStatus = True

        else:
            yawn_status=False

        if prev_yawn_status==True and yawn_status==False:
            yawn_cnt+=1
        cv2.putText(frame, "MAR: {:.2f}".format(mouear), (480, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("output",frame)
    button = cv2.waitKey(1) & 0xFF
    if button == ord("q"):
        break



cv2.destroyAllWindows()
cap.release()







