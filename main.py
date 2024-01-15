import cv2
import numpy as np
from scipy.spatial import distance
import dlib

# 모델 로드
detect = './detect/'
prototxt = detect + 'deploy.prototxt'
resnet = detect + 'res10_300x300_ssd_iter_140000.caffemodel'
face_model = cv2.dnn.readNet(resnet, prototxt)

landmark = detect + 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(landmark)

# 카메라 로드 (웹캠)
cam = cv2.VideoCapture(0)

# 최소 인식률
minimum_confidence = 0.5


# 눈 감은 정도 확인 함수
def ear(eyes):
    a = distance.euclidean(eyes[1], eyes[5])
    b = distance.euclidean(eyes[2], eyes[4])
    c = distance.euclidean(eyes[0], eyes[3])
    eye_aspect_ratio = (a + b) / (2.0 * c)
    return eye_aspect_ratio


while True:
    ret, frame = cam.read()

    if not ret:
        print('not ret')
        break

    # 얼굴 먼저 인식
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 124.0))
    face_model.setInput(blob)
    detections = face_model.forward()
    faces = []

    for i in range(0, detections.shape[2]):
        # 얼굴 인식 확률 추출
        confidence = detections[0, 0, i, 2]
        if confidence > minimum_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype('int')
            faces.append(dlib.rectangle(start_x, start_y, end_x, end_y))

            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(w - 1, end_x), min(h - 1, end_y)

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # 얼굴 인식 후 랜드마크로 눈 인식
    for face in faces:
        face_landmark = predictor(frame, face)

        left_eyes = []
        for left in range(36, 42):  # parts : 전체 구하기 / part(n) : n 부분 구하기
            left_eyes.append([face_landmark.part(left).x, face_landmark.part(left).y])

        right_eyes = []
        for right in range(42, 48):
            right_eyes.append([face_landmark.part(right).x, face_landmark.part(right).y])

        for i in range(6):
            cv2.circle(frame, (left_eyes[i][0], left_eyes[i][1]), 2, (255, 0, 0))
            cv2.circle(frame, (right_eyes[i][0], right_eyes[i][1]), 2, (255, 0, 0))

        left_eye, right_eye = ear(left_eyes), ear(right_eyes)
        if left_eye < 0.145 and right_eye < 0.145:
            print(f'left eye : {left_eye}')
            print(f'right eye : {right_eye}')

    cv2.imshow('test', frame)

    if cv2.waitKey(33) == 27:
        break

cam.release()
cv2.destroyAllWindows()
