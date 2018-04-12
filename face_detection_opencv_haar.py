import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image file')
ap.add_argument('-v', '--video', help='path to video file')
args = ap.parse_args()

image_file = args.image
video_file = args.video

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if image_file:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("face detection - opencv haar", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

if video_file:
    webcam = cv2.VideoCapture(video_file)
else:
    webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while(webcam.isOpened()):
    status, frame = webcam.read()
    if not status:
        print("Could not read frame")
        exit()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("face detection - opencv haar", frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
