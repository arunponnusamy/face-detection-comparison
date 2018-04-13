# import required packages
import cv2
import argparse

# handle command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image file')
ap.add_argument('-v', '--video', help='path to video file')
args = ap.parse_args()

image_file = args.image
video_file = args.video

# initialize face detector
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# process image input (if provided)
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

# check for video input / webcam
if video_file:
    webcam = cv2.VideoCapture(video_file)
else:
    webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# process frames one by one 
while(webcam.isOpened()):

    # read frame
    status, frame = webcam.read()
    if not status:
        print("Could not read frame")
        exit()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply face detection
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # draw boxes over detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # display output frame
    cv2.imshow("face detection - opencv haar", frame)

    # press 'Q' to stop the program
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
