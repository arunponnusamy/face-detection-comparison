import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained caffemodel file")
ap.add_argument("-th", "--threshold", type=float, default=0.5,
                help="probability threshold to ignore false detections")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

webcam = cv2.VideoCapture(0)
if(not webcam.isOpened()):
    print("Error opening webcam")
    exit()

while(webcam.isOpened()):
    status, frame = webcam.read()
    
    if(not status):
        print("Error reading frame")
        exit()
        
    (h, w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300,300), (104.0,177.0,123.0))

    net.setInput(blob)
    
    faces = net.forward()

    for i in range(0, faces.shape[2]):
        confidence = faces[0,0,i,2]

        if confidence < args["threshold"]:
            continue

        box = faces[0,0,i,3:7] * np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype('int')

        text = "face " + "{:.2f}%".format(confidence * 100)

        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        cv2.putText(frame, text, (startX,startY-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)
        
    cv2.imshow("output", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
