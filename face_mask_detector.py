# import the necessary packages
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import cv2
import os
import argparse
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-t", "--fps", type=int, default=2, help="path to (optional) output video file")
ap.add_argument("--display", action='store_true')
args = vars(ap.parse_args())


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "models/deploy.prototxt"
modelPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
mask_detector = load_model('models/mask_detector.model')

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = args["fps"]

writer = None

# grab the frame from the threaded video stream
t1 = time.time()
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame
	frame = imutils.resize(frame, width=800)
	(h, w) = frame.shape[:2]

	t2 = time.time()
	if t2-t1 > 1/fps:
		t1 = t2

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize faces in the input image
		face_detector.setInput(imageBlob)
		detections = face_detector.forward()
		
		if len(detections) > 0:

			faces, centroids, bbox, results, radius = [], [], [], [], []

			for i in range(0, detections.shape[2]):
			    confidence = detections[0, 0, i, 2]

			    if confidence > 0.7:
			        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			        (startX, startY, endX, endY) = box.astype("int")
			        x = int((endX-startX)/2 + startX)
			        y = int((endY-startY)/2 + startY)

			        face = frame[startY:endY, startX:endX]
			        (fH, fW) = face.shape[:2]

			        # ensure the face width and height are sufficiently large
			        if fW < 20 or fH < 20:
			        	continue

			        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			        face = cv2.resize(face, (224, 224))
			        face = img_to_array(face)
			        face = preprocess_input(face)
			        face = np.expand_dims(face, axis=0)
			        faces.append(face)
			        centroids.append((x,y))
			        radius.append(int((endX-startX)/5*3))
			        bbox.append((startX, startY, endX, endY))

			if len(faces) > 0:
				out = []
				for (face, center, rad, box) in zip(faces, centroids, radius, bbox):
					result = mask_detector.predict(face)
					(mask, withoutMask) = result[0]
					(startX, startY, endX, endY) = box

					if mask > withoutMask:
					    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
					    out.append('MASK')
					else:
					    cv2.circle(frame, center, rad, (0, 0, 255), 2)
					    out.append('NO MASK')
				print(out)

	if args["display"]:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

	# if an output video file path has been supplied and the video writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, fps, (frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output video file
	if writer is not None:
		writer.write(frame)

	if args["display"]:
		if key == ord('q'):
			break
