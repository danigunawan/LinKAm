from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2

"""
We read in our image and do some resizing. We then apply our EAST text detection
on a "blob" of our image. From that we get scores (probabilities) of there being
text in each geometry (portion of the image). decode_predictions will remove
geometries where the probabiilties are not high enough and non_max_suppression
will remove overlaps between squares. For each of the remaining boxes, we will
run our TESSERACT to recognize the text. Return the text with the biggest
boundary box."""

def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):
			if scoresData[x] < 0.5:
				continue
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)

image = cv2.imread('images/example_03.jpg')
orig = image.copy()
(origH, origW) = image.shape[:2]

(newW, newH) = (320, 320)
rW = origW / float(newW)
rH = origH / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

results = []
for (startX, startY, endX, endY) in boxes:
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(origW, endX)
	endY = min(origH, endY)

	roi = orig[startY:endY, startX:endX]

	config = ("-l eng --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)

	results.append(((startX, startY, endX, endY), text))

results = sorted(results, key=lambda r:r[0][1])

def area(ax, ay, bx, by):
	return abs(ax - bx) * abs(ay - by)

maxText = ''
maxArea = 0
for ((startX, startY, endX, endY), text) in results:
	currArea = area(startX, startY, endX, endY)
	if currArea > maxArea:
		maxArea = currArea
		maxText = text
print(maxText)
