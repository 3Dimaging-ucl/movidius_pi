# USAGE
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --display 1
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --confidence 0.5 --display 1

from mvnc import mvncapi as mvnc
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import numpy as np
import time
import cv2

CLASSES = ("background", "aeroplane", "bicycle", "bird",
	"boat", "bottle", "bus", "car", "cat", "chair", "cow",
	"diningtable", "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (900, 900)

DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]

def preprocess_image(input_image):
	preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
	preprocessed = preprocessed - 127.5
	preprocessed = preprocessed * 0.007843
	preprocessed = preprocessed.astype(np.float16)

	return preprocessed

def predict(image, graph):
	image = preprocess_image(image)
	graph.LoadTensor(image, None)
	(output, _) = graph.GetResult()

	num_valid_boxes = output[0]
	predictions = []

	for box_index in range(num_valid_boxes):
		base_index = 7 + box_index * 7

		if (not np.isfinite(output[base_index]) or
			not np.isfinite(output[base_index + 1]) or
			not np.isfinite(output[base_index + 2]) or
			not np.isfinite(output[base_index + 3]) or
			not np.isfinite(output[base_index + 4]) or
			not np.isfinite(output[base_index + 5]) or
			not np.isfinite(output[base_index + 6])):
			continue

		(h, w) = image.shape[:2]
		x1 = max(0, int(output[base_index + 3] * w))
		y1 = max(0, int(output[base_index + 4] * h))
		x2 = min(w,	int(output[base_index + 5] * w))
		y2 = min(h,	int(output[base_index + 6] * h))

		pred_class = int(output[base_index + 1])
		pred_conf = output[base_index + 2]
		pred_boxpts = ((x1, y1), (x2, y2))

		prediction = (pred_class, pred_conf, pred_boxpts)
		predictions.append(prediction)

	return predictions

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph", required=True,
	help="path to input graph file")
ap.add_argument("-c", "--confidence", default=.5,
	help="confidence threshold")
ap.add_argument("-d", "--display", type=int, default=0,
	help="switch to display image on screen")
args = vars(ap.parse_args())

print("[INFO] finding NCS devices...")
devices = mvnc.EnumerateDevices()

if len(devices) == 0:
	print("[INFO] No devices found. Please plug in a NCS")
	quit()

print("[INFO] found {} devices. device0 will be used. "
	"opening device0...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()

print("[INFO] loading the graph file into RPi memory...")
with open(args["graph"], mode="rb") as f:
	graph_in_memory = f.read()

print("[INFO] allocating the graph on the NCS...")
graph = device.AllocateGraph(graph_in_memory)

print("[INFO] starting the video stream and FPS counter...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(1)
fps = FPS().start()

while True:
	try:
		frame = vs.read()
		image_for_result = frame.copy()
		image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)

		predictions = predict(frame, graph)

		for (i, pred) in enumerate(predictions):
			(pred_class, pred_conf, pred_boxpts) = pred

			if pred_conf > args["confidence"]:
				print("[INFO] Prediction #{}: class={}, confidence={}, "
					"boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
					pred_boxpts))

				if args["display"] > 0:
					label = "{}: {:.2f}%".format(CLASSES[pred_class],
						pred_conf * 100)

					(ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
					ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
					ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
					(startX, startY) = (ptA[0], ptA[1])
					y = startY - 15 if startY - 15 > 15 else startY + 15

					cv2.rectangle(image_for_result, ptA, ptB,
						COLORS[pred_class], 2)
					cv2.putText(image_for_result, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)

		if args["display"] > 0:
			cv2.imshow("Output", image_for_result)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break

		fps.update()
	
	except KeyboardInterrupt:
		break

	except AttributeError:
		break

fps.stop()

if args["display"] > 0:
	cv2.destroyAllWindows()

vs.stop()

graph.DeallocateGraph()
device.CloseDevice()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))