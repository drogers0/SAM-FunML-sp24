import sys
import cv2
import statistics
	@@ -12,6 +17,7 @@
from helpers import *
import workbook

sys.path.append("..")

try:
	@@ -22,10 +28,13 @@
plt.rcParams['keymap.grid'].remove('g')
plt.rcParams['keymap.home'].remove('r')

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)

predictor = SamPredictor(sam)

names  = np.load("samples.npy", allow_pickle=True)
labels = np.load("labels.npy", allow_pickle=True)
	@@ -71,7 +80,7 @@

    if len(image.shape) == 2:
        image = cv2.cvtColor((np.array(((image + 1) / 2) * 255, dtype='uint8')), cv2.COLOR_GRAY2RGB)
    predictor.set_image(image)


    while True:
	@@ -146,13 +155,19 @@ def onclick(event):
                    input_point = np.concatenate((green, red))
                    input_label = np.concatenate(([1] * len(green), [0] * len(red)))

                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                    mask = masks[0]