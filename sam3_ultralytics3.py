import cv2

from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.utils.plotting import Annotator, colors

"""
다중 쿼리를 위한 이미지 특징 재사용
이미지 특징을 한 번 추출하고 여러 segment 쿼리에 재사용하여 효율성을 향상시킵니다.
"""


# Initialize predictors
overrides = dict(
    conf=0.50, task="segment", mode="predict", model="sam3.pt", verbose=False
)
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor2 = SAM3SemanticPredictor(overrides=overrides)

# Extract features from the first predictor
source = "path/to/image.jpg"
predictor.set_image(source)
src_shape = cv2.imread(source).shape[:2]

# Setup second predictor and reuse features
predictor2.setup_model()

# Perform inference using shared features with text prompt
masks, boxes = predictor2.inference_features(
    predictor.features, src_shape=src_shape, text=["person"]
)

# Perform inference using shared features with bounding box prompt
masks, boxes = predictor2.inference_features(
    predictor.features, src_shape=src_shape, bboxes=[[439, 437, 524, 709]]
)

# Visualize results
if masks is not None:
    masks, boxes = masks.cpu().numpy(), boxes.cpu().numpy()
    im = cv2.imread(source)
    annotator = Annotator(im, pil=False)
    annotator.masks(masks, [colors(x, True) for x in range(len(masks))])

    cv2.imshow("result", annotator.result())
    cv2.waitKey(0)
