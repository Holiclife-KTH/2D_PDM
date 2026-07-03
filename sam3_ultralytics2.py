from ultralytics.models.sam import SAM3SemanticPredictor

"""
이미지 예시 기반 세분화
바운딩 박스를 시각적 프롬프트로 사용하여 모든 유사한 인스턴스를 찾습니다. 이는 또한 SAM3SemanticPredictor 개념 기반 매칭을 위해 필요합니다.
"""

# Initialize predictor
overrides = dict(
    conf=0.25, task="segment", mode="predict", model="sam3.pt", half=True, save=True
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image
predictor.set_image("path/to/image.jpg")

# Provide bounding box examples to segment similar objects
results = predictor(bboxes=[[480.0, 290.0, 590.0, 650.0]])

# Multiple bounding boxes for different concepts
results = predictor(bboxes=[[539, 599, 589, 639], [343, 267, 499, 662]])
