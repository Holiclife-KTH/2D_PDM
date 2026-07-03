from ultralytics.models.sam import SAM3SemanticPredictor

"""
텍스트 기반 컨셉 세분화
텍스트 설명을 사용하여 개념의 모든 인스턴스를 찾아 segment합니다. 텍스트 프롬프트는 SAM3SemanticPredictor 인터페이스가 필요합니다.
"""

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="models/sam3.pt",
    half=True,  # Use FP16 for faster inference
    save=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image("data/pcd2.png")

# Query with multiple text prompts
results = predictor(text=["box", "magazine"])

print(results)
