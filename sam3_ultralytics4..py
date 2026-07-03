from ultralytics import SAM

"""
SAM 2 스타일 시각적 프롬프트
기본 SAM 인터페이스는 SAM 2와 정확히 동일하게 작동하며, 시각적 프롬프트(점, 상자 또는 마스크)로 표시된 특정 영역만 segment합니다.
"""


model = SAM("sam3.pt")

# Single point prompt - segments object at specific location
results = model.predict(source="path/to/image.jpg", points=[900, 370], labels=[1])
results[0].show()

# Multiple points - segments single object with multiple point hints
results = model.predict(
    source="path/to/image.jpg", points=[[400, 370], [900, 370]], labels=[1, 1]
)

# Box prompt - segments object within bounding box
results = model.predict(source="path/to/image.jpg", bboxes=[100, 150, 300, 400])
results[0].show()
