import sys
import os

print("\n" + "="*60)
print("1. 현재 런타임이 인식하는 SYS.PATH 목록")
print("="*60)
for path in sys.path:
    print(path)

print("\n" + "="*60)
print("2. 실제 로드된 모듈의 물리적 파일 위치 (__file__)")
print("="*60)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


try:
    # 1단계: utils 패키지 위치 확인
    from isaacsim.core.experimental.utils.transform import euler_angles_to_quaternion
    print(f"[module]  transform 위치: {euler_angles_to_quaternion.__module__}")
    
    
except Exception as e:
    print(f"출력 중 에러 발생 (인터프리터 환경을 확인하세요): {e}")
print("="*60 + "\n")