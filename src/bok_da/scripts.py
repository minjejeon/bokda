import os
import shutil
import sys
from importlib.metadata import distribution

def copy_examples():
    """
    현재 작업 디렉터리에 examples/ 노트북·데이터 폴더를 복사합니다.
    """
    # 1) 설치된 dist 패키지에서 examples 폴더 경로 찾기
    dist = distribution("bok-da")
    pkg_root = dist.locate_file("examples")
    if not pkg_root.exists():
        print("[bok-da] 패키지에 examples 폴더가 없습니다.")
        sys.exit(1)

    # 2) 복사 대상 경로
    target = os.path.join(os.getcwd(), "examples")
    if os.path.exists(target):
        print(f"[bok-da] '{target}' 이미 존재합니다. 삭제 후 다시 시도하세요.")
        return

    # 3) 복사
    shutil.copytree(pkg_root, target)
    print(f"[bok-da] 예제를 '{target}' 에 복사했습니다.")