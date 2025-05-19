import os
import sys
import shutil
from pathlib import Path

def copy_examples():
    """
    site-packages 안에 설치된 examples/ 폴더를
    현재 작업 디렉터리로 복사
    """
    # 1) sys.path에서 examples 폴더 위치 찾기
    examples_src = None
    for p in sys.path:
        cand = Path(p) / "examples"
        if cand.is_dir():
            examples_src = cand
            break

    if examples_src is None:
        print("[bok-da] 설치된 site-packages 에 examples 폴더가 없습니다.")
        sys.exit(1)

    # 2) 현재 워킹 디렉터리에 복사
    target = Path.cwd() / "examples"
    if target.exists():
        print(f"[bok-da] '{target}' 이미 존재합니다. 삭제 후 다시 시도하세요.")
        return

    shutil.copytree(examples_src, target)
    print(f"[bok-da] 예제를 '{target}' 에 복사했습니다.")