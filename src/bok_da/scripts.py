import os
import sys
import shutil
from pathlib import Path

def import_example_manual():
    """
    site-packages 안에 설치된 examples/, manual/ 폴더와 Dev Guide.md 파일을
    현재 작업 디렉터리로 복사
    """
    # 1) sys.path에서 설치된 site-packages 경로 찾기
    pkg_root = None
    for p in sys.path:
        cand = Path(p)
        if (cand / "examples").is_dir() or (cand / "manual").is_dir() or (cand / "Dev Guide.md").is_file():
            pkg_root = cand
            break

    if pkg_root is None:
        print("[bok-da] 설치된 site-packages에 examples/manual/Dev Guide.md가 없습니다.")
        sys.exit(1)

    # 복사 목록 정의
    items = [
        ("examples", "examples"),
        ("manual", "manual"),
        ("Dev Guide.md", "Dev Guide.md"),
    ]

    for src_name, tgt_name in items:
        src = pkg_root / src_name
        tgt = Path.cwd() / tgt_name

        if not src.exists():
            # 없으면 건너뜀
            continue

        if tgt.exists():
            print(f"[bok-da] '{tgt}' 이미 존재합니다. 삭제 후 다시 시도하세요.")
            continue

        # 디렉터리 vs 파일 분기
        if src.is_dir():
            shutil.copytree(src, tgt)
            print(f"[bok-da] '{src_name}' 폴더를 '{tgt}'에 복사했습니다.")
        else:
            shutil.copy2(src, tgt)
            print(f"[bok-da] '{src_name}' 파일을 '{tgt}'에 복사했습니다.")