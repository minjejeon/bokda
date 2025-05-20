import os
import sys
import shutil
from pathlib import Path

def import_example_manual():
    """
    site-packages 안에 설치된 examples/notebooks, examples/data, manual 폴더를
    현재 작업 디렉터리로 복사
    """
    # 1) sys.path에서 site-packages 경로 찾기
    pkg_root = None
    for p in sys.path:
        cand = Path(p)
        # examples/notebooks 또는 examples/data 또는 manual 폴더가 있는지 확인
        if (cand / "examples" / "notebooks").is_dir() \
        or (cand / "examples" / "data").is_dir() \
        or (cand / "manual").is_dir():
            pkg_root = cand
            break
        
    #if pkg_root is None:
    #    print("[bok-da] 설치된 site-packages에 examples/notebooks, examples/data, manual 폴더가 없습니다.")
    #    sys.exit(1)
    
    #####    
    # 2) 개발환경(Editable 모드) 경로 지정: 패키지 루트의 두 단계 상위 폴더
    dev_root = Path(__file__).resolve().parents[2]
    #####
    
    # 복사 목록 정의 (패키지 내부 경로, 작업 디렉터리 경로)
    items = [
        ("examples/notebooks", "examples/notebooks"),
        ("examples/data",      "examples/data"),
        ("manual",             "manual"),
    ]

    for src_rel, tgt_rel in items:
        #src = pkg_root / src_rel
        #tgt = Path.cwd() / tgt_rel
        
        ###############
        # 설치환경 우선, 없으면 개발환경 소스 트리에서
        if pkg_root and (pkg_root / src_rel).exists():
            src = pkg_root / src_rel
        else:
            src = dev_root / src_rel

        tgt = Path.cwd() / tgt_rel
        ###############
        
        if not src.exists():
            # 존재하지 않으면 건너뜀
            continue

        if tgt.exists():
            print(f"[bok-da] '{tgt}' 이미 존재합니다. 삭제 후 다시 시도하세요.")
            continue

        # 디렉터리 복사
        shutil.copytree(src, tgt)
        print(f"[bok-da] '{src_rel}' 폴더를 '{tgt_rel}'에 복사했습니다.")
        
if __name__ == '__main__':
    import_example_manual()