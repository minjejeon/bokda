# setup.py

from setuptools import setup, Extension
import os

# Cython과 NumPy는 pyproject.toml의 [build-system].requires에 명시
try:
    from Cython.Build import cythonize
    import numpy
    print("> Cython and NumPy found. Defining Cython modules.")

    # --- 모든 Cython 확장 모듈 정의 ---
    # 중요: 경로와 이름은 실제 src 레이아웃 구조에 맞게 정확히 지정해야 함
    # (예: src/bok_da/ts/ssm/cython_SSM.pyx -> bok_da.ts.ssm.cython_SSm)
    extensions = [
        # State-space model의 Cython 모듈
        Extension(
            "bok_da.ts.ssm.cython_SSM",                 # 파이썬 임포트 경로
            ["src/bok_da/ts/ssm/cython_SSM.pyx"],       # .pyx 소스 파일 경로 (src 기준)
            include_dirs=[numpy.get_include()],    # NumPy 헤더 포함
        ),
        # UCSV model의 Cython 모듈들
        Extension(
            "bok_da.ts.ucsv.ucsv_functions_cython",
            ["src/bok_da/ts/ucsv/ucsv_functions_cython.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "bok_da.ts.ucsv.ucsv_functions_cython_multivar",
            ["src/bok_da/ts/ucsv/ucsv_functions_cython_multivar.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        # 다른 Cython 모듈이 생기면 밑에 계속 추가
    ]
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}) # Python 3 사용 명시

except ImportError:
    print(">" * 60)
    print("> WARNING: Cython or NumPy not found during setup. Cannot build Cython extensions.")
    print("> The resulting package will likely be broken or missing functionality.")
    print(">" * 60)
    # 필수 모듈이라면 여기서 빌드를 중단시키는 것이 더 안전할 수 있습니다.
    # raise RuntimeError("Cython and NumPy are required to build this package.")
    ext_modules = []

# setup() 함수는 ext_modules 전달 역할만 수행
setup(
    ext_modules=ext_modules,
)