# 한국은행 데이터분석 라이브러리 

> 한국은행 데이터분석 라이브러리 리포지토리입니다


## 주요 사항

- Scikit-learn과 유사한 방식으로 모델 정의 후, `.fit()` 메서드를 통해 모델 적합을 수행하는 방식으로 구현되었습니다.
- 일부 모델에서 경우 cython 을 통한 계산 가속을 지원합니다. `pip install .`을 통해 패키지 설치 시, 기기 cython이 설치되어 있을 경우 경우 자동으로 코드를 컴파일합니다.
- `manual` 디렉토리에 모델별 사용 방*법을 담은 매뉴얼을 마크다운 형태로 정리하였고, `demo` 디렉토리에 주피터 노트북 기반의 테스트 코드를 포함하였습니다.


## 설치

설치 과정은 다음과 같습니다.
1. git clone / pull 등으로 리포지토리 소스코드 다운로드
2. (선택) 기기에 cython, numpy 설치 - [Installing Cython](https://cython.readthedocs.io/en/stable/src/quickstart/install.html) 
3. `setup.py` 파일이 있는 디렉토리에서 `pip install -U .` 을 통해 라이브러리 설치
4. demo_ipynb 폴더에서 demo 코드를 실행해 작동 확인

- 파이썬 3.11기반의 환경을 지원합니다.


## 디렉토리 구성

- `bok_da/`: 라이브러리 소스코드 디렉토리
    - `bayes/`: 베이지안 모델 코드 디렉토리
    - `freq/`: 빈도주의 모델 코드 디렉토리
    - `hlw/`: HLW 계열 모델 코드 디렉토리
    - `lbvar/`: LBVAR 계열 모델 코드 디렉토리
    - `ucsv/`: UCSV 계열 모델 코드 디렉토리
    - `__init__.py`: 파이썬 패키지 init 스크립트
    - `container_class.py`: Container Class 정의
    - `utils.py`: 공용 함수들 정의

- `demo_ipynb/`: 모델별 예시 코드 디렉토리 (pip install 로 라이브러리 설치 후 사용할 수 있는 버전)
    - `data/`: 데모 실행 시 사용하는 데이터 디렉토리

- `demo_ipynb_without_install/`: 모델별 예시 코드 디렉토리 (pip install 없이, requriements만 설치 후 사용해볼 수 있는 버전)
    - `data/`: 데모 실행 시 사용하는 데이터 디렉토리

- `manual/`: 라이브러리의 모델에 대한 자세한 설명이 기재된 매뉴얼이 위치
    - `UCSV Multivariate.md`
    - `UCSV Univariate.md`
    - `Probit Regression Frequentist.md`
    - `Probit Regression Bayes.md`
    - `Logistic Regression Frequentist.md`
    - `Logistic Regression Bayes.md`
    - `LBVAR Intro.md`: LBVAR 모델들에 대한 개요
    - `LBVAR Adaptive.md`
    - `LBVAR Asymmetry.md`
    - `LBVAR Noninfo.md`
    - `LBVAR Symmetry.md`
    - `HLW.md`
    - `HLW FC.md`
    - `HLW Covid.md`
    - `Container Class.md`: LBVAR, HLW에서 사용하는 Container Class에 대한 설명

- `README.md`: 프로젝트 README.md 문서
- `requirements.txt`: 파이썬 requirements 파일 (jupyter 포함)
- `setup.py`: 모듈 build를 위한 setup.py
- `Dev Guide.md`: 개발 가이드 문서


## 참고사항

- LBVAR 모델의 경우 매뉴얼 중 `LBVAR Intro.md` 에 모델 관련한 개요와 함께 세부 모델별 기능이 표로 정리되어 있습니다.
- cython과 numpy가 설치된 환경일 경우 pip install 시 자동으로 cython 코드 컴파일을 진행합니다. 
- cython 없이 설치한 환경에 추후 cython 을 추가하신 경우, `pip install . -U`를 통해 다시 설치해야 cython 기반의 가속을 사용하실 수 있습니다.
- LBVAR, HLW 모델에서는 Container Class를 사용하여 개발 및 변수 관리의 편의성을 향상시켰습니다. 여러 파라미터를 저장하고 확인하는 데 있어 기존 딕셔너리 기반의 방식보다 효율적인 사용 및 관리가 가능합니다.

- `import bok_da`시 `UserWarning: Distutils was imported before Setuptools...` 메세지가 발생할 경우 `pip install -U setuptools` 후 다시 `pip install -U bok_da`로 설치하면 warning 메세지를 없앨 수 있습니다.
    - 해당 warning은 bok_da라이브러리가 아닌 pip install 시 사용하는 파이썬의 setuptools 라이브러리 관련한 사항으로, 사용자 환경의 setuptools 버전을 업그레이드하여 해결할 수 있습니다.
