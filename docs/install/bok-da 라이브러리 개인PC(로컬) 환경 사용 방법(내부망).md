# 한국은행 데이터 분석 라이브러리(bok-da) 사용 매뉴얼(내부망)

내부망 개인 PC 환경(Anaconda 배포판 파이썬을 사용하는 JupyterLab 환경)에서 bok-da 라이브러리를 설치하고 사용하는 방법을 안내하는 매뉴얼입니다.

---

## I. 개발 환경 준비

Anaconda 배포판 파이썬 환경(Python 버전 3.9 이상) 준비

1. `S:드라이브` - `info` - `전직원용` - `데이터분석서비스` - `설치파일` - `Python3.11` 폴더로 이동

2. Python3.11 폴더 내 `# Python 설치 안내.txt`에 따라 파이썬 설치 

## II. JupyterLab 실행

1. Python3.11 폴더 내 `Jupyter Lab(BOK)` 아이콘 더블클릭하여 jupyter lab 실행

2. 웹브라우저가 자동으로 열리며, jupyter lab 인터페이스가 실행됨

## III. bok-da 라이브러리 설치

1. 새로운 작업 디렉토리(`bok_library`) 생성  
   예시: `D:/python_projects/bok_library`

2. `S 드라이브`의 `PC\999.한국은행 데이터 분석 라이브러리(bok-da)`에서 `python-wheels-windows.zip`과 `cython-3.1.1-py3-none-any.whl`파일을 `bok_library` 폴더로 이동 후 `zip`파일 압축해제

3. jupyter lab 터미널 실행 후 작업 디렉토리 경로 지정  

4. jupyter lab 터미널에서 cython과 bok-da 휠파일 설치  
   ```bash
   pip install cython-3.1.1-py3-none-any.whl
   pip install bok_da-0.3.1-cp311-cp311-win_amd64.whl
   ```

## IV. bok-da 라이브러리 사용

1. jupyter lab에서 새 Python 노트북 파일(.ipynb 확장자) 열기

2. 노트북 셀에서 아래와 같이 코드 입력하여 bok-da 라이브러리 불러오기
    ```python
    import bok_da as bd
    ```
    또는 필요한 클래스, 함수만 불러올 수 있음(예시: 시계열분석 패키지(ts)의 LBVAR 모듈에서 Adaptive LBVAR 모형 추정 클래스 불러오기)
    ```python
    from bok_da.ts.lbvar import LBVAR_Adaptive
    ```

## V. 예제코드 및 매뉴얼 활용

1. `S:\PC\999.한국은행 데이터 분석 라이브러리(bok-da)`의 `examples`, `docs` 폴더 참고.

2. `examples/notebooks` 폴더에 분석기능별 활용예제를 수록하였음  
   `examples/data` 폴더에 예제코드에서 사용하는 데이터가 수록되어 있음  
   `docs/reference` 폴더에 모형 사용 및 하이퍼파라미터 옵션에 대한 설명이 있음  
   `docs/dev` 폴더에 코딩 스타일 관련 가이드라인을 수록