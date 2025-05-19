# 한국은행 데이터 분석 라이브러리(bok-da) 사용 매뉴얼 (인터넷망)

이 문서는 인터넷망 개인 PC 환경(Anaconda 배포판 파이썬을 사용하는 JupyterLab 환경)에서 bok-da 라이브러리를 설치하고 사용하는 방법을 안내하는 매뉴얼입니다.

---

## I. 개발 환경 준비

Anaconda 배포판 파이썬 환경(Python 버전 3.9 이상) 준비

1. Anaconda 배포판 다운로드 사이트에서 운영체제에 맞는 설치 파일 다운로드  
   - https://www.anaconda.com/download  
2. 설치가 완료되면, **Anaconda Prompt** 실행  
3. Anaconda Prompt에서 가상환경 생성  
   ```bash
   conda create -n <환경이름> python=<버전>
   ```
   예시:
   ```bash
   conda create -n bok-da-env python=3.12
   ```
4. Anaconda Prompt에서 가상환경 활성화  
   ```bash
   conda activate <환경이름>
   ```
   예시:
   ```bash
   conda activate bok-da-env
   ```

## II. JupyterLab 실행

1. Anaconda Prompt에서 **jupyter lab** 설치
   ```bash
   pip install jupyterlab
   ```
1. Anaconda Prompt에서 jupyter lab 실행
    ```bash
    jupyter lab
    ```
    웹브라우저가 자동으로 열리며, jupyter lab 인터페이스가 실행됨

## III. bok-da 라이브러리 설치

1. 새로운 작업 디렉토리(`bok_library`) 생성  
   예시: `C:/Users/BOK/Desktop/bok_library`

2. `python-wheels-windows.zip`파일을 `bok_library` 폴더로 이동 후 압축해제

3. jupyter lab 터미널 실행 후 작업 디렉토리 경로 지정  
   - jupyter lab의 `Launcher` 탭에서 `Terminal` 클릭
   - (터미널에서) `cd Desktop/bok_library`

4. jupyter lab 터미널에서 가상환경의 파이썬 버전에 맞는 bok-da 휠파일 설치  
   예시: 파이썬 버전이 3.12인 경우,
   ```bash
   pip install bok_da-0.3.1-cp312-cp312-win_amd64.whl
   ```

## IV. bok-da 라이브러리 사용

1. jupyter lab에서 새 Python 노트북 파일(.ipynb 확장자) 열기

2. 노트북 셀에서 아래와 같이 코드 입력하여 bok-da 라이브러리 불러오기
    ```python
    import bok_da as bd
    ```

## V. 활용 예제코드 및 매뉴얼 불러오기

1. jupyter lab 터미널에서 아래 명령어를 실행하면, 예제코드, 데이터, 매뉴얼 폴더가 생성됨
   ```bash
   bokda-copy-examples_manual
   ```

2. `examples/notebooks` 폴더에서 예제코드를 확인하고 테스트할 수 있음  
   `examples/data` 폴더에 예제코드에서 사용하는 데이터가 수록되어 있음  
   `manual` 폴더에 모형 사용 및 하이퍼파라미터 옵션에 대한 설명이 있음