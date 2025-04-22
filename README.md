# 한국은행 데이터 분석 라이브러리(bok-da)

> 한국은행 데이터분석 라이브러리(bok-da) 리포지토리

- `bok-da` 라이브러리는 파이썬 기반 데이터 분석 라이브러리로 당행 데이터 분석 업무의 효율성 제고를 위해 개발
  - 기본적인 데이터 입수 및 시각화, 계량경제 분석 도구를 우선 개발
  - 향후 각 부서에서 생산한 분석 코드를 표준화$\cdot$최적화하여 수록함으로써 당행 지적자산으로 축적
  - 부서간 협업$\cdot$학술연구용역등을 통해 개발된 고급 모형(LBVAR, 중립금리 추정모형)을 수록하여 분석 도구 다각화 예정
  
- 라이브러리를 활용하여 데이터 분석의 단계별 작업(데이터 입수$\cdot$전치리, 분석, 시각화, 공유 등)을 단일 인터페이스(`BIDAS 고성능 데이터 분석환경`)에서 수행할 수 있음
   - 파이썬 환경에 익숙하지 않은 사용자를 위해 사용 매뉴얼 제공
   - 직원 누구나 사용 목적에 맞게 소스코드를 수정하거나 새로운 기능을 추가하여 개발에 참여할 수 있음


## 주요 사항

- Scikit-learn과 유사한 방식으로 모형 객체 생성 후, `.fit()` 메서드를 통해 모델 적합을 수행하는 방식으로 구현
- 일부 모델(상태공간모형, UCSV)에서 경우 cython을 통한 빠른 코드 실행을 지원
- `manual` 디렉토리에 모델별 사용 방*법을 담은 매뉴얼을 마크다운 형태로 정리
- `examples` 디렉토리에 주피터 노트북 기반의 예제 코드와 데이터를 포함
- Python 3.9, 3.12 기반의 환경을 지원


## 라이브러리 구조(디렉토리 구성)
- `src/bok_da/`: 라이브러리 소스코드 디렉토리
    - `bayes/`: 베이지안 모형 코드 디렉토리
    - `cs/`: 횡단면 데이터 기반 계량모형 디렉토리
    - `data/`: 데이터 입수 및 전처리 관련 코드 디렉토리
    - `linear/`: 선형 모형 코드 디렉토리
    - `panel/`: 패널 모형 코드 디렉토리
    - `patch/`: statsmodels 패키지의 패치를 위한 코드 디렉토리
    - `project/`: 행내 개발 모형 및 분석도구 코드 디렉토리
    - `stata/`: 파이썬 환경에서 stata 활용을 위한 코드 디렉토리
    - `ts/`: 시계열 모형 코드 디렉토리
    - `utils/`: 공용 함수 및 기타 데이터 분석 코드 디렉토리
    - `validation/`: 모형 검증 코드 디렉토리
    - `viz/`: 시각화 도구 코드 디렉토리
    - `__init__.py`: 파이썬 패키지 init 스크립트

- `examples/`: 예제 코드와 데이터를 모아놓은 디렉토리
    - `data/`: 예제 코드 실행에 필요한 데이터 디렉토리
    - `notebooks/`: 모형 및 분석도구별 예제 코드(.ipynb) 디렉토리

- `manual/`: 라이브러리가 지원하는 모형에 대한 자세한 설명이 기재된 문서 디렉토리(지속 업데이트 예정)

- `README.md`: bok-da 리파지토리의 readme 문서

- `Dev Guide.md`: 라이브러리 도구 개발 가이드 문서

- `pyproject.toml`, `setup.py`, `MANIFEXT.in`: bok-da 패키지 빌드를 위한 파일

# BIDAS 환경에서 라이브러리 설치

## 1. BIDAS Modelhub -> 고성능 데이터 분석환경 -> Jupyterlab(Python) 실행

![image](img/bidas_modelhub.PNG)

## 2. Jupyterlab에서 File -> New Launcher -> Terminal 실행

![image](img/terminal.PNG)

## 3. GitLab에서 라이브러리 리파지토리(repository) 불러오기
Terminal에서 아래 코드 순서대로 실행
```bash
cd 행번 (예시: cd 2310490)
git config --global http.sslVerify false
git clone --depth 1 https://bidas-gitlab.boknet.intra/digitaltech/bok_library.git
```
![image](img/git_clone.PNG)

## 4. 설치
Terminal에서 아래 코드 순서대로 실행
```bash
cd bok_library/bok_da/dist
pip install bok-da-0.3.0.tar.gz
```
![image](img/bok_installation.PNG)

# 라이브러리 사용
## 1. bok-da 라이브러리 불러오기
Jupyterlab에서 bok_library 폴더에 새 노트북 파일(.ipynb)을 생성하고, 아래 코드 실행
```python
import bok_da as bd
```
## 2. 매뉴얼 활용
```bash
bok_library 폴더에서 매뉴얼 노트북 파일 참조
```
![image](img/bok_manual.PNG)

## 3. gitlab에서 복제해서 만든 자신의 작업폴더를 새로운 gitlab project로 생성
```bash
git remote remove origin
git remote add origin https://행번:내부망윈도우패스워드@bidas-gitlab.boknet.intra/행번/프로젝트이름.git
git add .
git commit -m "initial commit"
git push origin main
```

# 개인PC(로컬) 환경에서 라이브러리 사용
내부망 `S드라이브` PC/999. 한국은행 데이터 분석 라이브러리(bok-da) 폴더에 `bok-da 라이브러리 개인PC(로컬) 환경 사용 방법(내부망).txt` 또는 `동파일명(외부망).txt` 참조

# 라이브러리 개발 기여
- 디지털혁신실 데이터사이언스팀
- 고려대학교 경제학과 한치록, 강규호 교수 연구진
- 서강대학교 경제학과 허준영, 김재호 교수 연구진
- 연세대학교 응용통계학부 진익훈 교수 연구진

# 라이브러리 관련 문의
디지털혁신실 데이터사이언스팀 이창훈 과장(4638)