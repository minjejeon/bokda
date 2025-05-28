# 한국은행 데이터 분석 라이브러리(bok-da)
- `bok-da` 라이브러리는 파이썬 기반 데이터 분석 라이브러리로 당행 데이터 분석 업무의 효율성 제고를 위해 개발
  - 기본적인 데이터 입수 및 시각화, 계량경제 분석 도구를 우선 개발
  - 향후 각 부서에서 생산한 분석 코드를 표준화$\cdot$최적화하여 수록함으로써 당행 지적자산으로 축적
  - 부서간 협업$\cdot$학술연구용역등을 통해 개발된 고급 모형(LBVAR, 중립금리 추정모형)을 수록하여 분석 도구 다각화 예정
  
- 라이브러리를 활용하여 데이터 분석의 단계별 작업(데이터 입수$\cdot$전치리, 분석, 시각화, 공유 등)을 단일 인터페이스(`BIDAS 고성능 데이터 분석환경`)에서 수행할 수 있음
   - 파이썬 환경에 익숙하지 않은 사용자를 위해 사용 매뉴얼 제공
   - 직원 누구나 사용 목적에 맞게 소스코드를 수정하거나 새로운 기능을 추가하여 개발에 참여할 수 있음

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
git clone https://bidas-gitlab.boknet.intra/digitaltech/bok_library.git
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
내부망 `S드라이브` 또는 인터넷망 `X드라이브`의 PC/999. 한국은행 데이터 분석 라이브러리(bok-da) 폴더에 `bok-da 라이브러리 개인PC(로컬) 환경 사용 방법.txt` 참조

# 라이브러리 개발 기여
- 디지털신기술팀
- 고려대학교 경제학과 한치록, 강규호 교수 연구진

# 문의
디지털혁신실 디지털신기술팀 이창훈 과장(4638)