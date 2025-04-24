import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

# 설정이 이미 완료되었는지 확인하기 위한 플래그
_HANGUL_SETUP_DONE = False

def setup_hangul_font(force=False):
    """
    운영체제에 맞춰 Matplotlib 및 Plotter의 한글 폰트 및 마이너스 부호 설정을 수행합니다.

    Matplotlib 및 Plotter를 사용하는 스크립트나 노트북 시작 부분에서 한 번 호출하는 것을 권장합니다.

    Args:
        force (bool): True로 설정하면 이미 설정이 완료되었더라도 강제로 다시 실행합니다.
                      기본값은 False입니다.
    """
    global _HANGUL_SETUP_DONE
    if _HANGUL_SETUP_DONE and not force:
        print("Hangul font setup already done.") # 디버깅용 메시지
        return

    print("> Initiating Hangul font setup for Graphs...") # 설정 시작 알림

    os_name = platform.system()
    font_set = False

    if os_name == 'Windows':
        # Windows 환경: 주로 'Malgun Gothic' 사용 시도, 없으면 'Gulim' 시도
        font_family = "Malgun Gothic"
        try:
            mpl.rc('font', family=font_family)
            mpl.rcParams['axes.unicode_minus'] = False # 마이너스 부호 설정
            font_set = True
            print(f"> Font set to '{font_family}' and unicode_minus=False.")
        except Exception as e:
            print(f"> Failed to set font to '{font_family}': {e}")
            # 대체 폰트 시도
            font_family = "Gulim"
            try:
                mpl.rc('font', family=font_family)
                mpl.rcParams['axes.unicode_minus'] = False
                font_set = True
                print(f"> Font set to '{font_family}' and unicode_minus=False.")
            except Exception as e_alt:
                print(f"> Failed to set font to '{font_family}': {e_alt}")

    elif os_name == 'Linux':
        # Linux 환경: 'Nanum' 계열 폰트 시도 (사전에 설치 필요할 수 있음)
        # 예: sudo apt-get update && sudo apt-get install fonts-nanum*
        font_family = "NanumGothicCoding" # 코딩용 나눔고딕
        try:
            mpl.rc('font', family=font_family)
            mpl.rcParams['axes.unicode_minus'] = False
            font_set = True
            print(f"> Font set to '{font_family}' and unicode_minus=False.")
        except Exception as e:
            print(f"> Failed to set font to '{font_family}': {e}")
            font_family = "NanumGothic" # 일반 나눔고딕 시도
            try:
                mpl.rc('font', family=font_family)
                mpl.rcParams['axes.unicode_minus'] = False
                font_set = True
                print(f"> Font set to '{font_family}' and unicode_minus=False.")
            except Exception as e_alt:
                 print(f"> Failed to set font to '{font_family}': {e_alt}")

    elif os_name == 'Darwin': # macOS
         font_family = "AppleGothic"
         try:
             mpl.rc('font', family=font_family)
             mpl.rcParams['axes.unicode_minus'] = False
             font_set = True
             print(f"> Font set to '{font_family}' and unicode_minus=False.")
         except Exception as e:
             print(f"> Failed to set font to '{font_family}': {e}")

    else:
         print(f"> Unsupported OS ({os_name}) for automatic Hangul font setup.")

    if not font_set:
         warnings.warn("> Hangul font could not be set automatically. Please install a suitable Hangul font (e.g., Malgun Gothic, NanumGothic) and set 'matplotlib.rcParams[\"font.family\"]' manually.", UserWarning)
         # 마이너스 부호 설정은 폰트와 별개로 시도
         try:
            mpl.rcParams['axes.unicode_minus'] = False
            print("> Set axes.unicode_minus to False (regardless of font setting success).")
         except Exception as e:
            print(f"> Warning: Could not set axes.unicode_minus: {e}")


    _HANGUL_SETUP_DONE = True