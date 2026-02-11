# Speech Lab OMR Check

OMR 답안지를 웹에서 업로드하면 자동으로 인식하고 채점하여 CSV 결과를 제공하는 원스톱 시스템입니다.

> **원본 프로젝트:** [OMRChecker](https://github.com/Udayraj123/OMRChecker) by Udayraj Deshmukh를 fork하여, 웹 기반 인터페이스와 이미지 처리 파이프라인을 추가 개발하였습니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **웹 기반 UI** | 브라우저에서 드래그앤드롭으로 이미지 업로드, 결과 확인, CSV 다운로드 |
| **용지 자동 크롭** | 3개 코너 마커를 감지하여 OMR 용지 영역만 자동 정렬 및 크롭 |
| **컬러 드롭아웃** | 빨간색 인쇄 템플릿이 마킹으로 오인식되지 않도록 max(B,G,R) 기법으로 컬러 잉크 제거 |
| **가로/세로 자동 판별** | 세로 이미지는 양방향 회전 시도 후 최적 방향을 자동 선택 |
| **답안 키 CSV 채점** | 답안 키 CSV 업로드 시 자동 채점 (정답/오답/미기입 배점 설정 가능) |
| **선행 0 무시** | `08`과 `8`을 동일한 정답으로 인정 |
| **가변 문제 수** | 답안 키 CSV에 있는 문제만 채점 (Q1~Q38이면 38문제만) |
| **디버그 모드** | 결과 테이블 행 클릭 시 OMR 버블 인식 결과 이미지 확인 |
| **빈 답안 키 템플릿** | 44/38/22문항 빈 CSV 템플릿 다운로드 제공 |

---

## 설치

### 요구사항

- Python 3.8+
- pip

### 설치 방법

```bash
git clone https://github.com/<your-username>/speech_lab_OMR_check.git
cd speech_lab_OMR_check
pip install -r requirements.txt
```

---

## 실행

```bash
python3 web_app.py
```

브라우저에서 `http://localhost:5000` 으로 접속합니다.

---

## 사용 방법

### Step 1: 답안 키 설정

1. **빈 템플릿 다운로드**: 44/38/22문항 중 선택하여 빈 CSV 다운로드
2. **정답 입력**: 다운로드한 CSV에 정답 기입 (예: `Q1,16`)
3. **업로드**: 완성된 CSV를 드래그하거나 파일 선택으로 업로드
4. **배점 설정**: 정답/오답/미기입 각각의 점수 지정

답안 키 CSV 형식 (헤더 없음):
```csv
Q1,16
Q2,17
Q3,8
...
Q44,36
```

### Step 2: 스캔 이미지 업로드

1. 스캐너에서 출력된 OMR 답안지 JPG/PNG 이미지를 드래그하거나 파일 선택
2. 여러 장 동시 업로드 가능
3. 옵션 설정:
   - **용지 자동 크롭**: 스캐너 출력물에서 OMR 용지만 잘라냄 (기본 ON)
   - **디버그 모드**: 인식 결과 이미지를 확인할 수 있음 (기본 ON)

### Step 3: 결과 확인

- 통계: 처리 완료, 다중 마킹, 오류, 평균/최고/최저 점수
- 테이블: 학번(Num) + 각 문제 응답 + 정답(초록)/오답(빨강) 하이라이트
- 디버그: 행 클릭 시 OMR 버블 인식 이미지 팝업
- **CSV 다운로드** 버튼으로 전체 결과 다운로드

---

## 프로젝트 구조

```
speech_lab_OMR_check/
├── web_app.py              # Flask 웹 앱 (메인)
├── template/
│   └── template.json       # OMR 레이아웃 템플릿 (QUIZ-3 기준)
├── src/                    # OMRChecker 코어 모듈
│   ├── core.py             # 버블 감지 및 OMR 응답 읽기
│   ├── template.py         # 템플릿 파싱
│   ├── evaluation.py       # 채점 로직
│   ├── constants.py        # 상수 정의
│   ├── logger.py           # 로깅
│   ├── defaults/           # 기본 설정값
│   ├── processors/         # 이미지 전처리기 (Levels, GaussianBlur 등)
│   ├── schemas/            # JSON 스키마 검증
│   └── utils/              # 유틸리티 (이미지, 파싱, 파일)
├── requirements.txt        # Python 패키지 의존성
├── LICENSE                 # MIT License
└── README.md               # 이 파일
```

---

## OMR 처리 파이프라인

```
스캔 이미지 (JPG/PNG)
    │
    ▼
[컬러 드롭아웃] max(B,G,R) → 컬러 잉크 제거, 검정 마킹만 보존
    │
    ▼
[마커 기반 크롭] 3개 코너 마커 감지 → Affine 변환 → 3507x2480 정렬
    │
    ▼
[전처리] Levels (대비 강화) → GaussianBlur (노이즈 제거)
    │
    ▼
[버블 감지] 각 버블 영역 밝기 계산 → 임계값 비교 → 마킹 판정
    │
    ▼
[응답 결합] roll_1~9 → 학번, Q1_1+Q1_2 → Q1 등
    │
    ▼
[채점] 답안 키 대조 → 정답/오답/미기입 판정 → 점수 산출
    │
    ▼
결과 CSV + 웹 테이블
```

---

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

이 프로젝트는 [OMRChecker](https://github.com/Udayraj123/OMRChecker) (MIT License, Copyright (c) 2024-present Udayraj Deshmukh)를 기반으로 합니다. 웹 인터페이스 및 이미지 처리 개선은 june-oh가 개발하였습니다.
