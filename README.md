# RememberMe Assistant

이미지 분석과 음성 인식을 통합한 일정 관리 도우미 서비스입니다.

## 1. 서버 설치 및 실행 (Ubuntu)

### 1.1 필수 패키지 설치
```bash
# 파이썬 및 필수 시스템 패키지 설치
sudo apt update
sudo apt install python3-pip python3-dev

# 프로젝트 클론 및 이동
git clone <repository_url>
cd RememberMe/Assistant

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 1.2 환경 변수 설정
`.env` 파일 생성:
```bash
cat > .env << EOL
AZURE_API_KEY=your_azure_openai_key
STT_API_KEY=your_azure_speech_key
EOL
```

### 1.3 서버 실행
```bash
# 개발 모드
uvicorn image_back:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
uvicorn image_back:app --host 0.0.0.0 --port 8000 --workers 4
```

## 2. 파이썬 클라이언트 사용 방법

### 2.1 이미지 분석 사용
```python
import requests
import json
from pathlib import Path

def analyze_image(image_path, server_url="http://localhost:8000"):
    """이미지 분석 함수"""
    # 이미지 파일 준비
    files = {
        "file": (Path(image_path).name, open(image_path, "rb"), "image/jpeg")
    }
    
    # API 호출
    response = requests.post(f"{server_url}/process/", files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"에러: {response.status_code}")
        return None

# 사용 예시
image_path = "img/coupon.jpg"
result = analyze_image(image_path)
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### 2.2 검색 기능 사용
```python
def search_data(query, server_url="http://localhost:8000"):
    """데이터 검색 함수"""
    response = requests.get(f"{server_url}/search/", params={"value": query})
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"에러: {response.status_code}")
        return None

# 사용 예시
results = search_data("검색어")
print(json.dumps(results, indent=2, ensure_ascii=False))
```

## 3. API 엔드포인트

### 3.1 이미지 처리
- `POST /process/`
  - 기능: 이미지 업로드 및 분석
  - 입력: 이미지 파일 (multipart/form-data)
  - 출력: 분석 결과 JSON

### 3.2 데이터 검색
- `GET /search/`
  - 기능: 저장된 데이터 검색
  - 파라미터: value (검색어)
  - 출력: 검색 결과 목록

### 3.3 카테고리별 조회
- `GET /category/{category}`
  - 기능: 특정 카테고리의 모든 데이터 조회
  - 카테고리: 쿠폰, 교통, 엔터테인먼트, 약속, 불명, 기타

## 4. 참고사항

- 이미지 지원 형식: JPG, JPEG, PNG
- 음성 인식: 마이크 필수, 한국어만 지원
- 데이터 보관: 7일 후 자동 삭제
- 서버 요구사항: Python 3.8 이상
- 기본 포트: 8000 (변경 가능)
