아래는 **Azure 가상머신(VM)**과 관련된 모든 작업을 CLI 명령어로 차례대로 정리한 **전체 플로우**야. 이걸 따라 하면 혼란 없이 원하는 기능을 설정하고 사용할 수 있어.

---

## **1. 로컬에서 가상머신에 접속 (SSH 접속)**

### **명령어**
```bash
ssh <username>@20.83.187.202
```

- `<username>`: Azure VM을 생성할 때 설정한 사용자 이름.
- 성공 시 Azure VM의 터미널로 접속됨.

---

## **2. 로컬에서 가상머신으로 파일 전달**

로컬에서 가상머신으로 파일을 전송하려면 **`scp` 명령어**를 사용.

### **명령어**
```bash
scp -r /path/to/local/files <username>@20.83.187.202:/path/to/destination/
```

- 예:
  ```bash
  scp -r ~/project_files <username>@20.83.187.202:/home/<username>/fastapi-app
  ```

- 설명:
  - `/path/to/local/files`: 로컬에서 보낼 파일이나 디렉토리 경로.
  - `/path/to/destination/`: 가상머신에서 파일을 저장할 위치.

---

## **3. 가상머신에서 FastAPI 백엔드 실행**

가상머신에 접속한 뒤, FastAPI를 실행하기 위해 필요한 명령어를 순서대로 실행.

### **명령어**

#### **3.1 Python 및 라이브러리 설치**
1. **Python 설치 확인 및 업데이트**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip -y
   ```

2. **가상환경(Venv) 생성 및 활성화**:
   ```bash
   python3 -m venv pyvenv
   source pyvenv/bin/activate
   ```

3. **필요한 패키지 설치**:
   - `requirements.txt` 파일이 있다면:
     ```bash
     pip install -r requirements.txt
     ```
   - 아니면 개별 설치:
     ```bash
     pip install fastapi uvicorn gunicorn
     ```

#### **3.2 FastAPI 서버 실행**
1. FastAPI 애플리케이션이 있는 디렉토리로 이동:
   ```bash
   cd /home/<username>/fastapi-app
   ```

2. `uvicorn`으로 서버 실행:
   ```bash
   uvicorn image_back:app --host 0.0.0.0 --port 8000
   ```

- `--host 0.0.0.0`: 외부에서 접근 가능하게 설정.
- `--port 8000`: FastAPI 서버가 8000번 포트에서 실행.

---

## **4. 가상머신 엔드포인트 외부 접근 허용**

### **Azure 네트워크 보안 그룹(NSG)에서 포트 8000 열기**

1. Azure 포털 > VM > **Networking(네트워킹)** 클릭.
2. **Add inbound port rule(인바운드 포트 규칙 추가)** 클릭.
3. 아래 값을 입력 후 저장:
   - **Source**: `Any`
   - **Source Port Ranges**: `*`
   - **Destination**: `Any`
   - **Destination Port Ranges**: `8000`
   - **Protocol**: `TCP`
   - **Action**: `Allow`
   - **Priority**: 낮은 숫자 (예: `100`).
   - **Name**: `AllowPort8000`

---

### **5. VM 내부 방화벽(UFW)에서 포트 8000 허용**

1. UFW(방화벽) 상태 확인:
   ```bash
   sudo ufw status
   ```

2. 포트 8000 허용:
   ```bash
   sudo ufw allow 8000
   sudo ufw enable
   ```

3. 방화벽 상태 재확인:
   ```bash
   sudo ufw status
   ```

---

## **6. 외부에서 FastAPI 서버 접근**

FastAPI 서버가 실행 중이고 포트 8000이 열려 있으면, 다음 명령어로 접근 가능.

### **명령어**
#### **이미지 업로드 요청**
```bash
curl -X POST "http://20.83.187.202:8000/process/" -F "file=@path_to_local_image.jpg"
```

- 예:
  ```bash
  curl -X POST "http://20.83.187.202:8000/process/" -F "file=@china_ticket.jpg"
  ```

#### **음성 인식 요청**
```bash
curl -X POST "http://20.83.187.202:8000/stt/" -F "file=@path_to_audio.wav"
```

---

## **7. 퍼블릭 포트를 80번으로 매핑 (옵션)**

포트 8000 대신 기본 HTTP 포트(80번)로 요청을 받을 수 있도록 Nginx를 설정.

### **명령어**
1. **Nginx 설치**:
   ```bash
   sudo apt install nginx -y
   ```

2. **Nginx 설정 파일 추가**:
   ```bash
   sudo nano /etc/nginx/sites-available/fastapi
   ```

3. 아래 내용 추가:
   ```nginx
   server {
       listen 80;
       server_name 20.83.187.202;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```

4. **설정 활성화**:
   ```bash
   sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled
   sudo nginx -t
   sudo systemctl restart nginx
   ```

5. 이제 기본 포트 80으로 접근 가능:
   ```bash
   curl -X POST "http://20.83.187.202/process/" -F "file=@china_ticket.jpg"
   ```

---

### **전체 요약**
1. **VM 접속**: `ssh <username>@20.83.187.202`
2. **파일 전송**: `scp -r /path/to/local/files <username>@20.83.187.202:/path/to/destination/`
3. **FastAPI 실행**: `uvicorn image_back:app --host 0.0.0.0 --port 8000`
4. **포트 열기**:
   - Azure NSG: 포트 8000 허용.
   - UFW: 포트 8000 허용.
5. **외부 접근**:
   - `curl -X POST "http://20.83.187.202:8000/process/" -F "file=@file.jpg"`

이 플로우를 따르면 문제 없이 FastAPI 서버를 실행하고 외부에서 접근 가능할 거야! 😊 추가로 궁금한 점이 있으면 언제든 물어봐.