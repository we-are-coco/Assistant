from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from pathlib import Path
from module import AImodule  # AI 모델 함수
from stt_module import STTModule  # STT 모듈 추가
#from azure.storage.blob import BlobServiceClient

# Blob Service Client 초기화
#blob_service_client = BlobServiceClient.from_connection_string("your_connection_string")
#container_name = "uploaded-images"

# 업로드된 파일 저장
#blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.filename)
#blob_client.upload_blob(await file.read())

ai_module = AImodule()  # AI 모듈 인스턴스 생성
stt_module = STTModule()  # STT 모듈 인스턴스 생성
app = FastAPI()

UPLOAD_DIR = Path("./img")  # 이미지 저장 디렉토리
UPLOAD_DIR.mkdir(exist_ok=True)  # 디렉토리 생성 (없으면)

# 업로드 및 처리 통합 엔드포인트
@app.post("/process/")
async def upload_and_process_image(file: UploadFile = File(...)):
    # 파일 저장
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # AI 모델로 이미지 처리
    try:
        result = ai_module.analyze_image(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "AI processing failed", "error": str(e)})

    # 결과 반환
    return {"message": "Image processed successfully", "result": result}

# 음성 인식 엔드포인트 추가
@app.post("/stt/")
async def speech_to_text():
    try:
        text = stt_module.recognize_speech()
        if text:
            return {"message": "Speech recognition successful", "text": text}
        else:
            return JSONResponse(
                status_code=400, 
                content={"message": "Speech recognition failed", "error": "No speech detected"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"message": "Speech recognition failed", "error": str(e)}
        )