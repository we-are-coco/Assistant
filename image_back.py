from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from pathlib import Path
from module import AImodule  # AI 모델 함수
#from azure.storage.blob import BlobServiceClient

# Blob Service Client 초기화
#blob_service_client = BlobServiceClient.from_connection_string("your_connection_string")
#container_name = "uploaded-images"

# 업로드된 파일 저장
#blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.filename)
#blob_client.upload_blob(await file.read())

ai_module = AImodule()  # AImodule 인스턴스 생성
app = FastAPI()

UPLOAD_DIR = Path("./img")  # 이미지 저장 디렉토리

# 이미지 업로드 엔드포인트
@app.post("/img/")
async def upload_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": "Image uploaded successfully", "file_path": str(file_path)}

# 이미지 처리 엔드포인트
@app.get("/process/")
async def process_uploaded_image(file_name: str):
    file_path = UPLOAD_DIR / file_name
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"message": "File not found"})
    
    # AI 모델로 이미지 처리
    result = ai_module.analyze_image(image_path)
    # 결과 반환 (JSON 또는 이미지 형태로)
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
