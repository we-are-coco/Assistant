from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime, timedelta
import uvicorn
import os
import time
import threading
from sqlalchemy import or_, create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import Optional
#from azure.storage.blob import BlobServiceClient

# Blob Service Client 초기화
#blob_service_client = BlobServiceClient.from_connection_string("your_connection_string")
#container_name = "uploaded-images"

# 업로드된 파일 저장
#blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.filename)
#blob_client.upload_blob(await file.read())

# --- 모듈 임포트 (가정) ---
from module import AImodule  # AI 모델 함수
from stt_module import STTModule  # STT 모듈

# 데이터베이스 설정
Base = declarative_base()

class Coupon(Base):
    __tablename__ = "coupons"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), nullable=False)
    brand = Column(String(255))
    type = Column(String(255))
    item = Column(String(255))
    valid_until = Column(String(50))
    code = Column(String(255))
    description = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

class Transportation(Base):
    __tablename__ = "transportations"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), nullable=False)
    type = Column(String(255))
    from_location = Column(String(255))
    to_location = Column(String(255))
    departure_date = Column(String(50))
    departure_time = Column(String(50))
    description = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

class Entertainment(Base):
    __tablename__ = "entertainments"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), nullable=False)
    type = Column(String(255))
    title = Column(String(255))
    date = Column(String(50))
    time = Column(String(50))
    location = Column(String(255))
    description = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), nullable=False)
    type = Column(String(255))
    date = Column(String(50))
    time = Column(String(50))
    location = Column(String(255))
    details = Column(String(1000))
    description = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

class Unsure(Base):
    __tablename__ = "unsures"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), nullable=False)
    type = Column(String(255))
    date = Column(String(50))
    time = Column(String(50))
    description = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

class Others(Base):
    __tablename__ = "others"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), nullable=False)
    description = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./results.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

# --- FastAPI 앱 초기화 ---
app = FastAPI()
ai_module = AImodule()
stt_module = STTModule()
UPLOAD_DIR = Path("./img")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- 백그라운드 정리 작업 ---
RETENTION_DAYS = 7  # 보관 기간
CLEANUP_INTERVAL = 3600  # 1시간마다 실행

def cleanup_old_results():
    while True:
        time.sleep(CLEANUP_INTERVAL)
        db = SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
            for table in [Coupon, Transportation, Entertainment, Appointment, Unsure, Others]:
                old_data = db.query(table).filter(table.created_at < cutoff).all()
                for data in old_data:
                    db.delete(data)
            db.commit()
        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            db.close()

@app.on_event("startup")
def start_cleanup():
    thread = threading.Thread(target=cleanup_old_results, daemon=True)
    thread.start()

# --- 데이터 저장 및 검색 로직 ---
def save_result(category: str, data: dict):
    db = SessionLocal()
    try:
        # 필수 필드 추가
        data["category"] = category
        if "description" not in data:
            data["description"] = "No description provided."

        if category == "쿠폰":
            db_item = Coupon(**data)
        elif category == "교통":
            db_item = Transportation(**data)
        elif category == "엔터테인먼트":
            db_item = Entertainment(**data)
        elif category == "약속":
            db_item = Appointment(**data)
        elif category == "불명":
            db_item = Unsure(**data)
        elif category == "기타":
            db_item = Others(**data)
        else:
            raise ValueError("잘못된 카테고리입니다.")

        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        return db_item
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def search_results(category: str, key: str = None, value: str = None):
    db = SessionLocal()
    try:
        if category == "쿠폰":
            query = db.query(Coupon)
        elif category == "교통":
            query = db.query(Transportation)
        elif category == "엔터테인먼트":
            query = db.query(Entertainment)
        elif category == "약속":
            query = db.query(Appointment)
        elif category == "불명":
            query = db.query(Unsure)
        elif category == "기타":
            query = db.query(Others)
        else:
            raise ValueError("잘못된 카테고리입니다.")

        if key and value:
            query = query.filter(getattr(query.column_descriptions[0]["entity"], key).contains(value))

        return query.all()
    except Exception as e:
        raise e
    finally:
        db.close()


# --- 엔드포인트 ---
@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    # 파일 저장
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # AI 처리
    try:
        result = ai_module.analyze_image(file_path)  # AI 모듈로 결과 생성
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # DB 저장 (오류 발생 시 서버에서만 출력)
    try:
        db_item = save_result(result["category"], result)
        print(f"저장된 ID: {db_item.id}, 파일 이름: {file.filename}")
    except Exception as e:
        print(f"DB 저장 실패: {e}")  # 서버에서만 출력

    # 사용자에게는 AI 결과만 반환
    return result

# 전체 데이터베이스 검색
@app.get("/search/")
async def search(value: str):
    try:
        db = SessionLocal()
        results = []
        for table in [Coupon, Transportation, Entertainment, Appointment, Unsure, Others]:
            query = db.query(table)
            # --- 변경된 부분: 특정 필드만 검색 + 부분 일치(contains) ---
            search_columns = {
                Coupon: ["brand", "item", "description"],
                Transportation: ["from_location", "to_location", "description"],
                Entertainment: ["title", "location", "description"],
                Appointment: ["location", "details", "description"],
                Unsure: ["description"],
                Others: ["description"]
            }
            
            # 해당 테이블에 대한 검색 컬럼 지정
            columns = search_columns.get(table, [])
            
            # 각 컬럼에 대해 부분 일치 검색 (OR 조건)
            filters = []
            for col in columns:
                column = getattr(table, col)
                filters.append(column.contains(value))
            
            if filters:
                query = query.filter(or_(*filters))  # SQLAlchemy의 or_ 사용
                results.extend(query.all())
            
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

# 카테고리별 모든 데이터 반환
@app.get("/category/{category}")
async def get_category_data(category: str):
    try:
        db = SessionLocal()
        if category == "쿠폰":
            results = db.query(Coupon).all()
        elif category == "교통":
            results = db.query(Transportation).all()
        elif category == "엔터테인먼트":
            results = db.query(Entertainment).all()
        elif category == "약속":
            results = db.query(Appointment).all()
        elif category == "불명":
            results = db.query(Unsure).all()
        elif category == "기타":
            results = db.query(Others).all()
        else:
            raise ValueError("잘못된 카테고리입니다.")

        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)