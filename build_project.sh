cat << 'EOF' > api/main.py
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid
import os

from db.database import engine, Base, get_db
from db.models import JobRecord
from worker.tasks import process_audio

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Enterprise Audio Pipeline API", version="1.0.0")

@app.post("/api/v1/upload")
async def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    job_id = str(uuid.uuid4())
    os.makedirs("/tmp/audio_uploads", exist_ok=True)
    file_path = f"/tmp/audio_uploads/{job_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    new_job = JobRecord(id=job_id, status="pending")
    db.add(new_job)
    db.commit()

    process_audio.delay(job_id, file_path)
    return {"job_id": job_id, "status": "pending", "message": "Audio queued for processing."}

@app.get("/api/v1/status/{job_id}")
def get_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job.id, "status": job.status, "error": job.error_detail}
EOF

echo "✅ Enterprise folder structure and codebase generated successfully!"
