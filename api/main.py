
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header, Response, Request
import uuid
import os
import time
from worker.tasks import process_audio
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from core.metrics import UPLOAD_COUNT, REQUEST_COUNT, REQUEST_LATENCY, get_json_logger

logger = get_json_logger("fastapi_app")
app = FastAPI(title="RAKSHA Audio Pipeline")

API_KEY = os.getenv("AASIOM_DISPATCH_API_KEY", "raksha_secure_key_999")


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        logger.warning("Unauthorized access attempt",
                       extra={"status_code": 401})
        raise HTTPException(status_code=401, detail="Invalid API Key")


@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    REQUEST_COUNT.inc()
    response = await call_next(request)
    REQUEST_LATENCY.observe(time.time() - start_time)
    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/upload")
async def upload_audio(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    UPLOAD_COUNT.inc()
    job_id = str(uuid.uuid4())
    temp_path = f"/tmp/{job_id}_{file.filename}"

    logger.info("Received audio upload", extra={
                "job_id": job_id, "filename": file.filename})

    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    process_audio.delay(job_id, temp_path)
    logger.info("Job dispatched to Celery", extra={"job_id": job_id})

    return {"job_id": job_id, "status": "pending", "message": "Dispatched to RAKSHA pipeline."}


@app.get("/api/v1/status/{job_id}")
async def get_status(job_id: str, api_key: str = Depends(verify_api_key)):
    logger.info("Status check requested", extra={"job_id": job_id})
    result = process_audio.AsyncResult(job_id)

    if result.state == 'PENDING':
        return {"job_id": job_id, "status": "pending"}
    elif result.state == 'SUCCESS':
        return result.result
    elif result.state == 'FAILURE':
        logger.error("Job failure reported", extra={
                     "job_id": job_id, "error": str(result.info)})
        return {"job_id": job_id, "status": "failed", "error": str(result.info)}
    return {"job_id": job_id, "status": result.state}
