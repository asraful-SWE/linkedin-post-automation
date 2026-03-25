"""
Approval Routes - Secure endpoints for approve/reject workflow
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from services.approval_service import ApprovalService


logger = logging.getLogger(__name__)
router = APIRouter(tags=["approval"])


def _get_approval_service(request: Request) -> ApprovalService:
    db_manager = getattr(request.app.state, "db_manager", None)
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return ApprovalService(db_manager)


def _save_uploaded_image(file: UploadFile) -> Optional[str]:
    if not file:
        return None
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    suffix = Path(file.filename or "upload").suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    save_path = uploads_dir / filename

    content = file.file.read()
    if not content:
        return None

    save_path.write_bytes(content)

    base_url = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
    return f"{base_url}/uploads/{filename}"


@router.get("/approve-post/{post_id}")
async def approve_post(post_id: int, token: str, request: Request, image_url: Optional[str] = None):
    service = _get_approval_service(request)
    result = service.approve_post(post_id=post_id, token=token, image_url=image_url)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Approval failed"))
    return JSONResponse(result)


@router.post("/approve-post/{post_id}")
async def approve_post_with_form(
    post_id: int,
    request: Request,
    token: str = Form(...),
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
):
    service = _get_approval_service(request)

    resolved_image_url = image_url
    if image_file is not None:
        uploaded_url = _save_uploaded_image(image_file)
        if uploaded_url:
            resolved_image_url = uploaded_url

    result = service.approve_post(post_id=post_id, token=token, image_url=resolved_image_url)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Approval failed"))

    return HTMLResponse(
        "<html><body style='font-family:Arial;padding:24px;'><h2>Post approved and published.</h2>"
        "<p>You can close this window now.</p></body></html>"
    )


@router.get("/reject-post/{post_id}")
async def reject_post(post_id: int, token: str, request: Request):
    service = _get_approval_service(request)
    result = service.reject_post(post_id=post_id, token=token)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Reject failed"))
    return JSONResponse(result)


@router.get("/approval-form/{post_id}", response_class=HTMLResponse)
async def approval_form(post_id: int, token: str):
    html = f"""
<!doctype html>
<html>
  <body style=\"font-family: Arial, Helvetica, sans-serif; background:#f5f5f5; padding:24px;\">
    <div style=\"max-width:620px; margin:0 auto; background:#fff; border:1px solid #ddd; border-radius:10px; padding:20px;\">
      <h2>Approve Post #{post_id}</h2>
      <p>Add image URL or upload image, then approve.</p>
      <form method=\"post\" action=\"/approve-post/{post_id}\" enctype=\"multipart/form-data\">
        <input type=\"hidden\" name=\"token\" value=\"{token}\" />

        <label>Image URL (optional)</label><br/>
        <input type=\"url\" name=\"image_url\" placeholder=\"https://example.com/image.jpg\" style=\"width:100%;padding:10px;margin:6px 0 12px;\"/>

        <label>Or upload image (optional)</label><br/>
        <input type=\"file\" name=\"image_file\" accept=\"image/*\" style=\"margin:6px 0 16px;\"/>

        <div>
          <button type=\"submit\" style=\"padding:10px 16px;background:#166534;color:#fff;border:none;border-radius:8px;\">Approve & Publish</button>
          <a href=\"/reject-post/{post_id}?token={token}\" style=\"margin-left:10px;padding:10px 16px;background:#b91c1c;color:#fff;text-decoration:none;border-radius:8px;\">Reject</a>
        </div>
      </form>
    </div>
  </body>
</html>
"""
    return HTMLResponse(html)
