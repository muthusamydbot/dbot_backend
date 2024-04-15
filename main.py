from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from secrets import token_hex
import uvicorn
import os
from fastapi.staticfiles import StaticFiles
import cv_modelling as cvModeling

app = FastAPI(title="Upload file using FastAPI")

def ensure_uploaded_files_directory():
    upload_directory = "./uploaded_files"
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)

ensure_uploaded_files_directory()

# Mount a directory to serve static files
app.mount("/uploaded_files", StaticFiles(directory="uploaded_files"), name="uploaded_files")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_name_without_extension, _ = os.path.splitext(file.filename)
    upload_directory = "./uploaded_files"
    file_path = os.path.join(upload_directory, file_name_without_extension, file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "ab") as f:
        content = await file.read()
        f.write(content)

    # Construct the URL of the uploaded file
    base_url = "http://localhost:8000"  # Replace with your actual domain
    file_url = f"{base_url}/uploaded_files/{file_name_without_extension}/{file.filename}"
    stl_url = r"https://0cc9-27-5-219-67.ngrok-free.app/uploaded_files/Workspace/Workspace.stl"
    data = cvModeling.main(url=file_url,folderName=file_name_without_extension,base_url=base_url,)
    print("jfff",data)
    return {"success": True, "file_url": file_url, "stl_url": data,"message": "File uploaded successfully"}

# @app.get("/download/{filename}")
# async def download_file(filename: str):
#     file_path = f"./uploaded_files/{filename}"
#     if os.path.isfile(file_path):
#         return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')
#     else:
#         raise HTTPException(status_code=404, detail="File not found.")
    

def replace_dot(string, replacement):

    return string.replace('.', replacement)
    
if __name__ == "__main__":
    uvicorn.run("main:app",host="127.0.0.1",reload=True)

# uvicorn.run("main:app",host="127.0.0.1",reload=True)