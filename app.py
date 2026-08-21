from fastapi import FastAPI, File, UploadFile
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from src.constants import APP_HOST, APP_PORT
from src.pipeline.training import TrainingPipeline
from src.pipeline.prediction import PredictionPipeline


app = FastAPI(
    title="Signature Recognition API",
    description="API for training and signature prediction",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "Signature Recognition API is running",
        "docs": "/docs"
    }


# ============================================================
# TRAIN
# ============================================================

@app.get("/train")
async def training():

    try:

        train_pipeline = TrainingPipeline()

        train_pipeline.run_pipeline()

        return Response(
            content="Training Successful !!!",
            media_type="text/plain"
        )

    except Exception as e:

        return Response(
            content=f"Error occurred! {str(e)}",
            media_type="text/plain",
            status_code=500
        )


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def prediction(
    image_file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # Validate uploaded file
        # ----------------------------------------------------

        if not image_file.content_type:
            return JSONResponse(
                content={
                    "error": "File type could not be determined."
                },
                status_code=400
            )

        allowed_types = [
            "image/jpeg",
            "image/png",
            "image/jpg"
        ]

        if image_file.content_type not in allowed_types:
            return JSONResponse(
                content={
                    "error": (
                        "Invalid file type. "
                        "Please upload a JPG, JPEG or PNG image."
                    )
                },
                status_code=400
            )

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        image_bytes = await image_file.read()

        if not image_bytes:
            return JSONResponse(
                content={
                    "error": "Uploaded image is empty."
                },
                status_code=400
            )

        # ----------------------------------------------------
        # Create prediction pipeline
        # ----------------------------------------------------

        prediction_pipeline = PredictionPipeline()

        # ----------------------------------------------------
        # Run prediction
        # ----------------------------------------------------

        final_output = prediction_pipeline.run_pipeline(
            image_bytes
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return JSONResponse(
            content={
                "filename": image_file.filename,
                "prediction": final_output
            }
        )

    except Exception as e:

        return JSONResponse(
            content={
                "error": f"Error Occurred! {str(e)}"
            },
            status_code=500
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT
    )