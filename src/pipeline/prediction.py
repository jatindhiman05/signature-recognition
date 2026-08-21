import os
import sys
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms

from src.constants import (
    DEVICE,
    CONFIG_PATH,
    LABEL_NAME
)
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import read_yaml_file
from src.configurations.s3_syncer import S3Sync


class PredictionPipeline:

    def __init__(self):

        logging.info(
            "Initializing PredictionPipeline"
        )

        try:

            # ====================================================
            # AWS S3
            # ====================================================

            self.s3 = S3Sync()

            # ====================================================
            # CONFIG
            # ====================================================

            self.config = read_yaml_file(
                CONFIG_PATH
            )

            # ====================================================
            # DATA TRANSFORMATION CONFIG
            # ====================================================

            transformation_config = self.config[
                "data_transformation_config"
            ]

            self.img_size = transformation_config[
                "img_size"
            ]

            self.mean = transformation_config[
                "mean"
            ]

            self.std = transformation_config[
                "std"
            ]

            # ====================================================
            # PREDICTION CONFIG
            # ====================================================

            prediction_config = self.config[
                "prediction_pipeline_config"
            ]

            self.bucket_name = prediction_config[
                "bucket_name"
            ]

            self.model_name = prediction_config[
                "model_name"
            ]

            self.threshold = prediction_config[
                "threshold"
            ]

            # ====================================================
            # LOCAL MODEL CACHE
            # ====================================================

            self.predict_model_dir = os.path.join(
                os.getcwd(),
                "artifacts",
                "PredictModel"
            )

            os.makedirs(
                self.predict_model_dir,
                exist_ok=True
            )

            self.local_model_path = os.path.join(
                self.predict_model_dir,
                self.model_name
            )

            logging.info(
                "PredictionPipeline initialized successfully"
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # IMAGE LOADER
    # ============================================================

    def image_loader(
        self,
        image_bytes
    ):

        try:

            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            return image

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # GET MODEL PATH
    # ============================================================

    def get_model_path(self) -> str:

        """
        Return the local model path.

        If the model already exists locally,
        use the cached copy.

        Otherwise download model.pt from AWS S3.
        """

        try:

            # ====================================================
            # LOCAL CACHE
            # ====================================================

            if os.path.isfile(
                self.local_model_path
            ):

                logging.info(
                    f"Using cached model: "
                    f"{self.local_model_path}"
                )

                return self.local_model_path

            # ====================================================
            # DOWNLOAD FROM S3
            # ====================================================

            logging.info(
                f"Downloading model from "
                f"s3://{self.bucket_name}/"
                f"{self.model_name}"
            )

            self.s3.sync_file_from_s3(
                bucket_name=self.bucket_name,
                filename=self.model_name,
                destination=self.local_model_path
            )

            # ====================================================
            # VERIFY DOWNLOAD
            # ====================================================

            if not os.path.isfile(
                self.local_model_path
            ):

                raise FileNotFoundError(
                    "Model download completed but "
                    "model file was not found at: "
                    f"{self.local_model_path}"
                )

            logging.info(
                f"Model downloaded successfully to: "
                f"{self.local_model_path}"
            )

            return self.local_model_path

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # PREPROCESS IMAGE
    # ============================================================

    def preprocess_image(
        self,
        image
    ) -> torch.Tensor:

        try:

            preprocess = transforms.Compose([

                transforms.Resize(
                    (
                        self.img_size,
                        self.img_size
                    )
                ),

                transforms.Grayscale(
                    num_output_channels=1
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=self.mean,
                    std=self.std
                )

            ])

            image_tensor = preprocess(
                image
            )

            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # PREDICT USING ALREADY LOADED MODEL
    # ============================================================

    def prediction(
        self,
        model,
        image
    ) -> dict:

        try:

            # ====================================================
            # PREPROCESS
            # ====================================================

            image_tensor = self.preprocess_image(
                image
            )

            image_tensor = image_tensor.to(
                DEVICE,
                non_blocking=True
            )

            # ====================================================
            # INFERENCE
            # ====================================================

            model.eval()

            with torch.no_grad():

                logits = model(
                    image_tensor
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                confidence, predicted_index = torch.max(
                    probabilities,
                    dim=1
                )

            confidence = confidence.item()

            predicted_index = predicted_index.item()

            # ====================================================
            # THRESHOLD
            # ====================================================

            if confidence < self.threshold:

                predicted_class = "Unknown"

            else:

                predicted_class = LABEL_NAME[
                    predicted_index
                ]

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "predicted_index": predicted_index
            }

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # RUN PIPELINE
    # ============================================================

    def run_pipeline(
        self,
        image_bytes,
        model
    ):

        try:

            # ====================================================
            # LOAD IMAGE
            # ====================================================

            image = self.image_loader(
                image_bytes
            )

            # ====================================================
            # PREDICT
            # ====================================================

            result = self.prediction(
                model=model,
                image=image
            )

            logging.info(
                f"Prediction result: {result}"
            )

            return result

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e