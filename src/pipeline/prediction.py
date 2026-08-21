import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from src.constants import DEVICE, CONFIG_PATH, LABEL_NAME
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import read_yaml_file
from src.configurations.gcloud_syncer import GCloudSync


class PredictionPipeline:

    def __init__(self):

        self.gcloud = GCloudSync()

        self.config = read_yaml_file(CONFIG_PATH)

        self.img_size = self.config[
            "data_transformation_config"
        ]["img_size"]

    # ============================================================
    # IMAGE LOADER
    # ============================================================

    def image_loader(self, image_bytes):

        """
        Load image bytes and convert them into a PIL image.

        Returns:
            PIL.Image.Image: Loaded RGB image
        """

        logging.info(
            "Entered the image_loader method "
            "of PredictionPipeline class"
        )

        try:

            logging.info(
                "Loading image bytes and saving image locally"
            )

            input_image = self.config[
                "prediction_pipeline_config"
            ]["input_image"]

            with open(input_image, "wb") as image:

                image.write(image_bytes)

            path = os.path.join(
                os.getcwd(),
                input_image
            )

            image = Image.open(path).convert("RGB")

            logging.info(
                f"Image saved successfully at: {path}"
            )

            logging.info(
                "Exited the image_loader method "
                "of PredictionPipeline class"
            )

            return image

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # GET MODEL FROM GCLOUD
    # ============================================================

    def get_model_from_gcloud(self) -> str:

        """
        Download the best model from GCloud Storage.

        Returns:
            str: Local path of downloaded model
        """

        logging.info(
            "Entered the get_model_from_gcloud method "
            "of PredictionPipeline class"
        )

        try:

            logging.info(
                "Loading the best model from GCloud bucket"
            )

            predict_model_path = os.path.join(
                os.getcwd(),
                "artifacts",
                "PredictModel"
            )

            os.makedirs(
                predict_model_path,
                exist_ok=True
            )

            bucket_name = self.config[
                "prediction_pipeline_config"
            ]["bucket_name"]

            model_name = self.config[
                "prediction_pipeline_config"
            ]["model_name"]

            self.gcloud.sync_file_from_gcloud(
                bucket_name,
                model_name,
                predict_model_path
            )

            best_model_path = os.path.join(
                predict_model_path,
                model_name
            )

            logging.info(
                f"Model downloaded to: {best_model_path}"
            )

            logging.info(
                "Exited the get_model_from_gcloud method "
                "of PredictionPipeline class"
            )

            return best_model_path

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # PREDICTION
    # ============================================================

    def prediction(
        self,
        best_model_path: str,
        image
    ) -> str:

        """
        Load the trained model and perform prediction.

        Returns:
            str: Predicted class name
        """

        logging.info(
            "Entered the prediction method "
            "of PredictionPipeline class"
        )

        try:

            # ----------------------------------------------------
            # LOAD MODEL
            # ----------------------------------------------------

            logging.info(
                "Loading the best model"
            )

            model = torch.load(
                best_model_path,
                map_location=DEVICE,
                weights_only=False
            )

            model.to(DEVICE)

            model.eval()

            logging.info(
                "Best model loaded successfully"
            )

            # ----------------------------------------------------
            # IMAGE PREPROCESSING
            # ----------------------------------------------------

            logging.info(
                "Loading the image and preprocessing it"
            )

            preprocess = transforms.Compose([

                transforms.Resize(
                    (
                        self.img_size,
                        self.img_size
                    )
                ),

                transforms.Grayscale(
                    num_output_channels=3
                ),

                transforms.ToTensor(),

            ])

            image = preprocess(image)

            # ----------------------------------------------------
            # ADD BATCH DIMENSION
            # ----------------------------------------------------

            logging.info(
                "Converting image to PyTorch tensor "
                "and sending it to device"
            )

            image = image.unsqueeze(0)

            image = image.to(
                DEVICE,
                non_blocking=True
            )

            # ----------------------------------------------------
            # MODEL PREDICTION
            # ----------------------------------------------------

            logging.info(
                "Making prediction"
            )

            with torch.no_grad():

                logits = model(image)

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                predicted_index = torch.argmax(
                    probabilities,
                    dim=1
                )

            predicted_label = predicted_index.item()

            logging.info(
                f"Predicted label index: {predicted_label}"
            )

            # ----------------------------------------------------
            # MAP INDEX TO CLASS NAME
            # ----------------------------------------------------

            logging.info(
                "Mapping predicted label "
                "to corresponding class name"
            )

            predicted_class_name = LABEL_NAME[
                predicted_label
            ]

            logging.info(
                f"Predicted class name: "
                f"{predicted_class_name}"
            )

            logging.info(
                "Exited the prediction method "
                "of PredictionPipeline class"
            )

            return predicted_class_name

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # RUN COMPLETE PREDICTION PIPELINE
    # ============================================================

    def run_pipeline(
        self,
        data
    ):

        """
        Run the complete prediction pipeline.

        Flow:

            image bytes
                  ↓
            load image
                  ↓
            download model from GCloud
                  ↓
            load model
                  ↓
            preprocess image
                  ↓
            make prediction
                  ↓
            return predicted class
        """

        logging.info(
            "Entered the run_pipeline method "
            "of PredictionPipeline class"
        )

        try:

            # ----------------------------------------------------
            # STEP 1: LOAD INPUT IMAGE
            # ----------------------------------------------------

            image = self.image_loader(
                data
            )

            # ----------------------------------------------------
            # STEP 2: DOWNLOAD MODEL
            # ----------------------------------------------------

            best_model_path = (
                self.get_model_from_gcloud()
            )

            # ----------------------------------------------------
            # STEP 3: MAKE PREDICTION
            # ----------------------------------------------------

            predicted_class = self.prediction(
                best_model_path,
                image
            )

            logging.info(
                f"Final prediction: {predicted_class}"
            )

            logging.info(
                "Exited the run_pipeline method "
                "of PredictionPipeline class"
            )

            return predicted_class

        except Exception as e:

            raise CustomException(e, sys) from e