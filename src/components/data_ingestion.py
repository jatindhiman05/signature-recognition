import os
import sys
import shutil
from zipfile import ZipFile

from src.logger import logging
from src.exception import CustomException
from src.configurations.s3_syncer import S3Sync
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:

    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig
    ):
        """
        :param data_ingestion_config:
            Configuration for data ingestion
        """

        self.data_ingestion_config = data_ingestion_config

        # AWS S3 storage handler
        self.s3 = S3Sync()

    # ============================================================
    # GET DATA FROM S3
    # ============================================================

    def get_data_from_s3(self) -> None:
        """
        Download dataset ZIP file from AWS S3.

        The downloaded ZIP file is stored inside the
        data ingestion artifacts directory.
        """

        logging.info(
            "Entered the get_data_from_s3 method "
            "of DataIngestion class"
        )

        try:

            # ----------------------------------------------------
            # CREATE ARTIFACT DIRECTORY
            # ----------------------------------------------------

            os.makedirs(
                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
                exist_ok=True
            )

            # ----------------------------------------------------
            # DOWNLOAD DATASET FROM S3
            # ----------------------------------------------------

            self.s3.sync_file_from_s3(
                bucket_name=(
                    self.data_ingestion_config.BUCKET_NAME
                ),
                filename=(
                    self.data_ingestion_config.ZIP_FILE_NAME
                ),
                destination=(
                    self.data_ingestion_config.ZIP_FILE_PATH
                )
            )

            logging.info(
                "Dataset successfully downloaded from AWS S3"
            )

            logging.info(
                "Exited the get_data_from_s3 method "
                "of DataIngestion class"
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # UNZIP AND CLEAN
    # ============================================================

    def unzip_and_clean(self) -> None:
        """
        Extract the downloaded ZIP file and remove
        the unnecessary nested 'dataset' directory.
        """

        logging.info(
            "Entered the unzip_and_clean method "
            "of DataIngestion class"
        )

        try:

            artifact_dir = (
                self.data_ingestion_config
                .DATA_INGESTION_ARTIFACTS_DIR
            )

            # ----------------------------------------------------
            # EXTRACT ZIP FILE
            # ----------------------------------------------------

            with ZipFile(
                self.data_ingestion_config.ZIP_FILE_PATH,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    artifact_dir
                )

            # ----------------------------------------------------
            # HANDLE NESTED DATASET DIRECTORY
            # ----------------------------------------------------

            dataset_dir = os.path.join(
                artifact_dir,
                "dataset"
            )

            if os.path.exists(dataset_dir):

                for item in os.listdir(dataset_dir):

                    source = os.path.join(
                        dataset_dir,
                        item
                    )

                    destination = os.path.join(
                        artifact_dir,
                        item
                    )

                    shutil.move(
                        source,
                        destination
                    )

                # Remove empty dataset directory
                os.rmdir(dataset_dir)

            logging.info(
                "Dataset extracted and cleaned successfully"
            )

            logging.info(
                "Exited the unzip_and_clean method "
                "of DataIngestion class"
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e

    # ============================================================
    # INITIATE DATA INGESTION
    # ============================================================

    def initiate_data_ingestion(
        self
    ) -> DataIngestionArtifacts:
        """
        Execute the complete data ingestion process:

        1. Download dataset.zip from AWS S3
        2. Extract the dataset
        3. Remove unnecessary nested directory
        4. Delete the ZIP file
        5. Return DataIngestionArtifacts
        """

        logging.info(
            "Entered the initiate_data_ingestion method "
            "of DataIngestion class"
        )

        try:

            # ----------------------------------------------------
            # STEP 1: DOWNLOAD DATASET FROM S3
            # ----------------------------------------------------

            self.get_data_from_s3()

            logging.info(
                "Fetched the zipped dataset from AWS S3 bucket"
            )

            # ----------------------------------------------------
            # STEP 2: EXTRACT DATASET
            # ----------------------------------------------------

            self.unzip_and_clean()

            logging.info(
                "Dataset successfully extracted"
            )

            # ----------------------------------------------------
            # STEP 3: DELETE ZIP FILE
            # ----------------------------------------------------

            logging.info(
                "Deleting downloaded dataset ZIP file"
            )

            if os.path.exists(
                self.data_ingestion_config.ZIP_FILE_PATH
            ):
                os.remove(
                    self.data_ingestion_config.ZIP_FILE_PATH
                )

            # ----------------------------------------------------
            # STEP 4: CREATE DATA INGESTION ARTIFACT
            # ----------------------------------------------------

            data_ingestion_artifacts = (
                DataIngestionArtifacts(
                    dataset_path=(
                        self.data_ingestion_config
                        .DATA_INGESTION_ARTIFACTS_DIR
                    )
                )
            )

            logging.info(
                f"Data ingestion artifact: "
                f"{data_ingestion_artifacts}"
            )

            logging.info(
                "Exited the initiate_data_ingestion method "
                "of DataIngestion class"
            )

            return data_ingestion_artifacts

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e