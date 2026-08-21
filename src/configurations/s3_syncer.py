import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.exception import CustomException


class S3Sync:
    """
    Handles file synchronization with AWS S3.

    Storage backend:
        AWS S3

    Authentication:
        boto3 automatically uses AWS credentials.
        On EC2, credentials can come from the attached IAM role.
    """

    # ============================================================
    # S3 CONFIGURATION
    # ============================================================

    S3_REGION = os.getenv(
        "AWS_REGION",
        "ap-south-1"
    )

    _s3_client = None

    # ============================================================
    # GET S3 CLIENT
    # ============================================================

    @classmethod
    def get_s3_client(cls):

        try:

            if cls._s3_client is None:

                cls._s3_client = boto3.client(
                    "s3",
                    region_name=cls.S3_REGION
                )

            return cls._s3_client

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # CHECK S3 CONNECTION
    # ============================================================

    @classmethod
    def check_s3(cls):

        try:

            s3 = cls.get_s3_client()

            response = s3.list_buckets()

            print(
                "AWS S3 connection successful."
            )

            buckets = response.get(
                "Buckets",
                []
            )

            print(
                "Available S3 buckets:",
                [
                    bucket["Name"]
                    for bucket in buckets
                ]
            )

        except (BotoCoreError, ClientError) as e:

            raise CustomException(e, sys) from e

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # CHECK WHETHER FILE EXISTS IN S3
    # ============================================================

    def is_file_exist_in_s3(
        self,
        bucket_name,
        filename
    ) -> bool:

        try:

            s3 = self.get_s3_client()

            print(
                "Checking S3 file:",
                f"s3://{bucket_name}/{filename}"
            )

            s3.head_object(
                Bucket=bucket_name,
                Key=filename
            )

            print(
                "File exists in S3:",
                f"s3://{bucket_name}/{filename}"
            )

            return True

        except ClientError as e:

            error_code = (
                e.response
                .get("Error", {})
                .get("Code")
            )

            if error_code in (
                "404",
                "NoSuchKey",
                "NotFound"
            ):

                print(
                    "File does not exist in S3:",
                    f"s3://{bucket_name}/{filename}"
                )

                return False

            raise CustomException(e, sys) from e

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # DOWNLOAD FILE FROM S3
    # ============================================================

    def sync_file_from_s3(
        self,
        bucket_name,
        filename,
        destination
    ):

        try:

            s3 = self.get_s3_client()

            print(
                "Downloading from S3:",
                f"s3://{bucket_name}/{filename}"
            )

            print(
                "Destination:",
                destination
            )

            destination_dir = os.path.dirname(
                os.path.abspath(destination)
            )

            if destination_dir:

                os.makedirs(
                    destination_dir,
                    exist_ok=True
                )

            s3.download_file(
                bucket_name,
                filename,
                destination
            )

            print(
                f"Successfully downloaded "
                f"{filename} from S3."
            )

        except (BotoCoreError, ClientError) as e:

            raise CustomException(e, sys) from e

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # UPLOAD FILE TO S3
    # ============================================================

    def sync_file_to_s3(
        self,
        bucket_name,
        filepath
    ):

        try:

            s3 = self.get_s3_client()

            filename = os.path.basename(
                filepath
            )

            print(
                "Uploading to S3:",
                filepath
            )

            print(
                "Destination:",
                f"s3://{bucket_name}/{filename}"
            )

            s3.upload_file(
                filepath,
                bucket_name,
                filename
            )

            print(
                f"Successfully uploaded "
                f"{filepath} to S3."
            )

        except (BotoCoreError, ClientError) as e:

            raise CustomException(e, sys) from e

        except Exception as e:

            raise CustomException(e, sys) from e