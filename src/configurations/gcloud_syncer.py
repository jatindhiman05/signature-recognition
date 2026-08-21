import os
import sys
import subprocess

from src.exception import CustomException


class GCloudSync:
    # ============================================================
    # GSUTIL PATH
    # ============================================================

    # ============================================================
    # LOCAL SETUP (Windows) - Commented for future reference
    # ============================================================
    # WINDOWS_GSUTIL_PATH = (
    #     r"C:\Users\Jatin Dhiman\AppData\Local\Google\Cloud SDK"
    #     r"\google-cloud-sdk\bin\gsutil.cmd"
    # )
    #
    # GSUTIL_COMMAND = (
    #     WINDOWS_GSUTIL_PATH
    #     if os.name == "nt"
    #     else "gsutil"
    # )
    # ============================================================

    # ============================================================
    # DOCKER SETUP - For google/cloud-sdk:latest
    # ============================================================
    # In the Docker image, gsutil is in the PATH
    # For Linux containers (default), use "gsutil"
    # For Windows containers, use "gsutil.cmd"
    # ============================================================

    GSUTIL_COMMAND = (
        "gsutil.cmd" if os.name == "nt" else "gsutil"
    )

    # ============================================================
    # CHECK GSUTIL
    # ============================================================

    @classmethod
    def check_gsutil(cls):

        try:

            print(
                f"Using gsutil command: {cls.GSUTIL_COMMAND}"
            )

            result = subprocess.run(
                [cls.GSUTIL_COMMAND, "version"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise Exception(
                    "gsutil was found but could not be executed.\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            print(
                f"gsutil is working:\n{result.stdout}"
            )

        except FileNotFoundError as e:

            raise Exception(
                "gsutil could not be found.\n"
                f"Expected path: {cls.GSUTIL_COMMAND}\n"
                "Please verify Google Cloud SDK is installed."
            ) from e

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # CHECK WHETHER FILE EXISTS IN GCLOUD
    # ============================================================

    def is_file_exist_in_gcloud(
            self,
            gcp_bucket_url,
            filename
    ) -> bool:

        try:

            command = [
                self.GSUTIL_COMMAND,
                "ls",
                f"gs://{gcp_bucket_url}/{filename}"
            ]

            print(
                "Checking GCloud file:",
                f"gs://{gcp_bucket_url}/{filename}"
            )

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )

            # ====================================================
            # FILE EXISTS
            # ====================================================

            if result.returncode == 0:
                print(
                    f"File exists in GCloud: "
                    f"gs://{gcp_bucket_url}/{filename}"
                )

                return True

            # ====================================================
            # FILE DOES NOT EXIST
            # ====================================================

            print(
                f"File does not exist in GCloud: "
                f"gs://{gcp_bucket_url}/{filename}"
            )

            return False

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # DOWNLOAD FILE FROM GCLOUD
    # ============================================================

    def sync_file_from_gcloud(
            self,
            gcp_bucket_url,
            filename,
            destination
    ):

        try:

            command = [
                self.GSUTIL_COMMAND,
                "cp",
                f"gs://{gcp_bucket_url}/{filename}",
                destination
            ]

            print(
                "Executing command:",
                " ".join(command)
            )

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )

            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(result.stderr)

            if result.returncode != 0:
                raise Exception(
                    f"Failed to download {filename} "
                    f"from GCloud Storage.\n"
                    f"Return code: {result.returncode}\n"
                    f"Error: {result.stderr}"
                )

        except Exception as e:

            raise CustomException(e, sys) from e

    # ============================================================
    # UPLOAD FILE TO GCLOUD
    # ============================================================

    def sync_file_to_gcloud(
            self,
            gcp_bucket_url,
            filepath
    ):

        try:

            command = [
                self.GSUTIL_COMMAND,
                "cp",
                filepath,
                f"gs://{gcp_bucket_url}/"
            ]

            print(
                "Executing command:",
                " ".join(command)
            )

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )

            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(result.stderr)

            if result.returncode != 0:
                raise Exception(
                    f"Failed to upload {filepath} "
                    f"to GCloud Storage.\n"
                    f"Return code: {result.returncode}\n"
                    f"Error: {result.stderr}"
                )

        except Exception as e:

            raise CustomException(e, sys) from e