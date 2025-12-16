import pandas as pd
import io
from typing import Union
from pathlib import Path


class DataLoader:
    """
    Utility class for loading data from various file formats.
    Supports CSV and XLSX files.
    """

    @staticmethod
    def load_dataframe(
        file_content: Union[str, bytes, io.BytesIO, Path],
        file_type: str = "csv"
    ) -> pd.DataFrame:
        """
        Load a DataFrame from file content.

        Args:
            file_content: File path (str/Path) or file content (bytes/BytesIO)
            file_type: File type ('csv' or 'xlsx')

        Returns:
            pd.DataFrame loaded from file
        """
        if file_type == "csv":
            if isinstance(file_content, (str, Path)):
                return pd.read_csv(file_content)
            elif isinstance(file_content, bytes):
                return pd.read_csv(io.BytesIO(file_content))
            elif isinstance(file_content, io.BytesIO):
                return pd.read_csv(file_content)
            else:
                raise ValueError(f"Unsupported file_content type: {type(file_content)}")

        elif file_type == "xlsx":
            if isinstance(file_content, (str, Path)):
                return pd.read_excel(file_content)
            elif isinstance(file_content, bytes):
                return pd.read_excel(io.BytesIO(file_content))
            elif isinstance(file_content, io.BytesIO):
                return pd.read_excel(file_content)
            else:
                raise ValueError(f"Unsupported file_content type: {type(file_content)}")

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path], file_type: str = "csv"):
        """
        Save a DataFrame to file.

        Args:
            df: DataFrame to save
            file_path: Path to save file
            file_type: File type ('csv' or 'xlsx')
        """
        if file_type == "csv":
            df.to_csv(file_path, index=False)
        elif file_type == "xlsx":
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def get_file_type_from_name(filename: str) -> str:
        """
        Determine file type from filename extension.

        Args:
            filename: Name of the file

        Returns:
            File type string ('csv', 'xlsx', 'txt', 'docx', 'pdf')
        """
        extension = Path(filename).suffix.lower()

        extension_map = {
            ".csv": "csv",
            ".xlsx": "xlsx",
            ".xls": "xlsx",
            ".txt": "txt",
            ".md": "md",
            ".docx": "docx",
            ".pdf": "pdf",
        }

        return extension_map.get(extension, "unknown")