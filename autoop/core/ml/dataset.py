from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    def __init__(self, *args, metadata=None, tags=None, **kwargs):
        # Set default values for missing fields
        metadata = metadata or {}
        tags = tags or []
        super().__init__(type="dataset", metadata=metadata, tags=tags,
                         *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1.0.0"):
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
            metadata={},   # Default empty metadata
            tags=[],       # Default empty tags list
        )

    def read(self) -> pd.DataFrame:
        """ Read data from a given path """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """ Save data to a given path """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
