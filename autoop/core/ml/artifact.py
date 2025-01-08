from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import base64


class Artifact(BaseModel):
    name: str
    asset_path: str
    version: str
    type: str = Field(..., description="Type of the artifact (e.g., dataset, "
                      "model)")
    data: Optional[bytes] = None
    metadata: Dict[str, str] = Field(default_factory=dict,
                                     description="Metadata for the artifact")
    tags: List[str] = Field(default_factory=list, description="Tags "
                            "associated with the artifact")
    id: Optional[str] = None  # This is set based on `asset_path` and `version`

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.id is None:  # Generate an id if not provided
            self.id = self.generate_id()

    def generate_id(self) -> str:
        """Generates a unique ID based on asset_path and version."""
        encoded_path = base64.urlsafe_b64encode(
                       self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """
        Simulates reading data for the artifact. In practice, this would
        interact with storage.
        """
        if self.data is None:
            raise ValueError("Data not available for this artifact")
        return self.data

    def save(self, data: bytes):
        """
        Simulates saving data for the artifact. In practice, this would
        save to storage.
        """
        self.data = data

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True
