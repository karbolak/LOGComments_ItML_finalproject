from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
import numpy as np


class Feature(BaseModel):
    name: str = Field(..., description="The name of the feature.")
    type: Literal["numerical", "categorical"] = Field(..., description="The "
                                                      "type of the feature, "
                                                      "either numerical or "
                                                      "categorical.")
    values: Optional[np.ndarray] = Field(default_factory=lambda: np.array([]),
                                         description="The values for the "
                                                     "feature.")

    @field_validator("values", mode="before")
    def convert_to_array(cls, v):
        """Ensures values are stored as a NumPy array for consistency."""
        return np.array(v) if v is not None else np.array([])

    def __str__(self):
        """String representation for Feature."""
        return f"{self.name} (Type: {self.type})"

    # New model configuration for Pydantic v2
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }
