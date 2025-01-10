"""Module contains functions to validate Pydantic models."""

import pydantic_core
from pydantic import BaseModel


def validate_pydantic_model(model: BaseModel, response: str) -> None:
    """Validate a Pydantic model."""
    try:
        model.model_validate_json(response)
    except pydantic_core._pydantic_core.ValidationError as e:
        print(f"Error: {e}")
        return
