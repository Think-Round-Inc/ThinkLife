"""
Response Schema Validator
==========================
Checks that a CortexFlow result dict has the expected shape before it
leaves the Brain. Used by PolicyValidator.validate_result.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, ValidationError, confloat


class CortexResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: StrictBool
    content: StrictStr
    metadata: Dict[str, Any]
    processing_time: Optional[confloat(ge=0)] = None
    confidence: Optional[confloat(ge=0, le=1)] = None


def validate_response_schema(result) -> List[str]:
    """Return a list of human-readable schema errors (empty list = valid)."""
    try:
        CortexResult.model_validate(result, strict=True)
    except ValidationError as e:
        return [f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in e.errors()]
    return []
