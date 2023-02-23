from dataclasses import dataclass
from typing import Optional, List


@dataclass
class RunConfig:
    target: str
    file_path: str
    features: Optional[List[str]] = None
    missing_value_method: Optional[str] = None
    outlier_detector_method: Optional[str] = None
    feature_transformer_method: Optional[str] = None
    feature_transformer_features: Optional[List[str]] = None
    feature_selector_k: Optional[int] = None
    feature_scaler_method: Optional[str] = None
    model_types: Optional[List[str]] = None
