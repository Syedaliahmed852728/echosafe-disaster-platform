from typing import Dict, Any


class EarthquakeMagnitudeInput:
    """Input schema for the earthquake magnitude estimator.

    Estimates the magnitude of a *detected* event from its location and depth.
    This is NOT a predictor of when or where earthquakes will occur.
    """

    REQUIRED_FIELDS = ["latitude", "longitude", "depth_km"]

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.validate()

    def validate(self):
        missing = [f for f in self.REQUIRED_FIELDS if f not in self.data]
        if missing:
            raise ValueError(f"Missing earthquake magnitude input fields: {missing}")


class MagnitudeEstimateOutput:
    """Standardised output for the magnitude estimator."""

    def __init__(
        self,
        region: str,
        estimated_magnitude: float,
        severity_label: str,
        model_name: str,
        message: str,
    ):
        self.region = region
        self.estimated_magnitude = estimated_magnitude
        self.severity_label = severity_label
        self.model_name = model_name
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disaster_type": "Earthquake Magnitude Estimate",
            "region": self.region,
            "estimated_magnitude": round(self.estimated_magnitude, 2),
            "severity_label": self.severity_label,
            "model_name": self.model_name,
            "message": self.message,
        }


class HailstormRiskInput:
    """Input schema for the hailstorm risk predictor.

    Estimates the probability that a given station-day will record observed
    hail (METAR GR/GS). Atmospheric features can be passed explicitly; missing
    fields are pulled from Open-Meteo for the supplied lat/lon when available.
    """

    REQUIRED_FIELDS = ["latitude", "longitude"]

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.validate()

    def validate(self):
        missing = [f for f in self.REQUIRED_FIELDS if f not in self.data]
        if missing:
            raise ValueError(f"Missing hailstorm input fields: {missing}")


class HailstormRiskOutput:
    """Standardised output for the hailstorm risk predictor."""

    def __init__(
        self,
        region: str,
        hail_probability: float,
        will_hail: bool,
        severity_label: str,
        model_name: str,
        message: str,
        features_used: Dict[str, Any] | None = None,
    ):
        self.region = region
        self.hail_probability = hail_probability
        self.will_hail = will_hail
        self.severity_label = severity_label
        self.model_name = model_name
        self.message = message
        self.features_used = features_used or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disaster_type": "Hailstorm Risk",
            "region": self.region,
            "hail_probability": round(self.hail_probability, 4),
            "will_hail": bool(self.will_hail),
            "severity_label": self.severity_label,
            "model_name": self.model_name,
            "features_used": self.features_used,
            "message": self.message,
        }


class HeatwaveRiskInput:
    """Input schema for the heatwave risk predictor.

    Estimates the probability that a given region-day will satisfy the PMD
    heatwave rule (Tmax >= day-of-year normal + 5 deg C, sustained over a
    5-day run). Atmospheric features can be passed explicitly; missing fields
    are pulled from Open-Meteo for the supplied lat/lon when available.
    """

    REQUIRED_FIELDS = ["latitude", "longitude"]

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.validate()

    def validate(self):
        missing = [f for f in self.REQUIRED_FIELDS if f not in self.data]
        if missing:
            raise ValueError(f"Missing heatwave input fields: {missing}")


class HeatwaveRiskOutput:
    """Standardised output for the heatwave risk predictor."""

    def __init__(
        self,
        region: str,
        heatwave_probability: float,
        will_be_heatwave: bool,
        severity_label: str,
        model_name: str,
        message: str,
        forecast: list | None = None,
        features_used: Dict[str, Any] | None = None,
    ):
        self.region = region
        self.heatwave_probability = heatwave_probability
        self.will_be_heatwave = will_be_heatwave
        self.severity_label = severity_label
        self.model_name = model_name
        self.message = message
        self.forecast = forecast or []
        self.features_used = features_used or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disaster_type": "Heatwave Risk",
            "region": self.region,
            "heatwave_probability": round(self.heatwave_probability, 4),
            "will_be_heatwave": bool(self.will_be_heatwave),
            "severity_label": self.severity_label,
            "model_name": self.model_name,
            "forecast": self.forecast,
            "features_used": self.features_used,
            "message": self.message,
        }


class PredictionOutput:
    """Standardized prediction output."""

    def __init__(
        self,
        disaster_type: str,
        region: str,
        risk_level: str,
        confidence: float,
        risk_score: float,
        message: str,
    ):
        self.disaster_type = disaster_type
        self.region = region
        self.risk_level = risk_level
        self.confidence = confidence
        self.risk_score = risk_score
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disaster_type": self.disaster_type,
            "region": self.region,
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 4),
            "risk_score": round(self.risk_score, 2),
            "message": self.message,
        }
