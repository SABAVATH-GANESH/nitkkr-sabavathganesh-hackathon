class RiskEngineError(Exception):
    """Base error."""

class IngestionError(RiskEngineError):
    pass

class ModelLoadError(RiskEngineError):
    pass

class StressTestError(RiskEngineError):
    pass
