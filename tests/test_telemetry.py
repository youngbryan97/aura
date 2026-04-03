from core.schemas import TelemetryPayload
try:
    p = TelemetryPayload(energy=80.0, curiosity=50.0, frustration=0.0, confidence=100.0)
    print("Success:", p)
except Exception as e:
    print("Error:", e)
