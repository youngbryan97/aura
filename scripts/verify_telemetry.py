
import asyncio
import json
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.event_bus import get_event_bus
from core.schemas import TelemetryPayload

async def test_telemetry_emission():
    print("Starting Telemetry Verification...")
    bus = get_event_bus()
    bus.set_loop(asyncio.get_running_loop())
    
    q = await bus.subscribe("telemetry")
    print("Subscribed to 'telemetry' topic.")

    try:
        # Mocking a pulse to verify schema logic
        mock_payload = {
            "energy": 85.0,
            "curiosity": 60.0,
            "frustration": 10.0,
            "confidence": 90.0,
            "gwt_winner": "test_source",
            "coherence": 1.0,
            "vitality": 0.95,
            "surprise": 0.05,
            "narrative": "Aura is performing a self-check."
        }
        print("Publishing mock telemetry...")
        bus.publish_threadsafe("telemetry", mock_payload)
        
        print("Waiting for pulse...")
        msg_wrapper = await asyncio.wait_for(q.get(), timeout=2.0)
        data = msg_wrapper["data"]
        print(f"✅ Received Telemetry: {json.dumps(data, indent=2)}")
        
        # Verify schema field Presence
        required_fields = ["gwt_winner", "coherence", "vitality", "surprise", "narrative"]
        all_ok = True
        for field in required_fields:
            if field in data:
                print(f"✅ Field '{field}' is present.")
            else:
                print(f"❌ Field '{field}' is MISSING.")
                all_ok = False
        
        if all_ok:
            print("\n✨ SCHEMA VERIFICATION PASSED")
        else:
            print("\n❌ SCHEMA VERIFICATION FAILED")
            raise SystemExit(1)
                
    except asyncio.TimeoutError:
        print("❌ Timeout: No telemetry pulse received.")
        raise SystemExit(1)
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    asyncio.run(test_telemetry_emission())
