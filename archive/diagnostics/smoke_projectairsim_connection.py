"""Minimal Project AirSim connection smoke test."""

from projectairsim import ProjectAirSimClient


def main() -> None:
    client = ProjectAirSimClient()

    try:
        print("[TEST] Connecting to Project AirSim...")
        client.connect()
        print("[PASS] Connected successfully.")
    finally:
        print("[TEST] Disconnecting...")
        client.disconnect()
        print("[PASS] Disconnected cleanly.")


if __name__ == "__main__":
    main()