#!/usr/bin/env python3
"""
Camera Trigger Controller
Controls Arduino-based camera sync trigger via serial.
"""

import logging
import serial
import serial.tools.list_ports
import time
from typing import Optional

log = logging.getLogger("trigger_controller")


def find_arduino_port() -> Optional[str]:
    """Auto-detect an Arduino serial port."""
    for p in serial.tools.list_ports.comports():
        if p.manufacturer and "arduino" in p.manufacturer.lower():
            return p.device
    # Fallback: match /dev/cu.usbmodem*
    for p in serial.tools.list_ports.comports():
        if "usbmodem" in p.device:
            return p.device
    return None


class TriggerController:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 2.0):
        """
        Initialize the trigger controller.

        Args:
            port: Serial port. If None, auto-detects the Arduino.
            baudrate: Serial baudrate (must match Arduino)
            timeout: Read timeout in seconds
        """
        if port is None:
            port = find_arduino_port()
            if port is None:
                raise RuntimeError("No Arduino found. Connect one or specify the port manually.")
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None

    def connect(self) -> bool:
        """Connect to the Arduino and wait for ready signal."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            # Wait for Arduino reset after serial connection
            time.sleep(2)

            # Clear any buffered data
            self.ser.reset_input_buffer()

            # Check if responsive
            if self.ping():
                log.info("Connected to trigger controller on %s", self.port)
                return True
            else:
                log.warning("Arduino on %s not responding to ping", self.port)
                return False

        except serial.SerialException as e:
            log.error("Arduino connection failed on %s: %s", self.port, e)
            return False

    def disconnect(self):
        """Disconnect from the Arduino."""
        if self.ser and self.ser.is_open:
            try:
                self.stop()
            except Exception as e:
                log.warning("Failed to stop trigger during disconnect: %s", e)
            self.ser.close()
            log.info("Disconnected from Arduino on %s", self.port)

    def _send_command(self, command: str) -> str:
        """Send a command and return the response."""
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Not connected")

        self.ser.reset_input_buffer()
        self.ser.write(f"{command}\n".encode())
        response = self.ser.readline().decode().strip()
        return response

    def ping(self) -> bool:
        """Check if the Arduino is responsive."""
        try:
            response = self._send_command("PING")
            return response == "PONG"
        except Exception as e:
            log.debug("Ping failed: %s", e)
            return False

    def start(self, frequency: float) -> bool:
        """
        Start the sync trigger at the specified frequency.

        Args:
            frequency: Trigger frequency in Hz (1-1000)

        Returns:
            True if successful
        """
        response = self._send_command(f"START:{frequency}")

        if response.startswith("OK:STARTED"):
            parts = response.split(":")
            actual_freq = parts[2] if len(parts) > 2 else "?"
            log.info("Trigger started at %s Hz", actual_freq)
            return True
        else:
            log.error("Trigger start failed: %s", response)
            return False

    def stop(self) -> bool:
        """
        Stop the sync trigger.

        Returns:
            True if successful
        """
        response = self._send_command("STOP")

        if response == "OK:STOPPED":
            log.info("Trigger stopped")
            return True
        else:
            log.error("Trigger stop failed: %s", response)
            return False

    def status(self) -> dict:
        """
        Get the current status.

        Returns:
            Dict with 'running' (bool) and 'frequency' (float)
        """
        response = self._send_command("STATUS")

        if response.startswith("STATUS:"):
            parts = response.split(":")
            try:
                return {
                    "running": parts[1] == "RUNNING",
                    "frequency": float(parts[2]) if len(parts) > 2 else 0.0,
                }
            except (IndexError, ValueError) as e:
                log.warning("Failed to parse status response '%s': %s", response, e)
        return {"running": False, "frequency": 0.0}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Camera Trigger Controller")
    parser.add_argument("command", choices=["start", "stop", "status"], help="Command")
    parser.add_argument("--port", default=None, help="Serial port (auto-detected if omitted)")
    parser.add_argument("--freq", type=float, default=30.0, help="Frequency in Hz (for start)")

    args = parser.parse_args()

    with TriggerController(args.port) as controller:
        if args.command == "start":
            controller.start(args.freq)
        elif args.command == "stop":
            controller.stop()
        elif args.command == "status":
            status = controller.status()
            print(f"Running: {status['running']}, Frequency: {status['frequency']} Hz")
