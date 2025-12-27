#!/usr/bin/env python3
"""
List available audio input/output devices using sounddevice (PortAudio).
"""

from __future__ import annotations

import sounddevice as sd


def main() -> None:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    default_in, default_out = sd.default.device

    print("Default devices:")
    print(f"  input:  {default_in}")
    print(f"  output: {default_out}")
    print("")

    print("Available devices:")
    for idx, dev in enumerate(devices):
        hostapi = hostapis[dev["hostapi"]]["name"]
        name = dev["name"]
        max_in = dev["max_input_channels"]
        max_out = dev["max_output_channels"]
        rate = int(dev.get("default_samplerate", 0))

        flags = []
        if idx == default_in:
            flags.append("default-in")
        if idx == default_out:
            flags.append("default-out")
        flag_str = f" [{' '.join(flags)}]" if flags else ""

        print(
            f"{idx:2d}: {name} ({hostapi}) "
            f"in:{max_in} out:{max_out} rate:{rate}{flag_str}"
        )


if __name__ == "__main__":
    main()
