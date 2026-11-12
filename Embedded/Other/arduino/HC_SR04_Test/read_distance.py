import serial
import serial.tools.list_ports
import time
import sys

def main():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    port = "COM13" if "COM13" in ports else (ports[0] if ports else None)
    
    if not port:
        print("No serial COM ports found!")
        sys.exit(1)

    print(f"Connecting to Arduino on {port} (9600 baud)...")
    try:
        ser = serial.Serial(port, 9600, timeout=2)
    except Exception as e:
        print(f"Error opening {port}: {e}")
        sys.exit(1)

    time.sleep(2)
    print("\n" + "="*50)
    print("      LIVE HC-SR04 DISTANCE LOGS (Press Ctrl+C to stop)      ")
    print("="*50 + "\n")

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"[HC-SR04] {line}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()

