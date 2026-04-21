import serial
import time
import sys
import threading

# --- MAC PORT SETUP ---
# Run `ls /dev/cu.usbmodem*` in your Mac terminal to find your exact port.
SERIAL_PORT = '/dev/cu.usbmodem1101'
BAUD_RATE = 115200

def listen_to_pico(pico_serial):
    """Background task: Continuously reads from Pico and prints to the console."""
    while True:
        try:
            if pico_serial.in_waiting > 0:
                # Read the line, decode it, and ignore weird serial garbage characters
                response = pico_serial.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    print(f"Pico: {response}")
        except Exception:
            # If the serial connection closes, this thread will quietly exit
            break
        time.sleep(0.01)

def main():
    try:
        print(f"Connecting to Pico on {SERIAL_PORT}...")
        pico = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Give the Pico a moment to reset upon connection
        pico.reset_input_buffer()
        
        # Start the background listening thread
        listener_thread = threading.Thread(target=listen_to_pico, args=(pico,), daemon=True)
        listener_thread.start()
        
        print("\n--- MAC COMMAND CENTER READY ---")
        print("Type commands to test the Pico architecture.")
        print("Examples: 'TEST:1', 'TEST:2', 'DRIVE:1500', 'STOP'")
        print("Type 'QUIT' to close this program.\n")
        
        # Interactive Command Loop
        while True:
            command = input()
            
            if command.upper() == 'QUIT':
                print("Exiting Command Center...")
                break
                
            # Send whatever you typed directly to the Pico
            pico.write((command + '\n').encode('utf-8'))
            
            # If you hit STOP, add a tiny delay just to make sure it sends cleanly
            if command.upper() == 'STOP':
                time.sleep(0.1)
                
    except serial.SerialException as e:
        print(f"\n[!] Serial Error: {e}")
        print("Did you update the SERIAL_PORT variable with your specific cu.usbmodem port?")
        print("Tip: Unplug the Pico, plug it back in, and run 'ls /dev/cu.usbmodem*")

if __name__ == '__main__':
    main()