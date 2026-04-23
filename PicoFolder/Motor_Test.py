from machine import Pin
import time

# Raw pin reads - no IRQ, no handlers, just direct polling
m4_a = Pin(0, Pin.IN, Pin.PULL_UP)
m4_b = Pin(1, Pin.IN, Pin.PULL_UP)

count = 0
last = m4_a.value()

print("Polling M1 encoder (pin 2). Spin M1 wheel slowly...")
print("Press Ctrl+C to stop.\n")

while True:
    current = m4_a.value()
    if current != last:
        count += 1
        last = current
        print(f"Edge detected! Total transitions: {count}")
    time.sleep_us(500)