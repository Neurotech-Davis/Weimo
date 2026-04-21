from machine import Pin
import time

# Motor 1 is mapped to GP12 and GP13
# The TXS0108E logic level converter is translating the 5V to 3.3V
pin_a = Pin(1, Pin.IN)
pin_b = Pin(0, Pin.IN)

tick_count = 0

# The Interrupt function that fires every time the wheel moves
def encoder_callback(pin):
    global tick_count
    # Check the other pin to determine forward or backward
    if pin_b.value() == 1:
        tick_count += 1
    else:
        tick_count -= 1

# Tell the Pico to watch GP12 like a hawk
pin_a.irq(trigger=Pin.IRQ_RISING, handler=encoder_callback)

print("Starting Encoder Test... Spin Motor 1 (Top Right) by hand!")

# Slowly print the results so you can read them
while True:
    print("Motor 1 Ticks:", tick_count)
    time.sleep(0.5)