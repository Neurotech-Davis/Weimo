import time
import cv2
import numpy as np
import requests
import os
from dotenv import load_dotenv

def vlm_worker(shared_state):
    load_dotenv()
    url = os.getenv("URL")
    shared_state.vlm_running.value = True
    
    print("[VLM] Worker started")

    while not shared_state.shutdown.is_set():
        current_dist = shared_state.target_dist.value
        
        # Trigger condition: We have a target distance, we aren't already busy, 
        # and we haven't checked this specific movement command yet.
        if (current_dist > 0 and 
            not shared_state.vlm_is_busy.value and 
            abs(current_dist - shared_state.vlm_processed_dist.value) > 1.0):
            
            # 1. Grab the latest frame from the path camera buffer
            if not shared_state.path_frame_ready.is_set():
                time.sleep(0.1)
                continue

            shared_state.vlm_is_busy.value = True
            
            with shared_state.path_frame_buffer.get_lock():
                raw_buf = np.frombuffer(shared_state.path_frame_buffer.get_obj(), dtype=np.uint8)
                frame = raw_buf.reshape((shared_state.PATH_FRAME_H, shared_state.PATH_FRAME_W, 3)).copy()

            # 2. Prepare the image for the API (encode to PNG in memory)
            _, buffer = cv2.imencode('.png', frame)
            img_bytes = buffer.tobytes()

            # 3. Call the API
            try:
                angle = shared_state.target_angle.value
                dist = shared_state.target_dist.value
                
                sys_prompt = "You are a wheelchair safety assistant."
                msg_prompt = f"The wheelchair plans to move {dist}mm at an angle of {angle} degrees. Looking at this camera feed, is the path clear? Answer 'CLEAR' or 'OBSTACLE' followed by a 1-sentence explanation."

                files = {"image": ("frame.png", img_bytes, "image/png")}
                data = {
                    "system_prompt": sys_prompt,
                    "message_prompt": msg_prompt,
                    "max_new_tokens": "100"
                }

                response = requests.post(url, files=files, data=data, timeout=10)
                
                if response.status_code == 200:
                    result = response.json().get("text", "No response") # Adjust based on actual JSON key
                    # Save the verdict back to shared state
                    shared_state.vlm_last_verdict.value = result.encode('utf-8')[:254]
                    print(f"[VLM] Verdict: {result}")
                else:
                    print(f"[VLM] API Error: {response.status_code}")

            except Exception as e:
                print(f"[VLM] Request failed: {e}")
            
            # Mark this distance as processed and release busy flag
            shared_state.vlm_processed_dist.value = current_dist
            shared_state.vlm_is_busy.value = False

        time.sleep(0.2) # Polling rate

    shared_state.vlm_running.value = False