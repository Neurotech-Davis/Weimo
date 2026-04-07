import math
import pygame
import time
from rplidar import RPLidar, RPLidarException

# Screen setup
WIDTH, HEIGHT = 800, 800
CENTER = (WIDTH // 2, HEIGHT // 2)
MIN_DISTANCE = 50     # mm (5 cm)
MAX_DISTANCE = 10000   # mm (300 cm)
SCALE = (WIDTH // 2) / MAX_DISTANCE

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
DARK_GREEN = (0, 100, 0)
WHITE = (255, 255, 255)

def polar_to_cartesian(angle_deg, distance_mm):
    angle_rad = math.radians(angle_deg)
    r = distance_mm * SCALE
    x = CENTER[0] + int(r * math.cos(angle_rad))
    y = CENTER[1] - int(r * math.sin(angle_rad))
    return x, y

def lidar_graphing():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LiDAR Mapping")
    clock = pygame.time.Clock()

    font_small = pygame.font.SysFont(None, 20)
    font_title = pygame.font.SysFont(None, 28, bold=True)

    # The following is all according to documentation
    PORT = "COM3"
    BAUD = 115200
    lidar = RPLidar(PORT, baudrate=BAUD, timeout=1)
    lidar.start_motor()
    time.sleep(0.2)

    running = True

    try:
        for scan in lidar.iter_scans(max_buf_meas=500, min_len=3):
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            if not running:
                break

            screen.fill(BLACK)

            for r in range(500, MAX_DISTANCE + 1, 500):
                pygame.draw.circle(screen, DARK_GREEN, CENTER, int(r * SCALE), 1)
                label = font_small.render(f"{r // 10} cm", True, WHITE)
                screen.blit(label, (CENTER[0] + int(r * SCALE) - 25, CENTER[1]))
                
            for quality, ang, dist in scan:
                if MIN_DISTANCE <= dist <= MAX_DISTANCE:
                    px, py = polar_to_cartesian(ang, dist)

                    if dist <= 1000:    # Within 1 meter
                        color = RED
                    elif dist <= 2000:  # Within 2 meters
                        color = YELLOW
                    else:
                        color = GREEN

                    pygame.draw.circle(screen, color, (px,py), 2)
            title_surface = font_title.render("Lidar Graphing", True, WHITE)
            screen.blit(title_surface, (WIDTH // 2 - title_surface.get_width() // 2, HEIGHT - 40))

            pygame.display.flip()
            clock.tick(30)

            if not running:
                break

    except KeyboardInterrupt:
        print("\nStopped")

    finally:
        try:
            lidar.stop()
        except Exception:
            pass

        try:
            lidar.stop_motor()
        except Exception:
            pass

        try:
            lidar.disconnect()
        except Exception:
            pass

        pygame.quit()

if __name__ == "__main__":
    lidar_graphing()