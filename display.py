import pygame
import numpy as np
from mcp.server.fastmcp import FastMCP
import io
from PIL import Image
import threading

class Polygon:
    def __init__(self, name, center_x, center_y, offsets, color=(255, 255, 255)):
        self.name = name

        self.center_x = center_x
        self.center_y = center_y

        assert sum([offset[0] for offset in offsets]) == 0 == sum([offset[1] for offset in offsets])

        self.true_offsets = offsets
        self.angle = 0

        self.color = color

    def _rotate_point(self, point):
        point_x, point_y = point
        return (point_x * np.cos(self.angle) - point_y * np.sin(self.angle),
                point_x * np.sin(self.angle) + point_y * np.cos(self.angle))

    @property
    def offsets(self):
        return [self._rotate_point(offset) for offset in self.true_offsets]

    @property
    def points(self):
        return [(self.center_x + off_x, self.center_y + off_y) for off_x, off_y in self.offsets]

    def move(self, dx, dy):
        """Move the current polygon to the right by dx and down by dy pixels"""
        self.center_x += dx
        self.center_y += dy

    def rotate(self, angle):
        """Rotate the current polygon clockwise by angle radians"""
        self.angle += angle

class WindowManager:
    def __init__(self, window):
        self.window = window
        self.polygons = []

    def add_polygon(self, polygon):
        self.polygons.append(polygon)

    def draw_polygons(self):
        for polygon in self.polygons:
            pygame.draw.polygon(self.window, polygon.color, polygon.points)

    def move_polygon(self, name, dx, dy):
        """Finds a polygon by name and moves it to the right by dx and down by dy pixels."""
        for polygon in self.polygons:
            if polygon.name == name:
                polygon.move(dx, dy)
                return True

        print(f"Warning: Polygon {name} not found for movement.")
        return False


    def rotate_polygon(self, name, angle):
        """Finds a polygon by name and rotates it clockwise by angle radians."""
        for polygon in self.polygons:
            if polygon.name == name:
                polygon.rotate(angle)
                return True

        print(f"Warning: Polygon {name} not found for rotation.")
        return False

from mcp.server.fastmcp import FastMCP
mcp = FastMCP("shape_tools")
mcp.add_prompts([])
mcp.add_resources([])

@mcp.tool()
def move_polygon(self, name, dx, dy):
    """Finds a polygon by name and moves it to the right by dx and down by dy pixels."""
    result = window_manager.move_polygon(name, dx, dy)
    pygame_step()
    return result

@mcp.tool()
def rotate_polygon(self, name, angle):
    """Finds a polygon by name and rotates it clockwise by angle radians."""
    result = window_manager.rotate_polygon(name, angle)
    pygame_step()
    return result

@mcp.tool()
def get_observation() -> Image.Image:
    """
    Captures the current pygame window and returns it as a PIL Image.
    This is the 'vision' for the LLM.
    """
    image_data = pygame.image.save_to_buffer(window, "PNG")
    return Image.open(io.BytesIO(image_data))

pygame.init()
window = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()

window_manager = WindowManager(window)

sq = Polygon("sq", 180, 180, ((-40, -40), (40, -40), (40, 40), (-40, 40)))
par = Polygon("par", 30, 60, ((-40, -20), (-20, 20), (40, 20), (20, -20)))
window_manager.add_polygon(sq)
window_manager.add_polygon(par)

_run_pygame = True

def pygame_step():
    global _run_pygame
    if not _run_pygame:
        return False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _run_pygame = False
            return False

    window.fill((10, 10, 10))
    window_manager.draw_polygons()
    pygame.display.flip()

    clock.tick(60)
    return True

if __name__ == "__main__":
    print("Starting fastmcp agent...")

    mcp_thread = threading.Thread(target=mcp.run, daemon=True)
    mcp_thread.start()

    pygame_step()

    while _run_pygame:
        if not pygame_step():
            break

    print("Pygame loop stopped. Shutting down.")
    pygame.quit()
    exit()

#run = True
# while run:
#     clock.tick(100)
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             run = False

#     keys = pygame.key.get_pressed()
#     if keys[pygame.K_RIGHT]:
#         sq.move(1, 0)
#     if keys[pygame.K_LEFT]:
#         sq.move(-1, 0)
#     if keys[pygame.K_UP]:
#         sq.move(0, -1)
#     if keys[pygame.K_DOWN]:
#         sq.move(0, 1)

#     if keys[pygame.K_s]:
#         pygame.image.save(window, "screenshot.png")
#         print("Screenshot saved!")

#     if keys[pygame.K_r]:
#         sq.rotate(np.pi / 100)

#     if keys[pygame.K_a]:
#         sq.rotate(-np.pi / 100)

#     window.fill(100)
#     pygame.draw.polygon(window, sq.color, sq.points)

#     pygame.display.flip()

# pygame.quit()
# exit()
