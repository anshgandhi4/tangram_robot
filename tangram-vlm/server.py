import pygame
import numpy as np
import math
from mcp.server.fastmcp import FastMCP
import io
from PIL import Image
import threading
import base64
import asyncio
import json

class Polygon:
    def __init__(self, name, center_x, center_y, offsets, color=(255, 255, 255)):
        self.name = name
        self._start_center_x = center_x
        self._start_center_y = center_y

        self.center_x = self._start_center_x
        self.center_y = self._start_center_y

        assert sum([offset[0] for offset in offsets]) == 0 == sum([offset[1] for offset in offsets])
        self.true_offsets = offsets
        self.start_angle = 0

        self.angle = self.start_angle
        self.color = color

    def _rotate_point(self, point):
        point_x, point_y = point
        return (point_x * np.cos(self.angle) - point_y * np.sin(self.angle),
                point_x * np.sin(self.angle) + point_y * np.cos(self.angle))

    @property
    def offsets(self):
        return [self._rotate_point(offset) for offset in self.true_offsets]

    @property
    def start_center(self):
        return self._start_center_x, self._start_center_y

    @property
    def points(self):
        return [(self.center_x + off_x, self.center_y + off_y) for off_x, off_y in self.offsets]

    def move(self, dx, dy):
        self.center_x += dx
        self.center_y += dy

    def rotate(self, angle):
        self.angle += angle

class RightTriangle(Polygon):
    def __init__(self, name, center_x, center_y, leg_length, color=(255, 255, 255), angle=0):
        offsets_0 = (-leg_length / 3, -leg_length / 3)
        offsets_1 = (-leg_length / 3, 2 * leg_length / 3)
        offsets_2 = (2 * leg_length / 3, -leg_length / 3)
        offsets = [offsets_0, offsets_1, offsets_2]

        super().__init__(name, center_x, center_y, offsets, color)
        self.rotate(angle)


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
        for polygon in self.polygons:
            if polygon.name == name:
                polygon.move(dx, dy)
                return True
        print(f"Warning: Polygon {name} not found for movement.")
        return False

    def rotate_polygon(self, name, angle):
        for polygon in self.polygons:
            if polygon.name == name:
                polygon.rotate(angle)
                return True
        print(f"Warning: Polygon {name} not found for rotation.")
        return False

    def get_all_points(self):
        return {polygon.name: [{"x": pt[0], "y": pt[1]} for pt in polygon.points] for polygon in self.polygons}

pygame.init()
window = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()
window_manager = WindowManager(window)

scale = 3

yellow_sq = Polygon("yellow", 40, 40,
                    ((-11 * scale, -11 * scale),
                     (11 * scale, -11 * scale),
                     (11 * scale, 11 * scale),
                     (-11 * scale, 11 * scale)),
                     color=pygame.color.Color("yellow1"))

red_tri = RightTriangle("red", 120, 50, 30 * scale, color=pygame.Color("firebrick1"))

blue_tri = RightTriangle("blue", 200, 250, 22 * scale, color=pygame.Color("blue"))
pink_tri = RightTriangle("pink", 300, 300, 22 * scale, color=pygame.Color("lightpink2"))

green_tri = RightTriangle("green", 300, 180, 43 * scale, color=pygame.Color("chartreuse3"))
terra_cotta_tri = RightTriangle("terra cotta", 80, 300, 43 * scale, color=pygame.Color("coral3"))

height_off = (21/math.sqrt(2) / 2) * scale
horiz_long_off = (15 + 21/math.sqrt(2) / 2) * scale
horiz_short_off = (15 - 21/math.sqrt(2) / 2) * scale
purple_parallelogram = Polygon("purple", 300, 60, ((horiz_short_off, height_off),
                                                   (-horiz_long_off, height_off),
                                                   (-horiz_short_off, -height_off),
                                                   (horiz_long_off, -height_off)),
                                                   color=pygame.color.Color("mediumpurple3"))

window_manager.add_polygon(yellow_sq)
window_manager.add_polygon(red_tri)
window_manager.add_polygon(blue_tri)
window_manager.add_polygon(pink_tri)
window_manager.add_polygon(green_tri)
window_manager.add_polygon(terra_cotta_tri)
window_manager.add_polygon(purple_parallelogram)

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


# --- FastMCP Agent Setup ---
mcp = FastMCP("shape_tools")

@mcp.tool()
async def move_polygon(name: str, dx: float, dy: float) -> bool:
    """Moves a polygon by dx, dy."""
    await asyncio.sleep(1)
    result = window_manager.move_polygon(name, dx, dy)
    return result

@mcp.tool()
async def rotate_polygon(name: str, angle: float) -> bool:
    """Rotates a polygon by angle radians."""
    await asyncio.sleep(1)
    result = window_manager.rotate_polygon(name, angle)
    return result

@mcp.tool()
async def get_observation() -> str:
    """Captures the current pygame window as a base64 PNG."""
    await asyncio.sleep(1)
    raw_str = pygame.image.tostring(window, "RGB")
    img = Image.frombytes("RGB", window.get_size(), raw_str)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    base64_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"

@mcp.tool()
async def get_observation_points() -> str:
    """Return the coordinates of all points of all polygons in the window """
    await asyncio.sleep(1)
    return json.dumps(window_manager.get_all_points())

def start_mcp_server():
    """Runs the FastMCP server in this function."""
    print("Starting FastMCP server on http://localhost:8000 ...")
    mcp.run(transport="sse")

mcp_thread = threading.Thread(target=start_mcp_server, daemon=True)
mcp_thread.start()

print("Starting Pygame main loop...")

while pygame_step():
    pass

print("Pygame window closed. Shutting down.")
pygame.quit()