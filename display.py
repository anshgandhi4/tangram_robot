import pygame
import numpy as np

pygame.init()
window = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()

class Polygon:
    def __init__(self, center_x, center_y, offsets, color=(255, 255, 255)):
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
        self.center_x += dx
        self.center_y += dy

    def rotate(self, angle):
        self.angle += angle


sq = Polygon(180, 180, ((-40, -20), (-20, 20), (40, 20), (20, -20)))

run = True
while run:
    clock.tick(100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        sq.move(1, 0)
    if keys[pygame.K_LEFT]:
        sq.move(-1, 0)
    if keys[pygame.K_UP]:
        sq.move(0, -1)
    if keys[pygame.K_DOWN]:
        sq.move(0, 1)

    if keys[pygame.K_s]:
        pygame.image.save(window, "screenshot.png")
        print("Screenshot saved!")

    if keys[pygame.K_r]:
        sq.rotate(np.pi / 100)

    window.fill(100)
    pygame.draw.polygon(window, sq.color, sq.points)

    pygame.display.flip()

pygame.quit()
exit()