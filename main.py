import pygame
from pygame import gfxdraw
# import numpy as np

width = 700
height = 700


def draw():

    screen = pygame.display.set_mode((width, height))
    # clear screen
    screen.fill((255, 255, 255))

    # update bounds to center around agent
    # all_poses = [entity.state.p_pos for entity in self.world.entities]
    # cam_range = np.max(np.abs(np.array(all_poses)))

    # update geometry and text positions
    # text_line = 0
    # for e, entity in enumerate(self.world.entities):

    # geometry
    x, y = 1, 1
    y *= (
        -1
    )  # this makes the display mimic the old pyglet setup (ie. flips image)
    x = (
        (x) * width // 2 * 0.9
    )  # the .9 is just to keep entities from appearing "too" out-of-bounds
    y = (y) * height // 2 * 0.9
    x += width // 2
    y += height // 2
    pygame.draw.circle(
        screen, (0.25, 0.25, 0.25), (x, y), 0.05 * 350
    )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
    pygame.draw.circle(
        screen, (0, 0, 0), (x, y), 0.05 * 350, 1
    )  # borders
    # assert (
    #     0 < x < self.width and 0 < y < self.height
    # ), f"Coordinates {(x, y)} are out of bounds."

    # text
    # if isinstance(entity, Agent):
    #     if entity.silent:
    #         continue
    #     if np.all(entity.state.c == 0):
    #         word = "_"
    #     elif self.continuous_actions:
    #         word = (
    #             "[" +
    #             ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
    #         )
    #     else:
    #         word = alphabet[np.argmax(entity.state.c)]

    #     message = entity.name + " sends " + word + "   "
    #     message_x_pos = self.width * 0.05
    #     message_y_pos = self.height * 0.95 - \
    #         (self.height * 0.05 * text_line)
    #     self.game_font.render_to(
    #         self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
    #     )
    #     text_line += 1


if __name__ == "__main__":
    pygame.init()

    screen = pygame.display.set_mode([700, 700])

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((16, 16, 16))

        pygame.draw.circle(screen, (97, 205, 205), (250, 250), 75)

        gfxdraw.aacircle(screen, 500, 250, 75, (97, 205, 205))
        gfxdraw.filled_circle(screen, 500, 250, 75, (97, 205, 205))

        pygame.display.flip()

    pygame.quit()

    # draw()
    # pygame.display.flip()
    # input()
