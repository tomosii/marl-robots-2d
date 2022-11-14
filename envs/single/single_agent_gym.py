import pygame
import math
import sys
import gym
from envs.single.objects import (
    Corridor,
    CorridorOrientation,
    CorridorPosition,
    Goal,
    Room,
    Wall,
)
from envs.single.players import NPC, Agent
from envs.single.utils import line_intersect


class SingleAgentEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    WIDTH = 600
    HEIGHT = 600
    OFFSET = 100
    INFO_HEIGHT = 220
    INFO_MARGIN = 40
    WINDOW_WIDTH = WIDTH + 2 * OFFSET
    WINDOW_HEIGHT = HEIGHT + OFFSET + INFO_HEIGHT

    AGENT_SIZE = 60
    ROOM_SIZE = 260
    GOAL_SIZE = 50
    CORRIDOR_WIDTH = AGENT_SIZE + 40
    WALL_WIDTH = 2

    AGENT_POS = (ROOM_SIZE // 2, HEIGHT - ROOM_SIZE // 2)
    NPC_POS = (WIDTH - ROOM_SIZE // 2, ROOM_SIZE // 2)

    BG_COLOR = (10, 16, 21)
    AGENT_BLUE = (70, 180, 180)
    AGENT_GREEN = (111, 200, 96)
    GOAL_BLUE = (60, 150, 150)
    ROOM_COLOR = (45, 50, 56)
    CORRIDOR_COLOR = (80, 85, 90)
    WALL_COLOR = (60, 67, 75)
    INFO_BG_COLOR = (30, 33, 36)
    INFO_TEXT_COLOR = (220, 220, 220)
    INFO_TEXT_COLOR2 = (145, 150, 155)
    LASER_COLOR = (100, 105, 115)
    LASER_POINT_COLOR = (200, 60, 60)

    FPS = 60

    ACC = 1.2
    FRIC = -0.12
    MAX_SPEED = 6
    NPC_VEL = 6

    LIDAR_ANGLE = 360
    LIDAR_INTERVAL = 5
    LIDAR_RANGE = 1200

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode([self.WINDOW_WIDTH, self.WINDOW_HEIGHT])
        self.map_screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.info_screen = pygame.Surface((self.WINDOW_WIDTH, self.INFO_HEIGHT))


pygame.display.set_caption("Two Corridors")
clock = pygame.time.Clock()


agent = Agent(AGENT_BLUE)
npc = NPC(AGENT_GREEN)


room1 = Room(ROOM_SIZE // 2, HEIGHT - ROOM_SIZE // 2)
room2 = Room(WIDTH - ROOM_SIZE // 2, ROOM_SIZE // 2)

goal1 = Goal(room2.rect.centerx, room2.rect.centery, GOAL_BLUE)

corridor1 = Corridor(
    room1.rect.center,
    room2.rect.center,
    CorridorOrientation.HORIZONTAL,
    CorridorPosition.TOPLEFT,
)
corridor2 = Corridor(
    room2.rect.center,
    room1.rect.center,
    CorridorOrientation.VERTICAL,
    CorridorPosition.TOPLEFT,
)
corridor3 = Corridor(
    room1.rect.center,
    room2.rect.center,
    CorridorOrientation.HORIZONTAL,
    CorridorPosition.BOTTOMRIGHT,
)
corridor4 = Corridor(
    room2.rect.center,
    room1.rect.center,
    CorridorOrientation.VERTICAL,
    CorridorPosition.BOTTOMRIGHT,
)

WALL_OUTER_LENGTH = HEIGHT - ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2
WALL_INNER_LENGTH = WALL_OUTER_LENGTH - CORRIDOR_WIDTH
ROOM_WALL_LENGTH = (ROOM_SIZE - CORRIDOR_WIDTH) // 2


walls = pygame.sprite.Group(
    Wall(
        (
            (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_OUTER_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (
            (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_OUTER_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
    Wall(
        (
            ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_INNER_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (
            ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ROOM_SIZE - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_INNER_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
    Wall(
        (
            WIDTH - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ROOM_SIZE,
        ),
        WALL_OUTER_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (
            WIDTH - ROOM_SIZE + (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
            ROOM_SIZE,
        ),
        WALL_INNER_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (
            ROOM_SIZE,
            HEIGHT - (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_OUTER_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
    Wall(
        (
            ROOM_SIZE,
            HEIGHT - ROOM_SIZE + (ROOM_SIZE - CORRIDOR_WIDTH) // 2,
        ),
        WALL_INNER_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
    Wall((0, HEIGHT - ROOM_SIZE), ROOM_SIZE, CorridorOrientation.VERTICAL),
    Wall((0, HEIGHT), ROOM_SIZE, CorridorOrientation.HORIZONTAL),
    Wall((WIDTH, 0), ROOM_SIZE, CorridorOrientation.VERTICAL),
    Wall((WIDTH - ROOM_SIZE, 0), ROOM_SIZE, CorridorOrientation.HORIZONTAL),
    Wall((0, HEIGHT - ROOM_SIZE), ROOM_WALL_LENGTH, CorridorOrientation.HORIZONTAL),
    Wall(
        (ROOM_SIZE - ROOM_WALL_LENGTH, HEIGHT - ROOM_SIZE),
        ROOM_WALL_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
    Wall(
        (ROOM_SIZE, HEIGHT - ROOM_SIZE), ROOM_WALL_LENGTH, CorridorOrientation.VERTICAL
    ),
    Wall(
        (ROOM_SIZE, HEIGHT - ROOM_WALL_LENGTH),
        ROOM_WALL_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall((WIDTH - ROOM_SIZE, 0), ROOM_WALL_LENGTH, CorridorOrientation.VERTICAL),
    Wall(
        (WIDTH - ROOM_SIZE, ROOM_SIZE - ROOM_WALL_LENGTH),
        ROOM_WALL_LENGTH,
        CorridorOrientation.VERTICAL,
    ),
    Wall(
        (WIDTH - ROOM_SIZE, ROOM_SIZE), ROOM_WALL_LENGTH, CorridorOrientation.HORIZONTAL
    ),
    Wall(
        (WIDTH - ROOM_WALL_LENGTH, ROOM_SIZE),
        ROOM_WALL_LENGTH,
        CorridorOrientation.HORIZONTAL,
    ),
)

maps = pygame.sprite.Group(
    room1,
    room2,
    goal1,
    corridor1,
    corridor2,
    corridor3,
    corridor4,
)

players = pygame.sprite.Group(agent, npc)


def laser_scan(laser, obstacle_lines):
    min_intersection = laser[1]
    min_distance = LIDAR_RANGE
    for obstacle in obstacle_lines:
        intersection = line_intersect(laser[0], laser[1], obstacle[0], obstacle[1])
        if intersection is not None:
            distance = math.sqrt(
                (intersection[0] - laser[0][0]) ** 2
                + (intersection[1] - laser[0][1]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                min_intersection = intersection

    return min_intersection


def get_obstacle_lines(walls, npc):
    lines = []
    for wall in walls:
        lines.append((wall.start, wall.end))
    lines.append((npc.rect.topleft, npc.rect.topright))
    lines.append((npc.rect.topleft, npc.rect.bottomleft))
    lines.append((npc.rect.topright, npc.rect.bottomright))
    lines.append((npc.rect.bottomleft, npc.rect.bottomright))
    return lines


def reset():
    pygame.display.flip()
    pygame.time.wait(200)
    agent.reset(AGENT_POS)
    npc.reset(NPC_POS)


if __name__ == "__main__":
    reset()

    episode = 0
    episode_font = pygame.font.SysFont("ubuntu", 30)
    result_font = pygame.font.SysFont("ubuntu", 45)
    info_font = pygame.font.SysFont("ubuntu", 22)

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill(BG_COLOR)
        map_screen.fill(BG_COLOR)
        info_screen.fill(BG_COLOR)

        maps.draw(map_screen)
        walls.draw(map_screen)

        agent.move()
        npc.auto_move()

        lasers = agent.create_lasers()
        obstacle_lines = get_obstacle_lines(walls, npc)
        for laser in lasers:
            intersection = laser_scan(laser, obstacle_lines)
            pygame.draw.line(map_screen, LASER_COLOR, agent.pos, intersection)
            pygame.draw.circle(map_screen, LASER_POINT_COLOR, intersection, 2.5)

        players.draw(map_screen)

        episode_text = episode_font.render(f"Episode: {episode}", True, INFO_TEXT_COLOR)
        timestep_text = distance_text = info_font.render(
            f"Timestep: 0", True, INFO_TEXT_COLOR2
        )
        distance_text = info_font.render(
            f"Distance from goal: {agent.distance_from_goal(goal1):.0f}",
            True,
            INFO_TEXT_COLOR2,
        )
        pygame.draw.rect(
            info_screen,
            INFO_BG_COLOR,
            (
                INFO_MARGIN,
                INFO_MARGIN,
                WINDOW_WIDTH - 2 * INFO_MARGIN,
                INFO_HEIGHT - 2 * INFO_MARGIN,
            ),
            0,
            20,
        )
        info_screen.blit(episode_text, (INFO_MARGIN + 24, INFO_MARGIN + 20))
        info_screen.blit(timestep_text, (INFO_MARGIN + 24, INFO_MARGIN + 65))
        info_screen.blit(distance_text, (INFO_MARGIN + 24, INFO_MARGIN + 95))

        screen.blit(map_screen, (OFFSET, INFO_HEIGHT))
        screen.blit(info_screen, (0, 0))

        pygame.display.flip()

        if agent.check_goal(goal1):
            result_text = result_font.render("CLEAR !!", True, INFO_TEXT_COLOR)
            screen.blit(result_text, (400, 80))
            reset()
            episode += 1

        if agent.collide_npc(npc) or agent.collide_walls(walls.sprites()):
            result_text = result_font.render("FAILED", True, INFO_TEXT_COLOR)
            screen.blit(result_text, (400, 80))
            reset()
            episode += 1

    pygame.quit()
    sys.exit()
