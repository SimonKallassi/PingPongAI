import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PongEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        pygame.init()

        # Constants
        self.WIDTH = 600
        self.HEIGHT = 400
        self.PADDLE_WIDTH = 10
        self.PADDLE_HEIGHT = 60
        self.BALL_SIZE = 8
        self.PADDLE_SPEED = 8
        self.BALL_SPEED = 7

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        # Game state variables
        self.current_game = 0
        self.total_games = 0
        self.max_hits = 0
        self.game_speed = 60

        # Rendering setup
        self.render_mode = render_mode
        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption('Pong RL')
            self.clock = pygame.time.Clock()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: stay, 1: up, 2: down

        # Observation space: [ball_x, ball_y, ball_vx, ball_vy, paddle_y, opponent_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -self.BALL_SPEED, -self.BALL_SPEED, 0, 0]),
            high=np.array([self.WIDTH, self.HEIGHT,
                           self.BALL_SPEED, self.BALL_SPEED,
                           self.HEIGHT, self.HEIGHT]),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Reset ball to center with random velocity
        self.ball_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        angle = self.np_random.uniform(-45, 45)
        self.ball_vel = np.array([
            -self.BALL_SPEED * np.cos(np.radians(angle)),
            self.BALL_SPEED * np.sin(np.radians(angle))
        ])

        # Reset paddles to center
        self.paddle_left = self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2
        self.paddle_right = self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2

        # Reset hits for this round
        self.hits = 0

        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([
            self.ball_pos[0],
            self.ball_pos[1],
            self.ball_vel[0],
            self.ball_vel[1],
            self.paddle_left,
            self.paddle_right
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False

        # Move AI paddle (left)
        if action == 1:  # Up
            self.paddle_left = max(0, self.paddle_left - self.PADDLE_SPEED)
        elif action == 2:  # Down
            self.paddle_left = min(
                self.HEIGHT - self.PADDLE_HEIGHT,
                self.paddle_left + self.PADDLE_SPEED
            )

        # Simple opponent AI
        if self.ball_vel[0] > 0:  # Ball moving right
            if self.ball_pos[1] > self.paddle_right + self.PADDLE_HEIGHT / 2:
                self.paddle_right = min(
                    self.HEIGHT - self.PADDLE_HEIGHT,
                    self.paddle_right + self.PADDLE_SPEED * 0.85
                )
            elif self.ball_pos[1] < self.paddle_right + self.PADDLE_HEIGHT / 2:
                self.paddle_right = max(
                    0,
                    self.paddle_right - self.PADDLE_SPEED * 0.85
                )

        # Update ball position
        self.ball_pos += self.ball_vel

        # Ball collision with top and bottom
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.HEIGHT:
            self.ball_vel[1] = -self.ball_vel[1]
            self.ball_pos[1] = np.clip(self.ball_pos[1], 0, self.HEIGHT)

        # Ball collision with paddles
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_SIZE / 2,
            self.ball_pos[1] - self.BALL_SIZE / 2,
            self.BALL_SIZE,
            self.BALL_SIZE
        )

        left_paddle = pygame.Rect(
            0,
            self.paddle_left,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        right_paddle = pygame.Rect(
            self.WIDTH - self.PADDLE_WIDTH,
            self.paddle_right,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        # Paddle hit detection and reward
        if ball_rect.colliderect(left_paddle):
            self.hits += 1
            self.ball_vel[0] = abs(self.ball_vel[0])  # Ensure ball moves right
            reward = 1.0  # Reward for hitting

            # Adjust vertical velocity based on where the ball hits the paddle
            relative_intersect_y = (self.paddle_left + self.PADDLE_HEIGHT / 2) - self.ball_pos[1]
            normalized_intersect = relative_intersect_y / (self.PADDLE_HEIGHT / 2)
            self.ball_vel[1] = -normalized_intersect * self.BALL_SPEED

        elif ball_rect.colliderect(right_paddle):
            self.hits += 1
            self.ball_vel[0] = -abs(self.ball_vel[0])  # Ensure ball moves left

        # Ball out of bounds
        if self.ball_pos[0] < 0:
            reward = -1.0
            done = True
        elif self.ball_pos[0] > self.WIDTH:
            reward = 0.5  # Small reward for winning point
            done = True

        # Additional positioning reward
        if not done:
            # Reward for being aligned with ball
            paddle_center = self.paddle_left + self.PADDLE_HEIGHT / 2
            ball_distance = abs(paddle_center - self.ball_pos[1])
            position_reward = 0.1 * (1 - ball_distance / self.HEIGHT)
            reward += position_reward

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, done, False, {"hits": self.hits}

    def _render_frame(self):
        self.screen.fill(self.BLACK)

        # Draw paddles
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            pygame.Rect(0, self.paddle_left, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        )
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            pygame.Rect(
                self.WIDTH - self.PADDLE_WIDTH,
                self.paddle_right,
                self.PADDLE_WIDTH,
                self.PADDLE_HEIGHT
            )
        )

        # Draw ball
        pygame.draw.circle(
            self.screen,
            self.WHITE,
            (int(self.ball_pos[0]), int(self.ball_pos[1])),
            self.BALL_SIZE // 2
        )

        # Draw center line
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.rect(
                self.screen,
                self.WHITE,
                pygame.Rect(self.WIDTH // 2 - 2, i, 4, 10)
            )

        # Draw stats
        font = pygame.font.Font(None, 36)
        stats_texts = [
            f"Game: {self.current_game}/{self.total_games}",
            f"Rally: {self.hits}  Max: {self.max_hits}",
            f"Speed: {self.game_speed} FPS"
        ]

        for i, text in enumerate(stats_texts):
            surface = font.render(text, True, self.WHITE)
            self.screen.blit(surface, (10, 10 + i * 30))

        pygame.display.flip()
        if hasattr(self, 'clock'):
            self.clock.tick(self.game_speed if self.game_speed > 0 else 0)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()