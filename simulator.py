import gymnasium as gym
import pygame
from get_action import get_eeg_action

pygame.init()

def read_event_id():
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return 2
    if keys[pygame.K_RIGHT]:
        return 3
    if keys[pygame.K_UP]:
        return 4
    if keys[pygame.K_DOWN]:
        return 5
    return 1

env = gym.make("CarRacing-v3", render_mode="human")
obs, info = env.reset()

done = False
last_action = [0.0, 0.0, 0.0]

INTERVAL_MS = 1000
next_update = pygame.time.get_ticks()

while not done:
    now = pygame.time.get_ticks()

    if now >= next_update:
        event_id = read_event_id()
        last_action = get_eeg_action(event_id)
        next_update = now + INTERVAL_MS

    obs, reward, terminated, truncated, info = env.step(last_action)
    done = terminated or truncated

env.close()
pygame.quit()