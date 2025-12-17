import gymnasium as gym
import pygame
import numpy as np

env = gym.make("WheelchairRacing-v0", render_mode="human", domain_randomize=False, continuous=True)
obs, info = env.reset()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            break

    keys = pygame.key.get_pressed()
    steer, gas, brake = 0.0, 0.0, 0.0
    if keys[pygame.K_UP]:
        gas = 0.3
    if keys[pygame.K_DOWN]:
        brake = 0.8
    if keys[pygame.K_LEFT]:
        steer = -0.25
    elif keys[pygame.K_RIGHT]: 
        steer = 0.25
    
    action = np.array([steer, gas, brake], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)

env.close()
pygame.quit()
