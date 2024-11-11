import numpy as np
from pong_env import PongEnv
from dqn_agent import DQNAgent
import time
import keyboard
import argparse
import pygame


def train_dqn(render=True, load_model=None, episodes=10000):
    # Create environment
    env = PongEnv(render_mode="human" if render else None)

    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Load previous training if specified
    if load_model:
        if not agent.load_model(load_model):
            print("\nStarting fresh training instead.")

    # Training settings
    save_interval = 100
    max_steps = 5000
    running = True
    display_enabled = render
    game_speed = 60 if render else 0
    last_time = time.time()

    print("\nControls:")
    print("0: Toggle display on/off")
    print("UP: Increase game speed")
    print("DOWN: Decrease game speed")
    print("Ctrl+C: Save and quit")

    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            hits = 0

            # Update game info
            env.current_game = episode + 1
            env.total_games = episodes

            for step in range(max_steps):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        continue

                try:
                    # For display toggle
                    current_time = time.time()
                    if keyboard.is_pressed('0') and current_time - last_time > 0.3:
                        display_enabled = not display_enabled
                        env.render_mode = "human" if display_enabled else None
                        print(f"\nDisplay {'enabled' if display_enabled else 'disabled'}")
                        print(f"Current Speed: {'unlimited' if game_speed == 0 else f'{game_speed} FPS'}")
                        last_time = current_time

                    # For speed adjustments
                    if keyboard.is_pressed('up'):
                        if game_speed == 0:
                            game_speed = 60
                        else:
                            game_speed = min(10000, game_speed * 2)
                        env.game_speed = game_speed
                        print(f"\nSpeed: {'unlimited' if game_speed == 0 else f'{game_speed} FPS'}")
                        time.sleep(0.1)
                    elif keyboard.is_pressed('down'):
                        game_speed = max(0, game_speed // 2)
                        env.game_speed = game_speed
                        print(f"\nSpeed: {'unlimited' if game_speed == 0 else f'{game_speed} FPS'}")
                        time.sleep(0.1)
                except:
                    # If keyboard fails don't worry just continue
                    pass

                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, done, _, info = env.step(action)

                # Get current hits and update max
                hits = info.get('hits', 0)
                env.max_hits = max(env.max_hits, hits)

                # Store transition and train
                agent.memory.push(state, action, reward, next_state, done)
                agent.train_step()

                state = next_state
                total_reward += reward

                if done:
                    break

            # Update target network periodically
            if episode % agent.target_update == 0:
                agent.update_target_network()

            # Save progress periodically
            if (episode + 1) % save_interval == 0:
                agent.save_model(episode + 1)
                epsilon = agent.eps_end + (agent.eps_start - agent.eps_end) * \
                          np.exp(-1. * agent.steps_done / 10000)
                print(f"\nEpisode {episode + 1}/{episodes}")
                print(f"Avg Reward (last 100): {total_reward:.2f}")
                print(f"Current Rally: {hits}")
                print(f"Epsilon: {epsilon:.3f} (Steps: {agent.steps_done})")
                print(f"Memory size: {len(agent.memory)}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final model
        agent.save_model(episode + 1)
        print("\nFinal model saved")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN for Pong')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--load', type=str, help='Load model from file')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')

    args = parser.parse_args()

    train_dqn(
        render=not args.no_render,
        load_model=args.load,
        episodes=args.episodes
    )