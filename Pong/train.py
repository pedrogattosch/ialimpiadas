from envpong import PongEnv
from bot import BotLeft, BotRight
import torch
import time

def main():
    start_time = time.time()

    env = PongEnv(debugPrint=False)
    player1_bot = BotLeft(env, is_training=True)
    player2_bot = BotRight(env)

    num_episodes = 2000

    last_reward = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        last_score_diff = 0
        last_dist_y = abs(obs[1] - obs[9])

        while not done:
            action_p1 = player1_bot.act(obs)
            action_p2 = player2_bot.act()
            new_obs, reward_from_env, done, truncated, info = env.step(action_p1, action_p2)

            # 1. Recompensa por marcar/sofrer ponto
            score_diff = env.game.states[-1].player1Score - env.game.states[-1].player2Score
            final_reward = score_diff - last_score_diff
            last_score_diff = score_diff

            # 2. Recompensa por se aproximar da bola
            current_dist_y = abs(new_obs[1] - new_obs[9])
            distance_reward = last_dist_y - current_dist_y
            last_dist_y = current_dist_y
            
            # 3. Recompensa por rebater a bola
            hit_reward = 0
            ball_vx_before = obs[10]
            ball_vx_after = new_obs[10]
            if ball_vx_before < 0 and ball_vx_after > 0:
                hit_reward = 0.5

            total_reward = final_reward + (distance_reward * 10) + hit_reward

            experience = (obs, action_p1, total_reward, new_obs, done)
            player1_bot.observe(experience)
            player1_bot.learn()

            obs = new_obs

        print(f"Episódio {episode + 1}/{num_episodes} concluído.")

        if (episode + 1) % 100 == 0:
            torch.save(player1_bot.model.state_dict(), player1_bot.model_path)

    end_time = time.time()
    total_seconds = int(end_time - start_time)

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Tempo total de treinamento: {hours} horas, {minutes} minutos e {seconds} segundos.")


if __name__ == "__main__":
    main()