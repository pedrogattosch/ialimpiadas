import os
from envpong import PongEnv, PongLogic
from bot import BotLeft 
import random

def run_evaluation(env, num_games=100):
    player1_ai = BotLeft(env, is_training=False)
    
    p1_wins = 0
    p2_wins = 0

    for game in range(num_games):
        obs, info = env.reset()
        done = False
        
        while not done:
            action_p1 = player1_ai.act(obs)
            
            action_p2 = random.choice([PongLogic.PaddleMove.DOWN, PongLogic.PaddleMove.STILL, PongLogic.PaddleMove.UP])

            new_obs, reward, done, truncated, info = env.step(action_p1, action_p2)
            obs = new_obs

        final_score_p1 = env.game.states[-1].player1Score
        final_score_p2 = env.game.states[-1].player2Score
        
        prev_score_p1 = env.game.states[-2].player1Score
        prev_score_p2 = env.game.states[-2].player2Score

        if final_score_p1 > prev_score_p1:
            p1_wins += 1
            print(f"Partida {game + 1}/{num_games}: Vitória da IA")
        else:
            p2_wins += 1
            print(f"Partida {game + 1}/{num_games}: Derrota da IA")

    print("\n--- RESULTADOS FINAIS ---")
    print(f"Partidas totais: {num_games}")
    print(f"Vitórias da IA: {p1_wins}")
    print(f"Derrotas da IA: {p2_wins}")
    
    win_rate = (p1_wins / num_games) * 100
    print(f"Taxa de vitória: {win_rate:.2f}%")
    
    return win_rate

def main():
    env = PongEnv(debugPrint=False)
    run_evaluation(env, num_games=100)

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)
    main()