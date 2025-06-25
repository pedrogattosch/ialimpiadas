# pongPlayGUI.py

import arcade
# A importação de PongLogic é necessária para o oponente aleatório
from envpong import PongGUIEnv, PongLogic
# Importamos o BotLeft (nossa IA) e BotRight (oponente)
from bot import BotLeft, BotRight
import os
import time
import threading
import random # <-- Adicionado para o oponente aleatório

def runLoop(env):
    # Player 1 é a nossa IA treinada
    player1_bot = BotLeft(env, is_training=False)
    # Player 2 pode ser um bot aleatório para teste visual
    player2_bot = BotRight(env)

    # Pega a observação (estado) inicial do jogo
    obs, info = env.reset()

    # Loop infinito para que o jogo continue rodando na janela
    while True: 
        # A IA (Player 1) decide a ação com base na observação atual
        actionp1 = player1_bot.act(obs)
        # O oponente (Player 2) decide sua ação
        actionp2 = player2_bot.act()
        
        # O ambiente executa um passo com as ações dos bots
        new_obs, reward, done, truncated, info = env.step(actionp1, actionp2)
        
        # Atualiza a observação para o próximo ciclo do loop
        obs = new_obs
        
        # Se uma partida terminou (alguém marcou ponto), reinicia o jogo
        if done:
            obs, info = env.reset()

        # Pausa para sincronizar a velocidade da lógica com a taxa de atualização da tela
        time.sleep(env.game.dt)
        
def main():
    # Cria o ambiente com a interface gráfica
    env = PongGUIEnv()
    
    # Inicia a lógica do jogo em uma thread separada para não travar a GUI
    threading.Thread(target=runLoop, args=(env,), daemon=True).start()
    
    # Roda a janela do Arcade (esta deve ser a thread principal)
    arcade.run()
        
############################
        
if __name__ == "__main__":
    # Define o diretório de trabalho
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_path)

    main()