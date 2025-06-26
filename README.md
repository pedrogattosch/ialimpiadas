# Pong

Este projeto apresenta a implementação de uma inteligência artificial capaz de aprender a jogar o clássico jogo Pong do zero, utilizando técnicas de aprendizado por reforço. O agente foi treinado para controlar o paddle da esquerda e, após o treinamento, alcançou uma taxa de vitória de 100% contra um oponente que joga de forma aleatória.

---

##  Requisitos

* Python versão 3.9
* `numpy`
* `torch`
* `gym==0.25.2`
* `arcade==2.6.3`

---

## Instalação

1. Crie e ative um ambiente virtual:

```bash
# Crie o ambiente na pasta do projeto
python3.9 -m venv .venv

# Ative o ambiente (Linux/macOS)
source .venv/bin/activate

# Ative o ambiente (Windows)
.venv\Scripts\activate
```

2. Instale as dependências:

```bash
pip install numpy torch "gym==0.25.2" "arcade==2.6.3"
```

---

## Execução

1. Treinar o agente (opcional):

```bash
python train.py
```

O processo pode levar bastante tempo. Ao final, será salvo o arquivo `modelo_pong.pth`.

2. Ver o agente:

```bash
python pongPlayGUI.py
```

A janela do jogo abrirá e o paddle da esquerda será controlado pela IA.

---

## Explicação

### Algoritmo: Deep Q-Network (DQN)

#### Arquitetura da rede neural:

* **Entrada:** 14 neurônios (estado do jogo: posições e velocidades da bola e dos paddles).
* **Camadas ocultas:** 128 e 64 neurônios.
* **Saída:** 3 neurônios (ações possíveis: SUBIR, FICAR PARADO, DESCER).

### Aprendizado do agente:

1. **Observação:** Recebe 14 números que descrevem o estado atual do jogo.
2. **Ação:** Calcula os Q-values para as três ações possíveis e escolhe a de maior valor.
3. **Recompensa:** Recebe +1 ao marcar ponto e -1 ao sofrer ponto.
4. **Aprendizado:** Utiliza memória de repetição e o algoritmo DQN para melhorar suas decisões com o tempo.

---

## Estrutura

* `bot.py`: Implementação da rede neural e lógica do agente.
* `train.py`: Script de treinamento.
* `envpong.py`: Implementação do ambiente e regras do jogo.
* `pongPlayGUI.py`: Execução do jogo com interface gráfica.
* `pongPlayNOGUI.py`: Execução do jogo sem interface gráfica.
* `modelo_pong.pth`: Modelo treinado.

