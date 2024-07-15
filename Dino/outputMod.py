from statistics import mean
import pygame
import os
import random
import time
from sys import exit

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"
RENDER_GAME = False

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
if RENDER_GAME:
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [
    pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
    pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png")),
]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [
    pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
    pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png")),
]

SMALL_CACTUS = [
    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png")),
]
LARGE_CACTUS = [
    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png")),
]

BIRD = [
    pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
    pygame.image.load(os.path.join("Assets/Bird", "Bird2.png")),
]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))
        pygame.draw.rect(
            SCREEN,
            self.color,
            (
                self.dino_rect.x,
                self.dino_rect.y,
                self.dino_rect.width,
                self.dino_rect.height,
            ),
            2,
        )

    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return self.type


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(
        self,
        distance,
        obHeight,
        speed,
        obType,
        nextObDistance,
        nextObHeight,
        nextObType,
    ):
        pass

    def updateState(self, state):
        pass


def first(x):
    return x[0]


# def sigmoid(x):
#     """
#     Função sigmoide numericamente estável.

#     Args:
#     x (numpy array): Entrada para a função sigmoide.

#     Returns:
#     numpy array: Saída da função sigmoide.
#     """
#     # Usar a versão numericamente estável da função sigmoide
#     pos_mask = x >= 0
#     neg_mask = ~pos_mask
#     z = np.zeros_like(x)
#     z[pos_mask] = np.exp(-x[pos_mask])
#     z[neg_mask] = np.exp(x[neg_mask])
#     top = np.ones_like(x)
#     top[neg_mask] = z[neg_mask]
#     return top / (1 + z)\


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


from sklearn.preprocessing import normalize


def fixInput(inputs):

    classes = {
        Bird: -1,
        SmallCactus: 2,
        LargeCactus: 3,
        Cloud: 4,
        Dinosaur: 5,
        Obstacle: 6,
    }

    for i, inp in enumerate(inputs):
        if i == 3:
            if isinstance(inputs[i], tuple(classes.keys())):
                inputs[i] = classes[type(inputs[i])]
        elif i == 6:
            if isinstance(inputs[i], tuple(classes.keys())):
                inputs[i] = classes[type(inputs[i])]
        else:
            inputs[i] = inp

    inputs = np.array(inputs) / (np.sum(inputs) + 1e-9)
    # inputs = normalize(inputs)
    return inputs


# Classe Neuronio
class Neuronio:
    def __init__(self, weights, bias, function):
        self.weights = np.array(weights)
        self.bias = bias
        self.function = function

    def forward(self, inputs):
        output = self.function(np.dot(self.weights, inputs) + self.bias)
        return output

    def setWeights(self, weights):
        self.weights = np.array(weights)


# Classe CamadaNeuronio
class CamadaNeuronio:
    def __init__(self, neuronios):
        self.neuronios = neuronios

    def forward(self, inputs):
        outputs = [neuronio.forward(inputs) for neuronio in self.neuronios]
        return outputs

    def setWeights(self, weights):
        for i, neuronio in enumerate(self.neuronios):
            neuronio.setWeights(weights[i])


# Classe NeuralNetwork
class NeuralNetwork:
    def __init__(self, state):
        self.state = state
        self.weights = state
        self.bias = 1

        # Camada 1: 7 entradas -> 14 neurônios
        self.camada1 = CamadaNeuronio(
            [
                Neuronio(self.weights[:7], self.bias, relu),
                Neuronio(self.weights[7:14], self.bias, relu),
                Neuronio(self.weights[14:21], self.bias, relu),
                Neuronio(self.weights[21:28], self.bias, relu),
                Neuronio(self.weights[28:35], self.bias, relu),
                Neuronio(self.weights[35:42], self.bias, relu),
                Neuronio(self.weights[42:49], self.bias, relu),
                Neuronio(self.weights[49:56], self.bias, relu),
                Neuronio(self.weights[56:63], self.bias, relu),
                Neuronio(self.weights[63:70], self.bias, relu),
                Neuronio(self.weights[70:77], self.bias, relu),
                Neuronio(self.weights[77:84], self.bias, relu),
                Neuronio(self.weights[84:91], self.bias, relu),
                Neuronio(self.weights[91:98], self.bias, relu),
            ]
        )

        # Camada 2: 14 entradas -> 4 neurônios
        self.camada2 = CamadaNeuronio(
            [
                Neuronio(self.weights[98:112], self.bias, relu),
                Neuronio(self.weights[112:126], self.bias, relu),
                Neuronio(self.weights[126:140], self.bias, relu),
                Neuronio(self.weights[140:154], self.bias, relu),
            ]
        )

        # Camada 3: 4 entradas -> 1 neurônio
        self.camada3 = CamadaNeuronio(
            [Neuronio(self.weights[154:158], self.bias, sigmoid)]
        )

    def keySelector(
        self,
        distance,
        obHeight,
        speed,
        obType,
        nextObDistance,
        nextObHeight,
        nextObType,
    ):
        inputs = [
            distance,
            obHeight,
            speed,
            obType,
            nextObDistance,
            nextObHeight,
            nextObType,
        ]

        inputs = fixInput(inputs)
        res1 = self.camada1.forward(inputs)
        res2 = self.camada2.forward(res1)
        res3 = self.camada3.forward(res2)
        return "K_UP" if res3[0] > 0.55 else "K_DOWN"

    # def updateState(self, state):
    #     self.state = state
    #     self.weights = state
    #     self.camada1.setWeights(
    #         [
    #             self.weights[:7],
    #             self.weights[7:14],
    #             self.weights[14:21],
    #             self.weights[21:28],
    #         ]
    #     )
    #     self.camada2.setWeights([self.weights[28:32]])

    # def setWeights(self, weights):
    #     self.weights = weights
    #     self.camada1.setWeights(
    #         [weights[:7], weights[7:14], weights[14:21], weights[21:28]]
    #     )
    #     self.camada2.setWeights([weights[28:32]])


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame(solutions):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True

    clock = pygame.time.Clock()
    cloud = Cloud()
    font = pygame.font.Font("freesansbold.ttf", 20)

    players = []
    players_classifier = []
    solution_fitness = []
    died = []

    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0

    obstacles = []
    death_count = 0
    spawn_dist = 0

    for solution in solutions:
        players.append(Dinosaur())
        players_classifier.append(NeuralNetwork(solution))
        solution_fitness.append(0)
        died.append(False)

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        if RENDER_GAME:
            text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (1000, 40)
            SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    def statistics():
        text_1 = font.render(
            f"Dinosaurs Alive:  {str(died.count(False))}", True, (0, 0, 0)
        )
        text_3 = font.render(f"Game Speed:  {str(game_speed)}", True, (0, 0, 0))

        SCREEN.blit(text_1, (50, 450))
        SCREEN.blit(text_3, (50, 480))

    while run and (False in died):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        if RENDER_GAME:
            SCREEN.fill((255, 255, 255))

        for i, player in enumerate(players):
            if not died[i]:
                distance = 1500
                nextObDistance = 2000
                obHeight = 0
                nextObHeight = 0
                obType = 2
                nextObType = 2
                if len(obstacles) != 0:
                    xy = obstacles[0].getXY()
                    distance = xy[0]
                    obHeight = obstacles[0].getHeight()
                    obType = obstacles[0]

                if len(obstacles) == 2:
                    nextxy = obstacles[1].getXY()
                    nextObDistance = nextxy[0]
                    nextObHeight = obstacles[1].getHeight()
                    nextObType = obstacles[1]

                userInput = players_classifier[i].keySelector(
                    distance,
                    obHeight,
                    game_speed,
                    obType,
                    nextObDistance,
                    nextObHeight,
                    nextObType,
                )

                player.update(userInput)

                if RENDER_GAME:
                    player.draw(SCREEN)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        for obstacle in list(obstacles):
            obstacle.update()
            if RENDER_GAME:
                obstacle.draw(SCREEN)

        if RENDER_GAME:
            background()
            statistics()
            cloud.draw(SCREEN)

        cloud.update()

        score()

        if RENDER_GAME:
            clock.tick(60)
            pygame.display.update()

        for obstacle in obstacles:
            for i, player in enumerate(players):
                if player.dino_rect.colliderect(obstacle.rect) and died[i] == False:
                    solution_fitness[i] = points
                    died[i] = True

    return solution_fitness


import numpy as np


def gerarPopulacao(tamPopulacao):
    populacao = []
    for _ in range(tamPopulacao):
        individuo = (
            ## Neuronio 1
            np.random.uniform(-0.8, 0.8, 159).tolist()
        )
        populacao.append(individuo)
    return populacao


# def gerarPopulacao(tamPopulacao):
#     populacao = []
#     for _ in range(tamPopulacao):
#         individuo = (
#             np.random.uniform(-0.5, 0.5, 4).tolist()
#             + np.random.uniform(-0.15, 0.15, 3).tolist()
#             + np.random.uniform(-0.5, 0.5, 4).tolist()
#             + np.random.uniform(-0.15, 0.15, 3).tolist()
#             + np.random.uniform(-0.5, 0.5, 4).tolist()
#             + np.random.uniform(-0.15, 0.15, 3).tolist()
#             + np.random.uniform(-0.5, 0.5, 4).tolist()
#             + np.random.uniform(-0.15, 0.15, 3).tolist()
#             + np.random.uniform(-0.5, 0.5, 4).tolist()
#         )
#         populacao.append(individuo)
#     return populacao


# def gerarPopulacao(tamPopulacao):
#     populacao = []
#     for _ in range(tamPopulacao):
#         individuo = np.random.uniform(-1, 1, 32).tolist()
#         populacao.append(individuo)
#     return populacao


# def gerarPopulacao(tamPopulacao, num_genes=32, intervalo=(-100, 100), perturbacao=0.1):
#     """
#     Gera uma população inicial diversificada.

#     Args:
#     tamPopulacao (int): Tamanho da população.
#     num_genes (int): Número de genes por indivíduo.
#     intervalo (tuple): Intervalo (min, max) para os valores iniciais.
#     perturbacao (float): Fator de perturbação para aumentar a diversidade.

#     Returns:
#     list: População inicial.
#     """
#     populacao = [np.random.uniform(intervalo[0], intervalo[1], num_genes).tolist() for _ in range(tamPopulacao)]

#     for i in range(tamPopulacao):
#         perturbacao_aleatoria = np.random.uniform(-perturbacao, perturbacao, num_genes)
#         populacao[i] = (np.array(populacao[i]) + perturbacao_aleatoria).tolist()

#     return populacao


def crossover(individuo1, individuo2, taxaCrossOver=0.6):
    filho = []
    for i in range(len(individuo1)):
        if random.random() < taxaCrossOver:
            filho.append(individuo1[i])
        else:
            filho.append(individuo2[i])
    return filho


def single_point_crossover(individuo1, individuo2):
    ponto = random.randint(0, len(individuo1) - 1)
    filho = individuo1[:ponto] + individuo2[ponto:]
    return filho


def two_point_crossover(individuo1, individuo2):
    ponto1 = random.randint(0, len(individuo1) - 1)
    ponto2 = random.randint(ponto1, len(individuo1) - 1)
    filho1 = individuo1[:ponto1] + individuo2[ponto1:ponto2] + individuo1[ponto2:]
    filho2 = individuo2[:ponto1] + individuo1[ponto1:ponto2] + individuo2[ponto2:]
    return (filho1, filho2)


def custom_crossover(individuo1, individuo2):
    # Na minha visão faz sentido fazer um multipoint que tem n pontos de corte
    # E quero testar essa possibilidade
    filho1 = []
    filho2 = []
    for i in range(len(individuo1)):
        if i % 2 == 0:
            filho1.append(individuo1[i])
            filho2.append(individuo2[i])

        else:
            filho1.append(individuo2[i])
            filho2.append(individuo1[i])

    return (filho1, filho2)


def mutacao(individuo):
    i = random.randint(0, len(individuo) - 1)
    # Aqui eu nao estou descartando a possibilidade de valor acima de 0.5 porque
    # caso esse valor seja ruim, sera descartado na prox iteracao provavelmente
    individuo[i] = np.random.uniform(-1, 1)
    return individuo


def elitismo(populacao, fitness, numElitismo):
    """
    Seleciona os melhores indivíduos da população baseado no fitness.

    Args:
    populacao (list): Lista de indivíduos na população.
    fitness (list): Lista de valores de fitness correspondentes aos indivíduos.
    numElitismo (int): Número de indivíduos a serem selecionados pelo elitismo.

    Returns:
    list: Lista dos melhores indivíduos.
    """

    populacao = [
        x
        for _, x in sorted(
            zip(fitness, populacao), key=lambda pair: pair[0], reverse=True
        )
    ]
    return populacao[:numElitismo]


def selecao(populacao, fitness):
    soma = sum(fitness)
    prob = [f / soma for f in fitness]
    pai1 = random.choices(populacao, prob)[0]
    pai2 = random.choices(populacao, prob)[0]
    return pai1, pai2


def torneio_selecao(populacao, fitness, num_selecionados, tamanho_torneio=4):
    """
    Realiza a seleção por torneio.

    Args:
    populacao (list): Lista de indivíduos na população.
    fitness (function): Função que calcula o fitness de um indivíduo.
    tamanho_torneio (int): Quantidade de indivíduos em cada torneio. (Default: 3)

    Returns:
    list: Lista de indivíduos selecionados.
    """
    selecionados = []
    for _ in range(num_selecionados):
        torneio = random.sample(list(zip(populacao, fitness)), tamanho_torneio)
        melhor = max(torneio, key=lambda x: x[1])[0]
        selecionados.append(melhor)

    return selecionados


# Teve alguma melhora
def rank_selecao(populacao, fitness, num_selecionados):
    """
    Realiza a seleção por rank.

    Args:
    populacao (list): Lista de indivíduos na população.
    fitness (function): Função que calcula o fitness de um indivíduo.

    Returns:
    list: Lista de indivíduos selecionados.
    """
    selecionados = []
    rank = [x for _, x in sorted(zip(fitness, populacao), key=lambda pair: pair[0])]
    for i in range(num_selecionados):
        selecionados.append(rank[i])
    return selecionados


#  Aparentemente bem pior
def random_selecao(populacao, fitness, num_selecionados):
    """
    Realiza a seleção por rank.

    Args:
    populacao (list): Lista de indivíduos na população.
    fitness (function): Função que calcula o fitness de um indivíduo.

    Returns:
    list: Lista de indivíduos selecionados.
    """
    selecionados = []
    rank = [x for _, x in sorted(zip(fitness, populacao), key=lambda pair: pair[0])]
    for i in range(num_selecionados):
        selecionados.append(random.choice(rank))
    return selecionados


# Bom também, talvez em par com o rank
def roulette_wheel_selection(populacao, fitness, num_selecionados):
    """
    Realiza a seleção por roleta.

    Args:
    populacao (list): Lista de indivíduos na população.
    fitness (function): Função que calcula o fitness de um indivíduo.

    Returns:
    list: Lista de indivíduos selecionados.
    """
    selecionados = []
    soma = sum(fitness)
    prob = [f / soma for f in fitness]
    for _ in range(num_selecionados):
        pai = random.choices(populacao, prob)[0]
        selecionados.append(pai)
    return selecionados


def stochastic_universal_sampling(populacao, fitness, num_selecionados):
    """
    Realiza a seleção universal estocástica (SUS).

    Args:
    populacao (list): Lista de indivíduos na população.
    fitness (function): Função que calcula o fitness de um indivíduo.
    num_selecionados (int): Número de indivíduos a serem selecionados.

    Returns:
    list: Lista de indivíduos selecionados.
    """
    selecionados = []
    soma_fitness = sum(fitness)
    ponto_inicial = random.uniform(0, soma_fitness / num_selecionados)
    pontos = [
        ponto_inicial + i * (soma_fitness / num_selecionados)
        for i in range(num_selecionados)
    ]

    prob_acumulada = [sum(fitness[: i + 1]) for i in range(len(fitness))]

    i = 0
    for ponto in pontos:
        while prob_acumulada[i] < ponto:
            i += 1
        selecionados.append(populacao[i])

    return selecionados


def evolucao(populacao, fitness, taxaCrossOver=0.6, taxaMutacao=0.1, taxaElitismo=0.02):
    novaPopulacao = []
    # Estudar a possibilidade de elitismo, adicionando duas vezes tentando fazer algo diferente, para que tenham mais influencia na pop grande
    novaPopulacao += elitismo(populacao, fitness, int(len(populacao) * taxaElitismo))

    tempPopulacao = torneio_selecao(
        populacao, fitness, len(populacao) - int(len(novaPopulacao)), 3
    )

    for i in range(0, len(tempPopulacao) - 1, 2):
        pai1 = tempPopulacao[i]
        pai2 = tempPopulacao[i + 1]
        if i < taxaCrossOver * len(tempPopulacao):
            filho1, filho2 = custom_crossover(pai1, pai2)
        else:
            filho1 = pai1
            filho2 = pai2
        if random.random() < taxaMutacao / 2:
            filho1 = mutacao(filho1)
        if random.random() < taxaMutacao / 2:
            filho2 = mutacao(filho2)

        novaPopulacao.append(filho1)
        novaPopulacao.append(filho2)

    # print(len(novaPopulacao), len(populacao))
    # Completar a população com indivíduos aleatórios  (Mais uma tentativa de aumentar a diversidade da população)
    while len(novaPopulacao) < len(populacao):
        novaPopulacao.append(np.random.uniform(-1, 1, 159).tolist())
    return novaPopulacao


def geneticAlgorithm(
    tamPopulacao, numGeracoes, taxaCrossOver=0.6, taxaMutacao=0.1, taxaElitismo=0.02
):
    populacao = gerarPopulacao(tamPopulacao)
    # print(populacao)

    start = time.process_time()
    time_max = 60 * 60 * 12
    end = 0
    i = 0

    while i <= numGeracoes and end - start <= time_max:

        if i < 0.25 * numGeracoes:
            taxaCrossOverI = 0.8
            taxaMutacaoI = 0.1
            taxaElitismoI = 0.01
        else:
            taxaCrossOverI = 0.7
            taxaMutacaoI = 0.05
            taxaElitismoI = 0.02

        fitness = manyPlaysResultsTrain(3, populacao)
        # print(max(fitness), mean(fitness), np.std(fitness))
        populacao = evolucao(
            populacao, fitness, taxaCrossOverI, taxaMutacaoI, taxaElitismoI
        )
        end = time.process_time()
        # print(populacao)
        melhor_resultado.append(max(fitness))
        # atualizar_grafico(i, melhor_resultado, linhas, ax)

        i += 1
    return populacao


from scipy import stats


def manyPlaysResultsTrain(rounds, solutions):
    results = []

    for round in range(rounds):
        results += [playGame(solutions)]

    npResults = np.asarray(results)

    mean_results = np.mean(npResults, axis=0) - np.std(
        npResults, axis=0
    )  # axis 0 calcula media da coluna

    # print(max(np.mean(npResults, axis=0)), max(np.std(npResults, axis=0)))
    # mean_results = np.mean(npResults, axis=0)
    return mean_results


def manyPlaysResultsTest(rounds, best_solution):
    results = []
    for round in range(rounds):
        results += [playGame([best_solution])[0]]

    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())


def saidaResults(rounds, solutions):
    results = []
    for round in range(rounds):
        results += [playGame(solutions)]

    npResults = np.asarray(results)
    return (npResults.T, npResults.mean(axis=0), npResults.std(axis=0))


import matplotlib.pyplot as plt


# Função para atualizar o gráfico
def atualizar_grafico(iteracao, melhor_resultado, linhas, eixo):
    if iteracao > len(melhor_resultado):
        iteracao = len(melhor_resultado)
    linhas.set_xdata(list(range(iteracao)))
    linhas.set_ydata(melhor_resultado)
    eixo.relim()
    eixo.autoscale_view()
    # plt.draw()
    # plt.pause(0.01)


import seaborn as sns

from scipy.stats import t
from math import sqrt
from statistics import stdev
from scipy import stats
import pandas as pd


def corrected_dependent_ttest(data1, data2, n_training_samples, n_test_samples):
    n = len(data1)
    differences = [(data1[i] - data2[i]) for i in range(n)]
    sd = stdev(differences)
    divisor = 1 / n * sum(differences)
    test_training_ratio = n_test_samples / n_training_samples
    denominator = sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, p


def main():

    prof_result = [
        1214.0,
        759.5,
        1164.25,
        977.25,
        1201.0,
        930.0,
        1427.75,
        799.5,
        1006.25,
        783.5,
        728.5,
        419.25,
        1389.5,
        730.0,
        1306.25,
        675.5,
        1359.5,
        1000.25,
        1284.5,
        1350.0,
        751.0,
        1418.75,
        1276.5,
        1645.75,
        860.0,
        745.5,
        1426.25,
        783.5,
        1149.75,
        1482.25,
    ]

    teste = [[-0.36984411076616336, 0.4517793003057553, -0.9260410211918653, -0.7582335197001739, -0.6973473381118653, -0.30227505157590207, -0.3821179220921611, -0.7908624455847462, -0.8714565307303572, -0.6654061520051763, -0.5329972920997577, 0.23291565646109813, -0.9835965988111317, 0.07112632232713789, 0.7114588736509144, 0.9281744303077968, 0.8506433776616138, 0.7850992109870008, 0.03169152037139855, -0.9386828301800045, 0.4177743531276683, 0.9530321320485509, -0.9082254432379075, -0.6820207102621932, 0.1450383612295949, -0.23028031891399303, 0.9626147486379313, -0.22338572186935102, 0.28526434603974016, 0.12920425953439452, -0.7300107169428547, 0.5544474069705672, 0.626945927772671, -0.16557360243073171, 0.4917431964304346, 0.4886165541138767, 0.12241709748513596, 0.4791942519522083, -0.18452940495653514, 0.09284212223477761, -0.9023864299623447, 0.6452963744298723, -0.0687543292184929, -0.05563827142279121, -0.769319683551016, 0.49764404762456715, 0.407769791958023, -0.8733397105022616, -0.6334203870155521, -0.518888507136916, -0.39753934298638827, 0.9496408993947798, -0.36453715149932775, -0.13305613718892229, 0.20300781618280528, 0.5795298270773706, -0.19124733729015597, 0.7605119852432676, -0.717955657108579, -0.7351842480362285, 0.7406407026518567, 0.42046025207880056, -0.7131201559647093, 0.05336353713449404, -0.8870841160819902, -0.9167327638935419, 0.5089153760931855, -0.6479328919685712, -0.4067741652026149, -0.6192473448379183, -0.07169063202103665, 0.42258425917706477, -0.023125566499857975, 0.23354817960767038, -0.400925538513482, 0.2772911573549417, 0.13700186425260696, -0.7261674257571992, -0.1948664562178748, -0.7559394257513152, -0.34202088660052343, 0.43197965900176594, 0.4383629831048239, 0.9893248762425402, 0.13780125337396976, 0.6257864663928447, -0.9003310556611563, 0.5455935831098253, -0.3796759695091275, -0.21286386787011335, -0.405798060193413, -0.6307625304372357, 0.5567127153677955, -0.21599441247528683, 0.9407988523134352, -0.3410663574637887, -0.015359557655066247, 0.8569141838800673, -0.6669048413979495, -0.4945379864709956, 0.1978978116951835, -0.10555646372589078, 0.742010246097464, -0.09360071395803127, 0.16229447849757805, 0.7118463567243167, -0.3083028410315118, -0.3981806556433101, 0.6770866321578237, -0.48877067224028664, -0.0963608859738736, 0.6584457781190307, 0.05430453021942916, -0.9315517853797182, 0.572920282658625, 0.29349522532617867, -0.6980534578510682, 0.48730508614251633, -0.2730148419309446, -0.6777930269194826, -0.13931687835519901, 0.758489756733213, -0.018066648858588885, 0.4345212019773592, -0.630725711982278, -0.24271081273645234, 0.30963225837088926, -0.5753269717980013, -0.43789467307358304, -0.1000367301130185, -0.7339266312474271, -0.7593680202292221, 0.7547427619524307, 0.5448650337178178, 0.7626347517788028, -0.16267837322087675, -0.1695255852898645, 0.4333208019389718, 0.7987956837620065, -0.19143325766531682, -0.6471761753532901, -0.34580934103818867, -0.4727918384829902, 0.43382184469053, 0.5096354855802145, -0.357665244937494, 0.1802684951875475, -0.04192205365676871, -0.11762754842176704, 0.30524113335973313, -0.6137873195528833, 0.5563064065054466, -0.4562732653157733, -0.44980164586412336, -0.776811539807067, -0.21489730572396826, 0.46797184510302614, -0.10172868829329484, 0.09559929973484604]]

    res = saidaResults(30, teste)
    
    print(res[0][0])

    data = [res[0][0], prof_result]

    testes = [[0 for i in range(2)] for j in range(2)]

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            testes[i][j] = stats.ttest_ind(data[i], data[j])[1]

    for i in range(len(data)):
        for j in range(i):
            testes[i][j] = stats.wilcoxon(data[i], data[j])[1]

    nomes = ["Aluno", "Prof"]

    for i in range(len(nomes)):
        testes[i][i] = nomes[i]

    print(testes)
    testes = pd.DataFrame(testes)
    print(testes)
    testes.to_latex("tabela_testes - Mod.tex", header=False, index=False)
    
    plt.clf()
    boxplots = sns.boxplot(data)
    boxplots.set_xticklabels(["Aluno", "Professor"])
    # plt.show()
    fig.savefig("boxplot_xprof - Mod.png")
    print("Gráfico salvo como 'boxplot_xprof.png'")
    plt.clf()


# Inicializar a figura e o eixo do gráfico
fig, ax = plt.subplots()
(linhas,) = ax.plot([], [], "b-")  # 'b-' é a cor e o estilo da linha (azul e sólida)
ax.set_xlabel("Iteração")
ax.set_ylabel("Melhor Resultado")
ax.set_title("Melhor Resultado por Iteração")
melhor_resultado = []

main()
