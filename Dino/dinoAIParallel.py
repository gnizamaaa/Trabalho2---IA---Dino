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

def sigmoid(x):
    """
    Função sigmoide numericamente estável.

    Args:
    x (numpy array): Entrada para a função sigmoide.

    Returns:
    numpy array: Saída da função sigmoide.
    """
    # Usar a versão numericamente estável da função sigmoide
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def relu(x):
    return np.maximum(0, x)


from sklearn.preprocessing import normalize


def fixInput(inputs):

    classes = {
        Bird: 1,
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

    # inputs = np.array(inputs) / (np.sum(inputs) + 1e-9)
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
        self.bias = 0

        self.camada1 = CamadaNeuronio(
            [
                Neuronio(self.weights[:7], self.bias, relu),
                Neuronio(self.weights[7:14], self.bias, relu),
                Neuronio(self.weights[14:21], self.bias, relu),
                Neuronio(self.weights[21:28], self.bias, relu),
            ]
        )
        self.camada2 = CamadaNeuronio(
            [Neuronio(self.weights[28:32], self.bias, sigmoid)]
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
        return "K_UP" if res2[0] > 0.55 else "K_DOWN"

    def updateState(self, state):
        self.state = state
        self.weights = state
        self.camada1.setWeights(
            [
                self.weights[:7],
                self.weights[7:14],
                self.weights[14:21],
                self.weights[21:28],
            ]
        )
        self.camada2.setWeights([self.weights[28:32]])

    def setWeights(self, weights):
        self.weights = weights
        self.camada1.setWeights(
            [weights[:7], weights[7:14], weights[14:21], weights[21:28]]
        )
        self.camada2.setWeights([weights[28:32]])


class KeySimplestClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state

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
        self.state = sorted(self.state, key=first)
        for s, d in self.state:
            if speed < s:
                limDist = d
                break
        if distance <= limDist:
            if isinstance(obType, Bird) and obHeight > 50:
                return "K_DOWN"
            else:
                return "K_UP"
        return "K_NO"

    def updateState(self, state):
        self.state = state


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


# Change State Operator
def change_state(state, position, vs, vd):
    aux = state.copy()
    s, d = state[position]
    ns = s + vs
    nd = d + vd
    if ns < 15 or nd > 1000:
        return []
    return aux[:position] + [(ns, nd)] + aux[position + 1 :]


# Neighborhood
def generate_neighborhood(state):
    neighborhood = []
    state_size = len(state)
    for i in range(state_size):
        ds = random.randint(1, 10)
        dd = random.randint(1, 100)
        new_states = [
            change_state(state, i, ds, 0),
            change_state(state, i, (-ds), 0),
            change_state(state, i, 0, dd),
            change_state(state, i, 0, (-dd)),
        ]
        for s in new_states:
            if s != []:
                neighborhood.append(s)
    return neighborhood


# Gradiente Ascent
def gradient_ascent(state, max_time):
    start = time.process_time()
    res, max_value = manyPlaysResultsTest(3, state)
    better = True
    end = 0
    while better and end - start <= max_time:
        neighborhood = generate_neighborhood(state)
        print(neighborhood)
        better = False

        results = playGame(neighborhood)
        for i, value in enumerate(results):
            if value > max_value:
                state = neighborhood[i]
                max_value = value
                better = True
        """
        for s in neighborhood:
            aiPlayer = KeySimplestClassifier(s)
            res, value = manyPlaysResults(3)
            if value > max_value:
                state = s
                max_value = value
                better = True
        """
        end = time.process_time()
    return state, max_value


import numpy as np


# def gerarPopulacao(tamPopulacao):
#     return [np.random.uniform(-100, 100, 32).tolist() for _ in range(tamPopulacao)]

import numpy as np

def gerarPopulacao(tamPopulacao, num_genes=32, intervalo=(-100, 100), perturbacao=0.1):
    """
    Gera uma população inicial diversificada.

    Args:
    tamPopulacao (int): Tamanho da população.
    num_genes (int): Número de genes por indivíduo.
    intervalo (tuple): Intervalo (min, max) para os valores iniciais.
    perturbacao (float): Fator de perturbação para aumentar a diversidade.

    Returns:
    list: População inicial.
    """
    populacao = [np.random.uniform(intervalo[0], intervalo[1], num_genes).tolist() for _ in range(tamPopulacao)]
    
    for i in range(tamPopulacao):
        perturbacao_aleatoria = np.random.uniform(-perturbacao, perturbacao, num_genes)
        populacao[i] = (np.array(populacao[i]) + perturbacao_aleatoria).tolist()
    
    return populacao


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
    ponto1 = random.randint(0, len(individuo1))
    ponto2 = random.randint(ponto1, len(individuo1))
    filho = individuo1[:ponto1] + individuo2[ponto1:ponto2] + individuo1[ponto2:]
    return filho


def mutacao(individuo, taxaMutacao):
    for i in range(len(individuo)):
        if random.random() < taxaMutacao:
            # individuo[i] = random.random()
            individuo[i] = np.random.uniform(-100, 100)
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

def evolucao(populacao, fitness, taxaCrossOver=0.9, taxaMutacao=0.1, taxaElitismo=0.02):
    novaPopulacao = []
    # Estudar a possibilidade de elitismo
    novaPopulacao += elitismo(populacao, fitness, int(len(populacao) * taxaElitismo))

    # 95 % da nova população será composta por indivíduos resultantes do crossover e mutação
    while len(novaPopulacao) < int(len(populacao) * 0.96):
        # Selecionar os mais aptos
        # pai1, pai2 = torneio_selecao(populacao, fitness, 2)
        # pai1, pai2 = rank_selecao(populacao, fitness, 2)
        # pai1, pai2 = random_selecao(populacao, fitness, 2)
        # pai1, pai2 = roulette_wheel_selection(populacao, fitness, 2)
        pai1, pai2 = stochastic_universal_sampling(populacao, fitness, 2)

        # Teste de diferentes crossover
        # filho = crossover(pai1, pai2, taxaCrossOver)
        # filho = single_point_crossover(pai1, pai2)
        if random.random() < taxaCrossOver:
            # filho = two_point_crossover(pai1, pai2)
            filho = single_point_crossover(pai1, pai2)
        else:
            filho = pai1
        filho = mutacao(filho, taxaMutacao)

        novaPopulacao.append(filho)

    # Completar a população com indivíduos aleatórios  (Mais uma tentativa de aumentar a diversidade da população)
    while len(novaPopulacao) < len(populacao):
        novaPopulacao.append(np.random.uniform(-100, 100, 32).tolist())
    return novaPopulacao


def geneticAlgorithm(
    tamPopulacao, numGeracoes, taxaCrossOver, taxaMutacao, taxaElitismo
):
    populacao = gerarPopulacao(tamPopulacao)
    # print(populacao)
    for i in range(numGeracoes):
        fitness = manyPlaysResultsTrain(50, populacao)
        print(max(fitness), mean(fitness), np.std(fitness))
        populacao = evolucao(
            populacao, fitness, taxaCrossOver, taxaMutacao, taxaElitismo
        )
        # print(populacao)
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

    #print(max(np.mean(npResults, axis=0)), max(np.std(npResults, axis=0)))
    # mean_results = np.mean(npResults, axis=0)
    return mean_results


def manyPlaysResultsTest(rounds, best_solution):
    results = []
    for round in range(rounds):
        results += [playGame([best_solution])[0]]

    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())


def main():

    teste = geneticAlgorithm(100, 200, 0.9, 0.05, 0.02)
    print(teste)

    print(playGame(teste))
    # print(playGame([[random.random() for i in range(32)]]))
    # initial_state = [(15, 250), (18, 350), (20, 450), (1000, 550)]
    # best_state, best_value = gradient_ascent(initial_state, 5000)

    # res, value = manyPlaysResultsTest(30, best_state)
    # npRes = np.asarray(res)
    # print(res, npRes.mean(), npRes.std(), value)


main()
