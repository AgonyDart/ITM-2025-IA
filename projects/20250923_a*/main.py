import pygame
import heapq
from math import sqrt
import random

ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("A*")
pygame.font.init()
FUENTE_PEQUEÑA = pygame.font.SysFont("Arial", 20)

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
ROJO = (255, 0, 0)
VERDE = (0, 255, 0)
AZUL = (0, 120, 255)
CELESTE = (173, 216, 230)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AMARILLO = (255, 255, 0)


class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.vecinos = []

        self.total_filas = total_filas
        self.g = float("inf")
        self.h = 0
        self.f = float("inf")
        self.padre = None
        self.visitado = False
        self.procesado = False

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def es_abierto(self):
        return self.color == AMARILLO

    def es_cerrado(self):
        return self.color == PURPURA

    def es_camino(self):
        return self.color == VERDE

    def restablecer(self):
        self.color = BLANCO
        self.g = float("inf")
        self.h = 0
        self.f = float("inf")
        self.padre = None

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = CELESTE

    def hacer_cerrado(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = ROJO

    def hacer_camino(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = VERDE

    def hacer_visitado(self):
        if (
            not self.es_inicio()
            and not self.es_fin()
            and not self.es_abierto()
            and not self.es_cerrado()
            and not self.es_camino()
        ):
            self.color = AMARILLO

    def hacer_procesado(self):
        if not self.es_inicio() and not self.es_fin() and not self.es_cerrado():
            self.color = AZUL

    def actualizar_valores(self, g, h, f):
        self.g = g
        self.h = h
        self.f = f

    def dibujar(self, ventana, mostrar_valores=True):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

        pygame.draw.rect(ventana, GRIS, (self.x, self.y, self.ancho, self.ancho), 1)

        if (
            mostrar_valores
            and not self.es_inicio()
            and not self.es_fin()
            and not self.es_pared()
        ):
            if self.f != float("inf"):
                texto_f = FUENTE_PEQUEÑA.render(f"f:{self.f:.0f}", True, NEGRO)
                ventana.blit(texto_f, (self.x + 2, self.y + 2))
            if self.g != float("inf"):
                texto_g = FUENTE_PEQUEÑA.render(f"g:{self.g:.0f}", True, NEGRO)
                ventana.blit(texto_g, (self.x + 2, self.y + 12))
            if self.h > 0:
                texto_h = FUENTE_PEQUEÑA.render(f"h:{self.h:.0f}", True, NEGRO)
                ventana.blit(texto_h, (self.x + 2, self.y + 22))

    def __lt__(self, otro):
        return self.f < otro.f


def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid


def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(
                ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho)
            )


def dibujar(ventana, grid, filas, ancho, mostrar_valores=True, velocidad=50):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana, mostrar_valores)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()


def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col


def get_vecinos(nodo, grid):
    vecinos = []
    fila, col = nodo.get_pos()
    direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for df, dc in direcciones:
        nueva_fila, nueva_col = fila + df, col + dc
        if 0 <= nueva_fila < len(grid) and 0 <= nueva_col < len(grid[0]):
            vecinos.append(grid[nueva_fila][nueva_col])

    return vecinos


def heuristica(nodo1, nodo2):
    f1, c1 = nodo1.get_pos()
    f2, c2 = nodo2.get_pos()
    return sqrt((f1 - f2) ** 2 + (c1 - c2) ** 2)


def distancia(nodo1, nodo2):
    f1, c1 = nodo1.get_pos()
    f2, c2 = nodo2.get_pos()
    if (f1 != f2) and (c1 != c2):
        return sqrt(2)
    return 1


def a_star(grid, inicio, fin, ventana, ancho, filas, velocidad, mostrar_valores=True):
    abiertos = []
    cerrados = set()

    inicio.g = 0
    inicio.h = heuristica(inicio, fin)
    inicio.f = inicio.g + inicio.h

    heapq.heappush(abiertos, inicio)

    paso = 0
    algoritmo_terminado = False

    while abiertos and not algoritmo_terminado:
        paso += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        nodo_actual = heapq.heappop(abiertos)
        cerrados.add(nodo_actual)

        if not nodo_actual.es_inicio() and not nodo_actual.es_fin():
            nodo_actual.hacer_cerrado()

        dibujar(
            ventana,
            grid,
            filas,
            ancho,
            mostrar_valores,
            velocidad,
        )
        pygame.time.delay(velocidad)

        if nodo_actual == fin:
            camino = []
            while nodo_actual:
                camino.append(nodo_actual)
                nodo_actual = nodo_actual.padre

            for i, nodo in enumerate(camino):
                if not nodo.es_inicio() and not nodo.es_fin():
                    nodo.hacer_camino()
                dibujar(
                    ventana,
                    grid,
                    filas,
                    ancho,
                    mostrar_valores,
                    velocidad,
                )
                pygame.time.delay(velocidad // 2)

            return True

        vecinos = get_vecinos(nodo_actual, grid)

        for i, vecino in enumerate(vecinos):
            if (
                not vecino.es_inicio()
                and not vecino.es_fin()
                and not vecino.es_pared()
                and vecino not in cerrados
            ):
                vecino.hacer_visitado()

            dibujar(
                ventana,
                grid,
                filas,
                ancho,
                mostrar_valores,
                velocidad,
            )
            pygame.time.delay(velocidad // 2)

            if vecino.es_pared() or vecino in cerrados:
                continue

            g_tentativo = nodo_actual.g + distancia(nodo_actual, vecino)

            if g_tentativo < vecino.g:
                vecino.padre = nodo_actual
                vecino.g = g_tentativo
                vecino.h = heuristica(vecino, fin)
                vecino.f = vecino.g + vecino.h

                if vecino not in abiertos:
                    heapq.heappush(abiertos, vecino)
                    if not vecino.es_fin():
                        vecino.hacer_abierto()

            dibujar(
                ventana,
                grid,
                filas,
                ancho,
                mostrar_valores,
                velocidad,
            )
            pygame.time.delay(velocidad // 2)

    dibujar(ventana, grid, filas, ancho, mostrar_valores, velocidad)
    pygame.time.delay(1000)
    return False


def generar_mapa_random(grid, filas, densidad=0.3):
    for fila in grid:
        for nodo in fila:
            nodo.restablecer()

    for i in range(filas):
        for j in range(filas):
            if random.random() < densidad:
                grid[i][j].hacer_pared()


def resetear_valores(grid):
    for fila in grid:
        for nodo in fila:
            if not nodo.es_pared() and not nodo.es_inicio() and not nodo.es_fin():
                nodo.restablecer()
            else:
                nodo.g = float("inf")
                nodo.h = 0
                nodo.f = float("inf")
                nodo.padre = None
                nodo.visitado = False
                nodo.procesado = False


def main(ventana, ancho):
    FILAS = 10
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None
    algoritmo_corriendo = False

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if inicio and fin:
                        resetear_valores(grid)
                        algoritmo_corriendo = True
                        if a_star(
                            grid, inicio, fin, ventana, ancho, FILAS, velocidad=50
                        ):
                            algoritmo_corriendo = False
                        else:
                            corriendo = False

                elif event.key == pygame.K_q:
                    corriendo = False

                elif event.key == pygame.K_r:
                    inicio = None
                    fin = None
                    generar_mapa_random(grid, FILAS, 0.25)

            if not algoritmo_corriendo:
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    fila, col = obtener_click_pos(pos, FILAS, ancho)
                    if 0 <= fila < FILAS and 0 <= col < FILAS:
                        nodo = grid[fila][col]
                        if not inicio and nodo != fin:
                            inicio = nodo
                            inicio.hacer_inicio()

                        elif not fin and nodo != inicio:
                            fin = nodo
                            fin.hacer_fin()

                        elif nodo != fin and nodo != inicio:
                            nodo.hacer_pared()

                elif pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    fila, col = obtener_click_pos(pos, FILAS, ancho)
                    if 0 <= fila < FILAS and 0 <= col < FILAS:
                        nodo = grid[fila][col]
                        nodo.restablecer()
                        if nodo == inicio:
                            inicio = None
                        elif nodo == fin:
                            fin = None

    pygame.quit()


main(VENTANA, ANCHO_VENTANA)
