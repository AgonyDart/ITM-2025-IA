import pygame
import math
from queue import PriorityQueue

# Configuraciones iniciales
ANCHO_VENTANA = 1400
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Algoritmo A* ")

# Paleta de diseñador (inicio = verde, paredes = negro, otros colores profesionales)
BLANCO = (250, 250, 250)  # fondo muy claro
NEGRO = (40, 40, 40)  # paredes / obstáculo
GRIS = (220, 220, 220)  # líneas de la cuadrícula
GRIS_CLARO = (245, 245, 245)  # estado por defecto de los nodos

INICIO_VERDE = (46, 204, 113)  # verde esmeralda para el nodo inicio
FIN_AZUL = (52, 152, 219)  # azul profesional para el nodo fin
FRONTERA = (241, 196, 15)  # amarillo cálido para la frontera (open set)
VISITADO = (155, 89, 182)  # púrpura suave para nodos visitados
CAMINO_COLOR = (26, 188, 156)  # teal para el camino final
ID_COLOR = (0, 0, 0)  # color del texto del ID

# Fuente para los IDs
pygame.font.init()
FUENTE = pygame.font.SysFont("arial", 14)


class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = GRIS_CLARO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []
        self.id = f"{fila}{col}"  # ID único para cada nodo

    def get_pos(self):
        return self.fila, self.col

    def get_id(self):
        return self.id

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == INICIO_VERDE

    def es_fin(self):
        return self.color == FIN_AZUL

    def restablecer(self):
        self.color = GRIS_CLARO

    def hacer_visitado(self):
        self.color = VISITADO

    def hacer_frontera(self):
        self.color = FRONTERA

    def hacer_camino(self):
        self.color = CAMINO_COLOR

    def hacer_inicio(self):
        self.color = INICIO_VERDE

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = FIN_AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

        # Dibujar el ID del nodo
        texto = FUENTE.render(self.id, True, ID_COLOR)
        ventana.blit(texto, (self.x + 3, self.y + 3))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        # Abajo
        if (
            self.fila < self.total_filas - 1
            and not grid[self.fila + 1][self.col].es_pared()
        ):
            self.vecinos.append(grid[self.fila + 1][self.col])
        # Arriba
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])
        # Derecha
        if (
            self.col < self.total_filas - 1
            and not grid[self.fila][self.col + 1].es_pared()
        ):
            self.vecinos.append(grid[self.fila][self.col + 1])
        # Izquierda
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])


def heuristica(p1, p2):
    # Distancia Manhattan
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def algoritmo_a_estella(dibujar, grid, inicio, fin):
    contador = 0
    frontera = PriorityQueue()
    frontera.put((0, contador, inicio))
    procedencia = {}
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())

    frontera_hash = {inicio}

    while not frontera.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        nodo_actual = frontera.get()[2]
        frontera_hash.remove(nodo_actual)

        if nodo_actual == fin:
            camino = reconstruir_camino(procedencia, fin, inicio, dibujar)
            fin.hacer_fin()
            inicio.hacer_inicio()

            # Imprimir el camino en consola
            print("\n" + "=" * 50)
            print("CAMINO MÁS CORTO ENCONTRADO:")
            print("=" * 50)
            ids_camino = [nodo.get_id() for nodo in camino]
            print(" -> ".join(ids_camino))
            print(f"Longitud del camino: {len(camino)} nodos")
            print("=" * 50 + "\n")

            return True

        for vecino in nodo_actual.vecinos:
            g_temp = g_score[nodo_actual] + 1

            if g_temp < g_score[vecino]:
                procedencia[vecino] = nodo_actual
                g_score[vecino] = g_temp
                f_score[vecino] = g_temp + heuristica(vecino.get_pos(), fin.get_pos())
                if vecino not in frontera_hash:
                    contador += 1
                    frontera.put((f_score[vecino], contador, vecino))
                    frontera_hash.add(vecino)
                    vecino.hacer_frontera()

        dibujar()

        if nodo_actual != inicio:
            nodo_actual.hacer_visitado()

    print("\n❌ No se encontró camino posible entre inicio y fin")
    return False


def reconstruir_camino(procedencia, nodo_actual, inicio, dibujar):
    camino = []
    current = nodo_actual

    # Reconstruir el camino desde el fin hasta el inicio
    while current in procedencia:
        camino.append(current)
        current = procedencia[current]
    camino.append(inicio)  # Agregar el nodo inicio
    camino.reverse()  # Invertir para tener inicio -> fin

    # Dibujar el camino (excluyendo inicio y fin)
    for nodo in camino[1:-1]:  # Excluir el primer (inicio) y último (fin) nodo
        nodo.hacer_camino()
        dibujar()
        pygame.time.delay(30)  # Pequeña pausa para visualización

    return camino


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


def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()


def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col


def main(ventana, ancho):
    FILAS = 8
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None
    ejecutado = False

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if ejecutado:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)
                    ejecutado = False
                continue

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                # Verificar que las coordenadas estén dentro del grid
                if 0 <= fila < FILAS and 0 <= col < FILAS:
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                        print(f"📍 Nodo inicio establecido en: {inicio.get_id()}")

                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                        print(f"🎯 Nodo fin establecido en: {fin.get_id()}")

                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()
                        print(f"🧱 Pared colocada en: {nodo.get_id()}")

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                # Verificar que las coordenadas estén dentro del grid
                if 0 <= fila < FILAS and 0 <= col < FILAS:
                    nodo = grid[fila][col]
                    nodo.restablecer()
                    if nodo == inicio:
                        print(f"❌ Nodo inicio removido de: {inicio.get_id()}")
                        inicio = None
                    elif nodo == fin:
                        print(f"❌ Nodo fin removido de: {fin.get_id()}")
                        fin = None
                    else:
                        print(f"🧹 Nodo limpiado en: {nodo.get_id()}")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    print(f"\n🚀 Iniciando búsqueda del camino más corto...")
                    print(f"Desde: {inicio.get_id()} → Hasta: {fin.get_id()}")

                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    algoritmo_a_estella(
                        lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin
                    )
                    ejecutado = True

                if event.key == pygame.K_c:
                    print("\n🧹 Grid limpiado - Comenzando de nuevo")
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)
                    ejecutado = False

    pygame.quit()


if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZADOR DE ALGORITMO A* - ENCONTRADOR DE CAMINOS")
    print("=" * 60)
    print("INSTRUCCIONES:")
    print("• Click IZQUIERDO: Colocar nodos (inicio → fin → paredes)")
    print("• Click DERECHO: Eliminar nodos")
    print("• ESPACIO: Ejecutar algoritmo A*")
    print("• C: Limpiar grid y comenzar de nuevo")
    print("• Cada casilla muestra su ID (fila-columna)")
    print("=" * 60)
    main(VENTANA, ANCHO_VENTANA)
