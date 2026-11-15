"""
Baneplanlegging med distansetransform på et rutenett.

Distansetransformen gir hver fri celle reisekostnaden til et kildepunkt. En bane finnes
ved å starte i en celle og hele tiden gå til den nabocellen som er nærmest kilden. Én
transform gir altså avstanden fra alle celler til punktet på en gang.

Bruk::

    grid = GridMap.from_occupancy_grid(map_msg, cell_size=0.1)
    planner = Planner(grid)

    path = planner.path((robot_x, robot_y), (goal_x, goal_y))    # [(x, y), ...] eller None

    field = planner.field(robot_x, robot_y)                      # én transform ...
    costs = [field.cost_at(x, y) for (x, y) in search_points]    # ... mange oppslag

Celler er (col, row). col går langs x og row langs y. Flat indeks er row * width + col,
som i nav_msgs/OccupancyGrid.
"""
from collections import deque
import heapq
import math

INF = math.inf
SQRT2 = math.sqrt(2.0)

# (d_col, d_row, steglengde i celler)
STRAIGHT = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0))
DIAGONAL = ((1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))
NEIGHBOURS = STRAIGHT + DIAGONAL


class GridMap:
    """Rutenett med opptatte og frie celler, og omregning til og fra verdenskoordinater."""

    def __init__(self, width, height, resolution, origin_x, origin_y, occupied):
        if len(occupied) != width * height:
            raise ValueError('occupied must have width * height entries')
        self.width = width
        self.height = height
        self.resolution = resolution
        # Verdensposisjonen til nedre venstre hjørne av celle (0, 0)
        self.origin_x = origin_x
        self.origin_y = origin_y
        # 1 for vegg eller ukjent, 0 for fri
        self.occupied = bytearray(occupied)

    @classmethod
    def from_data(cls, data, width, height, resolution, origin_x=0.0, origin_y=0.0,
                  cell_size=None, occupied_threshold=50, unknown_is_occupied=True):
        """
        Lag et rutenett fra belegningsverdier (0 til 100, -1 for ukjent), rad for rad.

        Med cell_size større enn resolution slås blokker av celler sammen. En samlet celle
        er opptatt hvis en av cellene i den er det, så vegger blir aldri tynnere. Kartene
        i kurset har celler på 1 cm, som er alt for mange å planlegge på.
        """
        if len(data) != width * height:
            raise ValueError('data must have width * height entries')

        factor = 1
        if cell_size is not None:
            factor = max(1, int(round(cell_size / resolution)))

        new_width = int(math.ceil(width / factor))
        new_height = int(math.ceil(height / factor))
        occupied = bytearray(new_width * new_height)

        for row in range(height):
            line = data[row * width:(row + 1) * width]
            base = (row // factor) * new_width
            for new_col in range(new_width):
                if occupied[base + new_col]:
                    continue
                block = line[new_col * factor:(new_col + 1) * factor]
                if max(block) >= occupied_threshold:
                    occupied[base + new_col] = 1
                elif unknown_is_occupied and min(block) < 0:
                    occupied[base + new_col] = 1

        return cls(new_width, new_height, resolution * factor, origin_x, origin_y, occupied)

    @classmethod
    def from_occupancy_grid(cls, msg, cell_size=None, occupied_threshold=50,
                            unknown_is_occupied=True):
        """
        Lag et rutenett fra en nav_msgs/OccupancyGrid-melding.

        Antar at kartet ikke er rotert, noe som stemmer for kartene i kurset.
        """
        return cls.from_data(
            msg.data, msg.info.width, msg.info.height, msg.info.resolution,
            msg.info.origin.position.x, msg.info.origin.position.y,
            cell_size=cell_size, occupied_threshold=occupied_threshold,
            unknown_is_occupied=unknown_is_occupied)

    def in_bounds(self, col, row):
        return 0 <= col < self.width and 0 <= row < self.height

    def index(self, col, row):
        return row * self.width + col

    def world_to_cell(self, x, y):
        """Returner (col, row) for cellen punktet ligger i. Kan være utenfor kartet."""
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return col, row

    def cell_to_world(self, col, row):
        """Returner verdensposisjonen (x, y) til midten av en celle."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y


def obstacle_distance(grid, max_distance=INF):
    """
    Returner avstanden i meter fra hver celle til nærmeste opptatte celle.

    Distansetransform med veggene som kilder. Hver celle husker hvilken veggcelle den ble
    nådd fra, og måler rett linje dit. Da blir resultatet nær den ekte avstanden, ikke en
    trappetrinn-tilnærming. Celler lenger unna enn max_distance blir stående på uendelig.
    """
    width = grid.width
    height = grid.height
    resolution = grid.resolution
    distance = [INF] * (width * height)
    heap = []

    for index, is_occupied in enumerate(grid.occupied):
        if is_occupied:
            distance[index] = 0.0
            heap.append((0.0, index, index))
    heapq.heapify(heap)

    while heap:
        dist, index, source = heapq.heappop(heap)
        if dist > distance[index]:
            continue
        col = index % width
        row = index // width
        source_col = source % width
        source_row = source // width
        for d_col, d_row, _ in NEIGHBOURS:
            n_col = col + d_col
            n_row = row + d_row
            if not (0 <= n_col < width and 0 <= n_row < height):
                continue
            n_dist = resolution * math.hypot(n_col - source_col, n_row - source_row)
            if n_dist > max_distance:
                continue
            n_index = n_row * width + n_col
            if n_dist < distance[n_index]:
                distance[n_index] = n_dist
                heapq.heappush(heap, (n_dist, n_index, source))

    return distance


def distance_transform(grid, traversable, sources, cell_cost=None):
    """
    Returner reisekostnaden fra hver celle til nærmeste kildecelle.

    grid         GridMap, for størrelse og oppløsning
    traversable  én verdi per celle, sann der robotens senter kan være
    sources      liste med (col, row)-celler som har kostnad 0
    cell_cost    valgfri faktor per celle (1.0 eller mer). Et steg koster lengden i meter
                 ganger snittet av faktorene i de to cellene. Uten den blir resultatet
                 banelengden i meter.

    Bølgefronten sprer seg fra kildene, billigste celle først, til de 8 naboene. Et
    diagonalt steg er bare lov når begge cellene det går mellom er farbare, så banen
    kutter aldri et vegghjørne. Celler som ikke kan nås, får kostnad uendelig.
    """
    width = grid.width
    height = grid.height
    resolution = grid.resolution
    cost = [INF] * (width * height)
    heap = []

    for col, row in sources:
        if not grid.in_bounds(col, row):
            continue
        index = row * width + col
        if traversable[index]:
            cost[index] = 0.0
            heap.append((0.0, index))
    heapq.heapify(heap)

    while heap:
        current, index = heapq.heappop(heap)
        if current > cost[index]:
            continue
        col = index % width
        row = index // width
        for d_col, d_row, length in NEIGHBOURS:
            n_col = col + d_col
            n_row = row + d_row
            if not (0 <= n_col < width and 0 <= n_row < height):
                continue
            n_index = n_row * width + n_col
            if not traversable[n_index]:
                continue
            if d_col != 0 and d_row != 0:
                if not (traversable[row * width + n_col] and traversable[n_row * width + col]):
                    continue
            step = length * resolution
            if cell_cost is not None:
                step *= 0.5 * (cell_cost[index] + cell_cost[n_index])
            n_cost = current + step
            if n_cost < cost[n_index]:
                cost[n_index] = n_cost
                heapq.heappush(heap, (n_cost, n_index))

    return cost


def descend(grid, traversable, cost, start, cell_cost=None):
    """
    Følg en distansetransform nedover fra start til kilden.

    Returnerer listen med (col, row)-celler fra start til kilden, eller None hvis start
    ikke når noen kilde. Bruker de samme stegreglene som distance_transform.
    """
    width = grid.width
    height = grid.height
    resolution = grid.resolution
    col, row = start
    if not grid.in_bounds(col, row) or cost[row * width + col] == INF:
        return None

    path = [(col, row)]
    while cost[row * width + col] > 0.0:
        index = row * width + col
        best = None
        best_total = INF
        for d_col, d_row, length in NEIGHBOURS:
            n_col = col + d_col
            n_row = row + d_row
            if not (0 <= n_col < width and 0 <= n_row < height):
                continue
            n_index = n_row * width + n_col
            # Bare steg til en billigere celle. Det sikrer også at løkken slutter.
            if cost[n_index] >= cost[index]:
                continue
            if d_col != 0 and d_row != 0:
                if not (traversable[row * width + n_col] and traversable[n_row * width + col]):
                    continue
            step = length * resolution
            if cell_cost is not None:
                step *= 0.5 * (cell_cost[index] + cell_cost[n_index])
            total = cost[n_index] + step
            if total < best_total:
                best_total = total
                best = (n_col, n_row)
        if best is None:
            return None
        col, row = best
        path.append(best)

    return path


def path_length(points):
    """Returner lengden i meter av en bane gitt som en liste med (x, y)-punkter."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


class DistanceField:
    """Resultatet av én distansetransform: reisekostnaden fra en kilde til hver celle."""

    def __init__(self, planner, cost, source):
        self._planner = planner
        self.cost = cost
        # Verdensposisjonen (x, y) til kildecellen
        self.source = source

    def cost_at(self, x, y):
        """
        Returner reisekostnaden mellom kilden og punktet (x, y), uendelig uten vei.

        Kostnaden er lik begge veier. Med wall_weight 0 er den banelengden i meter, ellers
        litt høyere, men fortsatt god nok til å sammenligne hvilke punkter som er nærmest.
        """
        cell = self._planner.snap(x, y)
        if cell is None:
            return INF
        return self.cost[self._planner.grid.index(*cell)]

    def path_from(self, x, y):
        """
        Returner banen fra punktet (x, y) til kilden som [(x, y), ...], eller None.

        Punktene er cellesentre, én celle fra hverandre.
        """
        planner = self._planner
        cell = planner.snap(x, y)
        if cell is None:
            return None
        cells = descend(planner.grid, planner.traversable, self.cost, cell, planner.cell_cost)
        if cells is None:
            return None
        return [planner.grid.cell_to_world(col, row) for col, row in cells]


class Planner:
    """
    Planlegger baner for en rund robot på et GridMap.

    robot_radius  Robotens senter holdes minst så langt fra hvert veggcellesenter.
                  Senteret er punktet mellom hjulene, som roboten dreier om. På en
                  TurtleBot3 Waffle er bakhjørnene ca. 0.26 m fra det, så de sveiper så
                  langt ut når roboten snur på stedet.
    soft_radius   Celler nærmere en vegg enn dette koster ekstra, så banene går midt i
                  korridorene i stedet for langs veggene.
    wall_weight   Hvor mye ekstra. Rett ved den forbudte sonen koster et steg
                  (1 + wall_weight) ganger lengden. 0 gir korteste bane.
    snap_radius   Start eller mål i den forbudte sonen flyttes til nærmeste lovlige celle,
                  maks så langt.
    """

    def __init__(self, grid, robot_radius=0.27, soft_radius=0.5, wall_weight=2.0,
                 snap_radius=0.5):
        self.grid = grid
        self.robot_radius = robot_radius
        self.soft_radius = max(soft_radius, robot_radius)
        self.wall_weight = wall_weight
        self.snap_radius = snap_radius

        self.clearance = obstacle_distance(grid, max_distance=self.soft_radius)
        self.traversable = bytearray(
            1 if distance > robot_radius else 0 for distance in self.clearance)
        # Uten sperresonene fra set_keep_out()
        self.open_traversable = bytes(self.traversable)

        self.cell_cost = None
        band = self.soft_radius - robot_radius
        if wall_weight > 0.0 and band > 0.0:
            self.cell_cost = [1.0] * len(self.clearance)
            for index, distance in enumerate(self.clearance):
                if robot_radius < distance < self.soft_radius:
                    closeness = (self.soft_radius - distance) / band
                    self.cell_cost[index] = 1.0 + wall_weight * closeness * closeness

    def set_keep_out(self, points, radius):
        """
        Hold robotens senter mer enn radius fra hvert punkt (x, y).

        Brukes for andre roboter som står stille, siden de ikke er på kartet. Erstatter
        punktene fra forrige kall. En tom liste fjerner alle.
        """
        grid = self.grid
        traversable = bytearray(self.open_traversable)
        reach = int(math.ceil(radius / grid.resolution))
        for x, y in points:
            col0, row0 = grid.world_to_cell(x, y)
            for row in range(row0 - reach, row0 + reach + 1):
                for col in range(col0 - reach, col0 + reach + 1):
                    if not grid.in_bounds(col, row):
                        continue
                    cx, cy = grid.cell_to_world(col, row)
                    if math.hypot(cx - x, cy - y) <= radius:
                        traversable[grid.index(col, row)] = 0
        self.traversable = traversable

    def snap(self, x, y):
        """
        Returner den lovlige cellen (col, row) nærmest punktet, eller None.

        Søket går ikke gjennom vegger, så en robot tett ved en vegg flyttes aldri til andre
        siden. Unntaket er et punkt inne i en veggcelle, for eksempel en markør. Da starter
        søket i veggen og kan komme ut på begge sider.
        """
        grid = self.grid
        col, row = grid.world_to_cell(x, y)
        if not grid.in_bounds(col, row):
            return None
        start = grid.index(col, row)
        if self.traversable[start]:
            return col, row

        through_walls = bool(grid.occupied[start])
        max_cells = self.snap_radius / grid.resolution
        seen = {start}
        queue = deque([(col, row)])
        best = None
        best_distance = INF
        while queue:
            c_col, c_row = queue.popleft()
            for d_col, d_row, _ in STRAIGHT:
                n_col = c_col + d_col
                n_row = c_row + d_row
                if not grid.in_bounds(n_col, n_row):
                    continue
                n_index = grid.index(n_col, n_row)
                if n_index in seen:
                    continue
                seen.add(n_index)
                distance = math.hypot(n_col - col, n_row - row)
                if distance > max_cells:
                    continue
                if self.traversable[n_index]:
                    if distance < best_distance:
                        best_distance = distance
                        best = (n_col, n_row)
                    continue
                if grid.occupied[n_index] and not through_walls:
                    continue
                queue.append((n_col, n_row))
        return best

    def field(self, x, y):
        """Kjør en distansetransform fra punktet (x, y). None hvis ingen lovlig celle er nær."""
        cell = self.snap(x, y)
        if cell is None:
            return None
        cost = distance_transform(self.grid, self.traversable, [cell], self.cell_cost)
        return DistanceField(self, cost, self.grid.cell_to_world(*cell))

    def path(self, start, goal):
        """
        Returner banen fra start (x, y) til goal (x, y) som [(x, y), ...], eller None.

        Transformen kjøres fra målet, og roboten går så nedover fra start.
        """
        field = self.field(goal[0], goal[1])
        if field is None:
            return None
        return field.path_from(start[0], start[1])
