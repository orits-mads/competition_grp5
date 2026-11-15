"""
Utforsking av et kart: hva kameraene har sett, og hvor de skal se neste gang.

Kartet er kjent, markørene er ikke. Målet er å få hver del av gulvet foran et kamera minst
én gang.

CoverageMap     gulvet i grove celler, merket som sett når et kamera har hatt fri sikt til
                dem innenfor rekkevidde

ViewPoints      steder å se fra, med cellene roboten ser når den snur på stedet der.

choose()        velger det som viser mest usett gulv for minst kjøring.

meeting_points  plasser nær et mål (den store brannen) der to roboter kan stå
"""
from collections import deque
import math

from competition_grp5.distance_transform import distance_transform
from competition_grp5.distance_transform import DistanceField
from competition_grp5.distance_transform import INF

TWO_PI = 2.0 * math.pi

# Hva som må ses av en dekningscelle. En celle ved en vegg har én bit per veggside, i
# retningen fra veggen og inn i rommet. ArUco-markører kan ikke leses fra siden, så en
# veggside teller bare som sett fra et kamera innenfor max_view_angle fra rett foran.
# En celle uten vegg ved siden av bruker FLOOR og kan ses fra alle retninger.
FACE_EAST = 1
FACE_WEST = 2
FACE_NORTH = 4
FACE_SOUTH = 8
FLOOR = 16
# (bit, x, y) for retningen hver veggside peker
FACE_NORMALS = ((FACE_EAST, 1.0, 0.0), (FACE_WEST, -1.0, 0.0),
                (FACE_NORTH, 0.0, 1.0), (FACE_SOUTH, 0.0, -1.0))


class CoverageMap:
    """
    Hvilke deler av gulvet et kamera har sett.

    grid: planleggerens GridMap
    starts: punkter (x, y) der robotene starter. Bare gulv koblet til dem må ses, så
        området utenfor bygningen tas ikke med.
    cell_size: størrelsen på en dekningscelle, m. Grovere enn planleggerens rutenett.
    max_view_angle: se FACE_EAST og de andre konstantene over, rad

    Kartet deles i ruter på cell_size. Går en vegg gjennom en rute, blir gulvet på hver side
    en egen celle. Ellers ville det å se én side av en tynn vegg telle som å se begge, og en
    markør på andre siden ble oversett.

    cell_id har én verdi per rutenettcelle: dekningscellen den hører til, eller -1 for vegg.
    """

    def __init__(self, grid, starts, cell_size=0.25, max_view_angle=math.radians(60.0)):
        self.grid = grid
        self.cos_max_angle = math.cos(max_view_angle)
        self.factor = max(1, int(round(cell_size / grid.resolution)))
        self.cell_size = self.factor * grid.resolution
        # Kartets størrelse i ruter
        self.width = int(math.ceil(grid.width / self.factor))
        self.height = int(math.ceil(grid.height / self.factor))

        self.cell_id = [-1] * (grid.width * grid.height)
        # Ruten hver dekningscelle ligger i
        self.square = []
        sums = []
        self._split_squares(sums)
        self.centers = [(grid.origin_x + (sum_x / count + 0.5) * grid.resolution,
                         grid.origin_y + (sum_y / count + 0.5) * grid.resolution)
                        for sum_x, sum_y, count in sums]

        # 1 for celler som må ses, 0 for gulv ingen kommer til
        self.wanted = bytearray(len(self.square))
        connected = self._connected_floor(starts)
        for index, is_connected in enumerate(connected):
            if is_connected:
                self.wanted[self.cell_id[index]] = 1
        # Per celle: bitene som må ses, og bitene sett så langt. seen er 1 når alle er sett.
        self.need = self._wall_faces()
        self.face_seen = bytearray(len(self.square))
        self.seen = bytearray(len(self.square))
        self.total = sum(self.wanted)
        self.remaining = self.total

    def _split_squares(self, sums):
        """Gi hver fri rutenettcelle nummeret til gulvbiten sin i ruten."""
        grid = self.grid
        width = grid.width
        factor = self.factor
        occupied = grid.occupied
        cell_id = self.cell_id
        for index in range(width * grid.height):
            if occupied[index] or cell_id[index] >= 0:
                continue
            number = len(self.square)
            col = index % width
            row = index // width
            square_col = col // factor
            square_row = row // factor
            self.square.append(square_row * self.width + square_col)
            cell_id[index] = number
            sum_x = 0
            sum_y = 0
            count = 0
            queue = deque([(col, row)])
            while queue:
                c, r = queue.popleft()
                sum_x += c
                sum_y += r
                count += 1
                for n_col, n_row in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
                    if n_col // factor != square_col or n_row // factor != square_row:
                        continue
                    if not grid.in_bounds(n_col, n_row):
                        continue
                    n_index = n_row * width + n_col
                    if occupied[n_index] or cell_id[n_index] >= 0:
                        continue
                    cell_id[n_index] = number
                    queue.append((n_col, n_row))
            sums.append((sum_x, sum_y, count))

    def _wall_faces(self):
        """Gi veggsidene hver dekningscelle grenser til."""
        grid = self.grid
        width = grid.width
        occupied = grid.occupied
        cell_id = self.cell_id
        need = bytearray(len(self.square))
        # En vegg i (col + dc, row + dr) har en side som peker motsatt vei
        sides = ((1, 0, FACE_WEST), (-1, 0, FACE_EAST), (0, 1, FACE_SOUTH),
                 (0, -1, FACE_NORTH))
        for index in range(width * grid.height):
            cell = cell_id[index]
            if cell < 0:
                continue
            col = index % width
            row = index // width
            for dc, dr, bit in sides:
                n_col = col + dc
                n_row = row + dr
                if grid.in_bounds(n_col, n_row) and occupied[n_row * width + n_col]:
                    need[cell] |= bit
        for cell, bits in enumerate(need):
            if not bits:
                need[cell] = FLOOR
        return need

    def ray_bits(self, angle):
        """Gi bitene en stråle i retning angle ser av cellene den når."""
        # Kameraet er bak cellen, i retning -(cos, sin) fra den
        to_x = -math.cos(angle)
        to_y = -math.sin(angle)
        bits = FLOOR
        for bit, normal_x, normal_y in FACE_NORMALS:
            if to_x * normal_x + to_y * normal_y >= self.cos_max_angle:
                bits |= bit
        return bits

    def _connected_floor(self, starts):
        """Gi én verdi per rutenettcelle, 1 for fri celle koblet til en start."""
        grid = self.grid
        connected = bytearray(grid.width * grid.height)
        queue = deque()
        for x, y in starts:
            col, row = grid.world_to_cell(x, y)
            if not grid.in_bounds(col, row):
                continue
            index = grid.index(col, row)
            if not grid.occupied[index] and not connected[index]:
                connected[index] = 1
                queue.append((col, row))
        while queue:
            col, row = queue.popleft()
            for n_col, n_row in ((col + 1, row), (col - 1, row), (col, row + 1),
                                 (col, row - 1)):
                if not grid.in_bounds(n_col, n_row):
                    continue
                n_index = grid.index(n_col, n_row)
                if grid.occupied[n_index] or connected[n_index]:
                    continue
                connected[n_index] = 1
                queue.append((n_col, n_row))
        return connected

    def cell_at(self, x, y):
        """Gi dekningscellen i punktet, eller -1 for vegg eller utenfor kartet."""
        col, row = self.grid.world_to_cell(x, y)
        if not self.grid.in_bounds(col, row):
            return -1
        return self.cell_id[self.grid.index(col, row)]

    def square_at(self, x, y):
        """Gi indeksen til ruten punktet ligger i."""
        col, row = self.grid.world_to_cell(x, y)
        return (row // self.factor) * self.width + col // self.factor

    def cell_center(self, cell):
        """Gi midten (x, y) av en dekningscelle."""
        return self.centers[cell]

    def visible(self, x, y, view_range, start_angle=0.0, span=TWO_PI):
        """
        Gi dekningscellene med fri sikt fra (x, y) som {celle: biter}.

        Strålene går fra start_angle og span radianer mot klokka, og stopper ved første veggcelle.
        """
        grid = self.grid
        width = grid.width
        height = grid.height
        cell_id = self.cell_id
        # Så tette stråler at ingen celle på full avstand havner mellom to
        step_angle = min(0.1, 0.5 * self.cell_size / max(view_range, 1e-6))
        rays = max(1, int(math.ceil(span / step_angle)))
        # Halve celler per steg, så en stråle ikke smetter mellom to veggceller som bare
        # møtes i et hjørne
        steps = int(2.0 * view_range / grid.resolution)
        start_x = (x - grid.origin_x) / grid.resolution
        start_y = (y - grid.origin_y) / grid.resolution

        cells = {}
        for ray in range(rays):
            if span >= TWO_PI:
                angle = start_angle + ray * span / rays
            else:
                angle = start_angle + (ray + 0.5) * span / rays
            bits = self.ray_bits(angle)
            step_x = 0.5 * math.cos(angle)
            step_y = 0.5 * math.sin(angle)
            fx = start_x
            fy = start_y
            last = -1
            for _ in range(steps + 1):
                col = int(math.floor(fx))
                row = int(math.floor(fy))
                if not (0 <= col < width and 0 <= row < height):
                    break
                index = row * width + col
                if index != last:
                    last = index
                    cell = cell_id[index]
                    if cell < 0:
                        break
                    cells[cell] = cells.get(cell, 0) | bits
                fx += step_x
                fy += step_y
        return cells

    def mark_seen(self, x, y, yaw, view_range, fov):
        """
        Merk det et kamera i (x, y) med retning yaw ser. Returnerer antall nye celler.

        fov (field of view) er kameraets horisontale synsfelt, rad.
        """
        new = 0
        for cell, bits in self.visible(x, y, view_range, yaw - 0.5 * fov, fov).items():
            if not self.wanted[cell] or self.seen[cell]:
                continue
            self.face_seen[cell] |= bits & self.need[cell]
            if self.face_seen[cell] == self.need[cell]:
                self.seen[cell] = 1
                new += 1
        self.remaining -= new
        return new

    def is_unseen(self, cell):
        return bool(self.wanted[cell]) and not self.seen[cell]

    def adds(self, cell, bits):
        """Gi True hvis disse bitene av cellen viser noe nytt."""
        return (bool(self.wanted[cell]) and not self.seen[cell]
                and bool(bits & self.need[cell] & ~self.face_seen[cell]))

    def reset(self):
        """Glem alt som er sett, for å søke gjennom hele kartet på nytt."""
        self.face_seen = bytearray(len(self.square))
        self.seen = bytearray(len(self.square))
        self.remaining = self.total

    def occupancy_data(self):
        """
        Gi rutene som nav_msgs/OccupancyGrid-data, for rviz.

        100 der noe gulv ikke er sett, 0 der alt er sett, -1 der det ikke er gulv.
        """
        data = [-1] * (self.width * self.height)
        for cell, square in enumerate(self.square):
            if not self.wanted[cell]:
                continue
            if not self.seen[cell]:
                data[square] = 100
            elif data[square] < 0:
                data[square] = 0
        return data


class ViewPoints:
    """
    Steder å se fra, og hva som kan ses fra hvert av dem.

    Kartet deles i ruter på spacing x spacing. I hver rute blir stedet lengst fra
    veggene som roboten kan nå et utsiktspunkt, så også smale korridorer får punkter
    langs midten.

    planner: distance_transform.Planner
    coverage: et CoverageMap
    fields: distansefelt fra startposisjonene. Et punkt må kunne nås i minst ett.
    spacing: sidelengden på rutene, m
    view_range: hvor langt unna kameraet kjenner igjen en markør, m
    """

    def __init__(self, planner, coverage, fields, spacing=0.5, view_range=2.5):
        grid = planner.grid
        block = max(1, int(round(spacing / grid.resolution)))
        self.points = []
        self.cells = []
        # Per utsiktspunkt: cellene det viser, og i view_bits bitene til hver av dem
        self.views = []
        self.view_bits = []

        for block_row in range(0, grid.height, block):
            for block_col in range(0, grid.width, block):
                best = None
                best_key = None
                center_col = block_col + 0.5 * block
                center_row = block_row + 0.5 * block
                for row in range(block_row, min(block_row + block, grid.height)):
                    for col in range(block_col, min(block_col + block, grid.width)):
                        index = row * grid.width + col
                        if not planner.traversable[index]:
                            continue
                        if not any(field.cost[index] < INF for field in fields):
                            continue
                        # Lengst fra veggene først, så nærmest midten av ruten
                        clearance = min(planner.clearance[index], 10.0)
                        key = (-clearance, math.hypot(col - center_col, row - center_row))
                        if best_key is None or key < best_key:
                            best_key = key
                            best = (col, row)
                if best is None:
                    continue
                x, y = grid.cell_to_world(*best)
                shown = sorted((cell, bits) for cell, bits
                               in coverage.visible(x, y, view_range).items()
                               if coverage.wanted[cell] and bits & coverage.need[cell])
                if not shown:
                    continue
                self.points.append((x, y))
                self.cells.append(grid.index(*best))
                self.views.append(tuple(cell for cell, _ in shown))
                self.view_bits.append(tuple(bits for _, bits in shown))

    def __len__(self):
        return len(self.points)

    def gain(self, view, coverage, claimed=()):
        """Gi hvor mange usette celler utsiktspunktet viser, uten claimed."""
        return sum(1 for cell, bits in zip(self.views[view], self.view_bits[view])
                   if cell not in claimed and coverage.adds(cell, bits))

    def unseen(self, view, coverage):
        """Gi cellene utsiktspunktet viser noe nytt av."""
        return [cell for cell, bits in zip(self.views[view], self.view_bits[view])
                if coverage.adds(cell, bits)]

    def choose(self, coverage, field, claimed=(), banned=(), min_gain=4, travel_bias=1.0,
               max_cost=INF, avoid=(), spread=0.0):
        """
        Gi nummeret på det beste utsiktspunktet å kjøre til, eller None.

        Det beste punktet har flest usette celler per meter kjøring.

        field: DistanceField fra robotens posisjon
        claimed: celler en annen robot allerede er på vei for å se
        banned: punkter som ikke skal brukes, for eksempel de roboten ikke nådde
        min_gain: minste antall usette celler
        travel_bias: legges til reisekostnaden, så et punkt rett ved roboten ikke vinner bare
            fordi det nesten ikke koster noe
        max_cost: bare punkter maks så langt unna i reisekostnad
        avoid: punkter (x, y) de andre robotene er på vei til
        spread: punkter nærmere avoid enn dette får lavere poeng, så robotene sprer seg
        """
        best = None
        best_score = 0.0
        for view, index in enumerate(self.cells):
            if view in banned:
                continue
            cost = field.cost[index]
            if cost == INF or cost > max_cost:
                continue
            gain = self.gain(view, coverage, claimed)
            if gain < min_gain:
                continue
            score = gain / (cost + travel_bias)
            if spread > 0.0 and avoid:
                x, y = self.points[view]
                nearest = min(math.hypot(x - ax, y - ay) for ax, ay in avoid)
                if nearest < spread:
                    score *= max(0.05, nearest / spread)
            if score > best_score:
                best_score = score
                best = view
        return best


def nearest_traversable(planner, x, y, radius):
    """Gi den lovlige cellen (col, row) nærmest (x, y) innenfor radius, eller None."""
    grid = planner.grid
    col0, row0 = grid.world_to_cell(x, y)
    reach = int(math.ceil(radius / grid.resolution)) + 1
    best = None
    best_distance = INF
    for row in range(row0 - reach, row0 + reach + 1):
        for col in range(col0 - reach, col0 + reach + 1):
            if not grid.in_bounds(col, row) or not planner.traversable[grid.index(col, row)]:
                continue
            cx, cy = grid.cell_to_world(col, row)
            distance = math.hypot(cx - x, cy - y)
            if distance <= radius and distance < best_distance:
                best_distance = distance
                best = (col, row)
    return best


def place_field(planner, target, radius):
    """
    Gi et DistanceField fra den lovlige cellen nærmest target, eller None.

    cost_at() gir hvor langt en robot må kjøre for å komme til target.
    """
    source = nearest_traversable(planner, target[0], target[1], radius)
    if source is None:
        return None
    cost = distance_transform(planner.grid, planner.traversable, [source], planner.cell_cost)
    return DistanceField(planner, cost, planner.grid.cell_to_world(*source))


def meeting_points(planner, target, robots, radius=1.2, separation=0.6, count=2,
                   keep_order=False, field=None):
    """
    Velg robotene som skal møtes ved target, og hvor hver skal stå.

    target: (x, y) å møtes ved, for eksempel den store brannen
    robots: liste med (navn, (x, y))
    radius: hver plass er maks så langt fra target i rett linje. Den kan være på andre siden
        av en vegg, det godtar poengnoden.
    separation: minste avstand mellom plassene, så robotene ikke kolliderer
    count: hvor mange roboter som sendes
    keep_order: False: send de med kortest vei, og den første går innerst. True: send de
        første count som kan nå fram, i gitt rekkefølge.
    field: place_field() for target og radius, hvis den allerede er regnet ut

    Returnerer {navn: (x, y)}, med færre enn count hvis ikke nok roboter kan nå fram.

    Én robot går innerst, til radius - separation, så den neste kan stoppe i kanten av
    radius.
    Hver robot stopper ved første punkt på veien inn som er nær nok.
    """
    if field is None:
        field = place_field(planner, target, radius)
    if field is None:
        return {}

    reachable = []
    for order, (name, (x, y)) in enumerate(robots):
        robot_cost = field.cost_at(x, y)
        if robot_cost < INF:
            reachable.append((order if keep_order else robot_cost, name, (x, y)))
    reachable.sort()

    chosen = {}
    for _, name, (x, y) in reachable:
        if len(chosen) >= count:
            break
        depth = len(chosen)
        limit = max(radius - (count - 1 - depth) * separation, 0.0)
        point = None
        path = field.path_from(x, y) or []
        if path and math.hypot(path[-1][0] - target[0], path[-1][1] - target[1]) > limit:
            # Veien inn slutter utenfor grensen: kjør til enden
            limit = math.hypot(path[-1][0] - target[0], path[-1][1] - target[1])
        for px, py in path:
            if math.hypot(px - target[0], py - target[1]) > limit:
                continue
            if all(math.hypot(px - ox, py - oy) >= separation
                   for ox, oy in chosen.values()):
                point = (px, py)
                break
        if point is None:
            point = _cheapest_place(planner, field, target, radius, separation,
                                    list(chosen.values()))
        if point is not None:
            chosen[name] = point
    return chosen


def _cheapest_place(planner, field, target, radius, separation, taken):
    """Gi den billigste plassen innenfor radius, unna taken, eller None."""
    grid = planner.grid
    col0, row0 = grid.world_to_cell(target[0], target[1])
    reach = int(math.ceil(radius / grid.resolution)) + 1
    best = None
    best_cost = INF
    for row in range(row0 - reach, row0 + reach + 1):
        for col in range(col0 - reach, col0 + reach + 1):
            if not grid.in_bounds(col, row):
                continue
            index = grid.index(col, row)
            if not planner.traversable[index] or field.cost[index] >= best_cost:
                continue
            x, y = grid.cell_to_world(col, row)
            if math.hypot(x - target[0], y - target[1]) > radius:
                continue
            if any(math.hypot(x - ox, y - oy) < separation for ox, oy in taken):
                continue
            best_cost = field.cost[index]
            best = (x, y)
    return best
