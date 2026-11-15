"""
Banefølging for en robot med differensialdrift.

PathFollower  gjør en bane og robotens posisjon om til fart og dreiehastighet.
Navigator     planlegger en bane til et mål, følger den, stopper for hindringer lidaren
              ser, og rygger og planlegger på nytt hvis roboten blir blokkert eller sitter
              fast.
"""
import math

INF = math.inf

IDLE = 'idle'
DRIVING = 'driving'
REACHED = 'reached'
FAILED = 'failed'


def normalize_angle(angle):
    """Gi vinkelen pakket inn i området -pi til pi."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x, y, z, w):
    """Gi rotasjonen om z-aksen for et kvaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def strip_clearance(ranges, angle_min, angle_increment, half_width,
                    range_min=0.0, backwards=False):
    """
    Gi avstanden til nærmeste lidarpunkt i stripen roboten kjører inn i.

    Stripen går rett fram (eller rett bak) fra lidaren og er half_width bred til hver
    side. En stripe i stedet for en sektor gjør at en dørkarm ved siden av roboten ikke
    teller, men noe i veien gjør det. Uendelig hvis stripen er tom.
    """
    nearest = INF
    for i, distance in enumerate(ranges):
        if math.isnan(distance) or math.isinf(distance) or distance < range_min:
            continue
        angle = angle_min + i * angle_increment
        along = distance * math.cos(angle)
        if backwards:
            along = -along
        if along <= 0.0:
            continue
        if abs(distance * math.sin(angle)) <= half_width and along < nearest:
            nearest = along
    return nearest


class PathFollower:
    """
    Styrer roboten langs en bane ved å sikte på et punkt litt foran på banen.

    max_speed       fart på rett bane, m/s
    max_turn        høyeste dreiehastighet, rad/s
    lookahead       hvor langt fram på banen roboten sikter, m
    goal_tolerance  banen er ferdig når roboten er så nær siste punkt, m
    heading_gain    dreiehastighet per radian retningsfeil
    stop_angle      med retningsfeil over dette snur roboten på stedet, rad
    """

    def __init__(self, max_speed=0.26, max_turn=1.5, lookahead=0.35, goal_tolerance=0.12,
                 heading_gain=2.5, stop_angle=0.9):
        self.max_speed = max_speed
        self.max_turn = max_turn
        self.lookahead = lookahead
        self.goal_tolerance = goal_tolerance
        self.heading_gain = heading_gain
        self.stop_angle = stop_angle
        # Laveste fart inn mot målet, og fart per meter igjen til målet
        self.min_speed = 0.05
        self.goal_gain = 1.0
        # Hvor langt langs banen vi leter etter punktet nærmest roboten
        self.search_distance = 1.0

        self.done = True
        # Avstand fra roboten til nærmeste punkt på banen
        self.cross_track = 0.0
        self._path = []
        self._index = 0

    def set_path(self, points):
        """Begynn å følge en ny bane gitt som en liste med (x, y)-punkter."""
        self._path = list(points)
        self._index = 0
        self.cross_track = 0.0
        self.done = not self._path

    def update(self, x, y, yaw):
        """Gi (fart, dreiehastighet) for posisjonen. Setter done ved målet."""
        path = self._path
        if not path:
            self.done = True
            return 0.0, 0.0
        last = len(path) - 1

        # Finn punktet nærmest roboten. Søk bare framover fra forrige, så roboten aldri
        # snur tilbake langs banen.
        nearest = self._index
        nearest_distance = INF
        travelled = 0.0
        index = self._index
        while True:
            distance = math.hypot(path[index][0] - x, path[index][1] - y)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = index
            if index == last or travelled > self.search_distance:
                break
            travelled += math.hypot(path[index + 1][0] - path[index][0],
                                    path[index + 1][1] - path[index][1])
            index += 1
        self._index = nearest
        self.cross_track = nearest_distance

        goal_distance = math.hypot(path[last][0] - x, path[last][1] - y)
        if goal_distance < self.goal_tolerance:
            self.done = True
            return 0.0, 0.0

        # Sikt på første punkt som er minst lookahead unna
        target = last
        for index in range(nearest, last + 1):
            if math.hypot(path[index][0] - x, path[index][1] - y) >= self.lookahead:
                target = index
                break

        heading = math.atan2(path[target][1] - y, path[target][0] - x)
        error = normalize_angle(heading - yaw)

        turn = self.heading_gain * error
        turn = max(-self.max_turn, min(self.max_turn, turn))

        # Full fart når roboten peker langs banen, stillestående når den peker langt feil
        speed = self.max_speed * max(0.0, 1.0 - abs(error) / self.stop_angle)
        if target == last:
            speed = min(speed, self.min_speed + self.goal_gain * goal_distance)

        return speed, turn


class Navigator:
    """
    Kjører roboten til et mål: planlegg, følg, og kom løs når noe er i veien.

    planner          en distance_transform.Planner
    follower         en PathFollower
    stop_distance    stopp når lidaren ser noe så nær i kjørestripen. Målt fra lidaren,
                     som sitter ca. 0.14 m bak fronten på en TurtleBot3 Waffle.
    slow_distance    begynn å bremse ved denne avstanden
    blocked_timeout  sekunder å vente på at en hindring flytter seg før roboten rygger
    stuck_timeout    sekunder uten bevegelse før roboten regnes som fast
    max_retries      gi opp målet etter så mange forsøk på samme sted

    Kall update() jevnlig. status er 'idle', 'driving', 'reached' eller 'failed'.
    """

    def __init__(self, planner, follower, stop_distance=0.30, slow_distance=0.60,
                 blocked_timeout=3.0, stuck_timeout=8.0, max_retries=3,
                 off_path_distance=0.40, backup_speed=0.10, backup_time=1.5):
        self.planner = planner
        self.follower = follower
        self.stop_distance = stop_distance
        self.slow_distance = slow_distance
        self.blocked_timeout = blocked_timeout
        self.stuck_timeout = stuck_timeout
        self.max_retries = max_retries
        self.off_path_distance = off_path_distance
        self.backup_speed = backup_speed
        self.backup_time = backup_time

        self.status = IDLE
        self.goal = None
        self.path = []
        self._retries = 0
        self._retry_position = None
        self._blocked_since = None
        self._backup_until = None
        self._progress_position = None
        self._progress_time = 0.0

    def set_goal(self, goal, x, y, now):
        """
        Planlegg en bane fra (x, y) til goal og begynn å kjøre.

        Returnerer False og setter status 'failed' hvis det ikke finnes en bane.
        """
        self.goal = (goal[0], goal[1])
        self._retries = 0
        self._retry_position = None
        self._backup_until = None
        return self._plan(x, y, now)

    def cancel(self):
        """Stopp og glem målet."""
        self.status = IDLE
        self.goal = None
        self.path = []
        self._backup_until = None

    def update(self, now, x, y, yaw, front=INF, rear=INF):
        """
        Gi (fart, dreiehastighet).

        now    tid i sekunder
        front  strip_clearance() foran roboten
        rear   strip_clearance() bak roboten
        """
        if self.status != DRIVING:
            return 0.0, 0.0

        if self._backup_until is not None:
            if now < self._backup_until and rear > self.stop_distance:
                return -self.backup_speed, 0.0
            self._backup_until = None
            if not self._plan(x, y, now):
                return 0.0, 0.0

        speed, turn = self.follower.update(x, y, yaw)
        if self.follower.done:
            self.status = REACHED
            return 0.0, 0.0

        if self.follower.cross_track > self.off_path_distance:
            # Dyttet eller skled bort fra banen: planlegg på nytt herfra
            return self._recover(now, x, y, rear, back_up=False)

        # Noe som står i veien for lenge
        if speed > 0.0 and front < self.stop_distance:
            speed = 0.0
            if self._blocked_since is None:
                self._blocked_since = now
            elif now - self._blocked_since > self.blocked_timeout:
                return self._recover(now, x, y, rear, back_up=True)
        else:
            self._blocked_since = None
            if front < self.slow_distance:
                span = self.slow_distance - self.stop_distance
                speed *= max(0.25, (front - self.stop_distance) / span)

        # Kommer ikke av flekken, for eksempel hjul som spinner mot noe lidaren ikke ser
        if self._progress_position is None or math.hypot(
                x - self._progress_position[0], y - self._progress_position[1]) > 0.05:
            self._progress_position = (x, y)
            self._progress_time = now
        elif now - self._progress_time > self.stuck_timeout:
            return self._recover(now, x, y, rear, back_up=True)

        return speed, turn

    def _plan(self, x, y, now):
        path = self.planner.path((x, y), self.goal)
        if path is None:
            self.status = FAILED
            self.path = []
            return False
        self.path = path
        self.follower.set_path(path)
        self.status = DRIVING
        self._blocked_since = None
        self._progress_position = (x, y)
        self._progress_time = now
        return True

    def _recover(self, now, x, y, rear, back_up):
        # Forsøk teller bare mot grensen så lenge roboten er på samme sted
        if self._retry_position is not None and math.hypot(
                x - self._retry_position[0], y - self._retry_position[1]) > 0.5:
            self._retries = 0
        self._retry_position = (x, y)
        self._retries += 1
        if self._retries > self.max_retries:
            self.status = FAILED
            self.path = []
            return 0.0, 0.0

        self._blocked_since = None
        self._progress_position = (x, y)
        self._progress_time = now
        if back_up and rear > self.stop_distance:
            self._backup_until = now + self.backup_time
            return -self.backup_speed, 0.0
        self._plan(x, y, now)
        return 0.0, 0.0
