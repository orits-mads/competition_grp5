"""
Markører: hvor de er, og hvilke som skal rapporteres til poengnoden.

MarkerEstimator  brukes av marker_detector på hver robot. Samler målte posisjoner og
                 gir ett stabilt estimat per markør.
ReportBook       brukes av koordinatoren. Samler estimatene fra alle robotene og velger
                 hva som sendes til poengnoden.
                 
AI: Claude wrote this file
"""
import math

INF = math.inf

# Hva hver ArUco-id betyr i konkurransen
MARKER_NAMES = {0: 'fire', 1: 'fire', 2: 'human', 3: 'fire', 4: 'big fire'}
BIG_FIRE_ID = 4


def rotate_vector(rotation, vector):
    """Roter vektoren (x, y, z) med kvaternionet rotation (x, y, z, w)."""
    x, y, z, w = rotation
    vx, vy, vz = vector
    # v + 2w (q x v) + 2 q x (q x v), der q er vektordelen av kvaternionet
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx
    ccx = y * cz - z * cy
    ccy = z * cx - x * cz
    ccz = x * cy - y * cx
    return (vx + 2.0 * (w * cx + ccx),
            vy + 2.0 * (w * cy + ccy),
            vz + 2.0 * (w * cz + ccz))


def transform_point(translation, rotation, point):
    """
    Flytt et punkt fra én ramme til en annen.

    translation og rotation er transformen fra punktets ramme til målrammen, slik et
    tf-oppslag av mål <- kilde gir den.
    """
    rotated = rotate_vector(rotation, point)
    return (rotated[0] + translation[0],
            rotated[1] + translation[1],
            rotated[2] + translation[2])


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


class MarkerEstimator:
    """
    Gjør mange støyete målinger av hver markør om til én posisjon.

    Målinger blir dårligere jo lenger unna markøren er, så bare de nærmeste beholdes.
    Estimatet er medianen av dem, som ser bort fra enkeltmålinger som bommer mye.

    max_range         målinger lenger unna enn dette kastes, m
    keep              hvor mange av de nærmeste målingene som beholdes per markør
    min_observations  ingen estimat før markøren er sett så mange ganger, mot falske
                      deteksjoner
    """

    def __init__(self, max_range=4.0, keep=15, min_observations=3):
        self.max_range = max_range
        self.keep = keep
        self.min_observations = min_observations
        # markør-id -> liste med (avstand, x, y, z), nærmeste først
        self._observations = {}

    def add(self, marker_id, position, distance):
        """
        Lagre én måling. Returnerer False hvis den ble kastet.

        position  markørens posisjon (x, y, z) i map-rammen
        distance  avstanden fra kameraet til markøren
        """
        if marker_id not in MARKER_NAMES:
            return False
        if not 0.0 < distance <= self.max_range:
            return False
        if any(math.isnan(value) or math.isinf(value) for value in position):
            return False
        observations = self._observations.setdefault(marker_id, [])
        observations.append((distance, position[0], position[1], position[2]))
        observations.sort()
        del observations[self.keep:]
        return True

    def estimate(self, marker_id):
        """Gi estimert posisjon (x, y, z), eller None hvis usikker."""
        observations = self._observations.get(marker_id, [])
        if len(observations) < self.min_observations:
            return None
        return (median([o[1] for o in observations]),
                median([o[2] for o in observations]),
                median([o[3] for o in observations]))

    def count(self, marker_id):
        return len(self._observations.get(marker_id, []))


class ReportBook:
    """
    Holder styr på hva som er rapportert til poengnoden, og hva som skal rapporteres.

    Et estimat rapporteres så snart det finnes, og på nytt når en robot har et bedre,
    til markøren er godkjent.

    min_change      et estimat må ha flyttet seg så langt for å regnes som nytt, m
    retry_interval  korteste tid mellom to rapporter av samme markør, s
    resend_after    send et uendret estimat på nytt etter så lang tid, s
    """

    def __init__(self, min_change=0.05, retry_interval=1.0, resend_after=10.0):
        self.min_change = min_change
        self.retry_interval = retry_interval
        self.resend_after = resend_after
        # markør-id -> (posisjon, robot) for godkjente markører
        self.accepted = {}
        # (markør-id, robot) -> siste estimat fra roboten
        self._estimates = {}
        # (markør-id, robot) -> (posisjon, tid) for siste rapport
        self._reported = {}
        # markør-id -> tid for siste rapport av markøren
        self._last_report = {}

    def add_estimate(self, marker_id, robot, position):
        """Lagre siste estimat (x, y, z) av en markør fra én robot."""
        if marker_id not in MARKER_NAMES or marker_id in self.accepted:
            return
        self._estimates[(marker_id, robot)] = (position[0], position[1], position[2])

    def next_report(self, now):
        """
        Gi (markør-id, robot, posisjon) som skal sendes, eller None.

        Markøren som har ventet lengst går først. Kall reported() når den er sendt.
        """
        best = None
        best_waited = None
        for (marker_id, robot), position in sorted(self._estimates.items()):
            if marker_id in self.accepted:
                continue
            last = self._last_report.get(marker_id)
            if last is not None and now - last < self.retry_interval:
                continue
            previous = self._reported.get((marker_id, robot))
            if previous is not None:
                moved = math.dist(previous[0], position)
                if moved < self.min_change and now - previous[1] < self.resend_after:
                    continue
            waited = INF if last is None else now - last
            if best is None or waited > best_waited:
                best = (marker_id, robot, position)
                best_waited = waited
        return best

    def reported(self, marker_id, robot, position, now):
        """Noter at estimatet er sendt til poengnoden."""
        self._reported[(marker_id, robot)] = (position, now)
        self._last_report[marker_id] = now

    def result(self, marker_id, robot, position, accepted):
        """Noter svaret fra poengnoden. True første gang en markør godkjennes."""
        if not accepted or marker_id in self.accepted:
            return False
        self.accepted[marker_id] = (position, robot)
        for key in [key for key in self._estimates if key[0] == marker_id]:
            del self._estimates[key]
        return True

    def missing(self):
        """Gi id-ene til markørene som ikke er godkjent ennå."""
        return [marker_id for marker_id in sorted(MARKER_NAMES)
                if marker_id not in self.accepted]
