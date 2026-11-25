"""
Lagets plan: hvilken robot leter hvor, og når to av dem møtes ved den store brannen.

Koordinatoren gir den posisjoner, navigasjonsstatus og godkjente markører, og utfører
handlingene den returnerer.

Moduser for hver robot:

    start    venter på første valg
    explore  kjører til et utsiktspunkt
    look     snur på stedet ved et utsiktspunkt
    meet     kjører til, eller venter på, plassen sin ved den store brannen
    park     hjelper som står stille
    idle     ingenting å gjøre nå
    done     oppdraget er ferdig

Hjelperen leter aldri. Den venter til den store brannen er funnet og kjører dit. Roboten
som fant brannen leter i nærheten mens hjelperen er på vei, og drar tilbake når
hjelperen er omtrent like nær. Etter møtet blir hjelperen stående.

Handlinger fra step():

    ('goal', robot, (x, y))   kjør til punktet via robot_controller
    ('cancel', robot)         stopp
    ('turn', robot, rate)     snu på stedet med rate rad/s, 0.0 for å stoppe

Møtet skjer så snart den store brannen er godkjent, fordi klokken først stopper når
møtet er gjort og alle markører er rapportert. Uten hjelper møtes de to robotene med
kortest vei.

AI: Claude wrote this file
"""
import math

from competition_grp5.distance_transform import INF
from competition_grp5.exploration import CoverageMap
from competition_grp5.exploration import meeting_points
from competition_grp5.exploration import place_field
from competition_grp5.exploration import ViewPoints
from competition_grp5.marker_tracking import BIG_FIRE_ID
from competition_grp5.marker_tracking import MARKER_NAMES
from competition_grp5.path_follower import DRIVING
from competition_grp5.path_follower import FAILED
from competition_grp5.path_follower import normalize_angle
from competition_grp5.path_follower import REACHED

START = 'start'
EXPLORE = 'explore'
LOOK = 'look'
MEET = 'meet'
PARK = 'park'
IDLE = 'idle'
DONE = 'done'


class RobotState:

    def __init__(self, name, helper=False):
        self.name = name
        self.helper = helper
        self.pose = None
        self.mode = START
        # Siste mål roboten fikk, og når
        self.goal = None
        self.goal_time = 0.0
        # Nummeret på utsiktspunktet roboten er på vei til eller ser fra
        self.view = None
        # Siste nav_status: ('driving', (x, y)), ('idle', None) osv.
        self.nav = ('idle', None)
        # Snuing ved et utsiktspunkt
        self.turned = 0.0
        self.turn_direction = 1.0
        self.last_yaw = None
        # Når vi ser etter nytt utsiktspunkt etter å ikke ha funnet noe
        self.next_choice = 0.0
        self.last_check = 0.0

    def nav_is_for_goal(self, tolerance=0.01):
        goal = self.nav[1]
        return (self.goal is not None and goal is not None
                and abs(goal[0] - self.goal[0]) < tolerance
                and abs(goal[1] - self.goal[1]) < tolerance)


class Mission:
    """
    Bestemmer hva hver robot gjør.

    planner        en distance_transform.Planner for kartet
    robots         robotnavn
    helpers        navnene i robots som bare hjelper ved den store brannen
    view_range     hvor langt unna kameraet kjenner igjen en markør, m
    fov            kameraets horisontale synsfelt, rad. 1.085 på Waffle Pi.
    max_view_angle se CoverageMap
    coverage_cell  størrelsen på en dekningscelle, m
    spacing        avstand mellom utsiktspunkter, m
    min_gain       et utsiktspunkt må vise så mange usette celler for å være verdt turen
    local_radius   utsiktspunkter så nær i reisekostnad tas først, m
    local_min_gain lavere krav for punkter i nærheten, så små hjørner sjekkes mens
                   roboten er der
    spread         punkter nærmere enn dette til en annen robots mål får lavere poeng, m
    turn_rate      dreiehastighet ved et utsiktspunkt, rad/s
    meet_radius    robotene sendes til plasser maks så langt fra den store brannen, m
    meet_done_radius
                   møtet er gjort når begge er så nær, m. Poengnoden krever 2 m fra den
                   ekte markøren, og godkjent posisjon er maks 0.5 m feil, så 1.45 m er
                   alltid nok. Større enn meet_radius, så møtet er gjort før robot nummer
                   to har stoppet.
    meet_hold      hvor lenge robotene blir stående sammen, s. Poengnoden sjekker hvert
                   0.5 s, så 1 s dekker minst én sjekk.
    ack_timeout    send målet på nytt hvis roboten ikke har startet etter så lang tid, s
    recall_margin  roboten som fant brannen drar tilbake når hjelperen er maks så mye
                   lenger unna enn den selv, m reisekostnad
    """

    def __init__(self, planner, robots, view_range=2.5, fov=1.085, coverage_cell=0.25,
                 spacing=0.5, min_gain=4, local_radius=3.0, local_min_gain=2, spread=1.5,
                 turn_rate=0.8, meet_radius=1.2, meet_done_radius=1.45,
                 meet_separation=0.6, meet_hold=1.0, ack_timeout=4.0, helpers=(),
                 recall_margin=0.5, max_view_angle=math.radians(60.0)):
        self.planner = planner
        self.robots = {name: RobotState(name, helper=name in helpers) for name in robots}
        self.view_range = view_range
        self.fov = fov
        self.max_view_angle = max_view_angle
        self.coverage_cell = coverage_cell
        self.spacing = spacing
        self.min_gain = min_gain
        self.local_radius = local_radius
        self.local_min_gain = min(local_min_gain, min_gain)
        self.spread = spread
        self.turn_rate = turn_rate
        self.meet_radius = meet_radius
        self.meet_done_radius = max(meet_done_radius, meet_radius)
        self.meet_separation = meet_separation
        self.meet_hold = meet_hold
        self.ack_timeout = ack_timeout
        self.recall_margin = recall_margin

        self.coverage = None
        self.view_points = None
        # Utsiktspunkter robotene ikke nådde, og de de har sett fra. Brukes ikke igjen
        # før søket starter på nytt.
        self.banned = set()
        self.visited = set()
        self.passes = 1
        # markør-id -> posisjon (x, y, z) for godkjente markører
        self.accepted = {}
        # robotnavn -> plass ved den store brannen, når møtet er planlagt
        self.meeting = None
        self.meeting_since = None
        self.meeting_done = False
        self.meeting_retry = 0.0
        # Reisekostnad til den store brannen, som DistanceField og per utsiktspunkt
        self.fire_field = None
        self.fire_cost = None
        # Roboten som leter i nærheten til hjelperen nesten er fram
        self.waiting = None
        self.finished = False
        # Meldinger noden skal logge
        self.messages = []

    # Inndata fra noden

    def update_pose(self, robot, x, y, yaw):
        state = self.robots[robot]
        if state.mode == LOOK and state.last_yaw is not None:
            state.turned += abs(normalize_angle(yaw - state.last_yaw))
        state.last_yaw = yaw
        state.pose = (x, y, yaw)

    def update_nav(self, robot, status, goal):
        self.robots[robot].nav = (status, goal)

    def marker_accepted(self, marker_id, position):
        self.accepted[marker_id] = position

    def take_messages(self):
        messages = self.messages
        self.messages = []
        return messages

    @property
    def ready(self):
        return self.view_points is not None

    # Valg

    def prepare(self):
        """Lag dekningskartet og utsiktspunktene når posisjonene er kjent."""
        starts = [state.pose[:2] for state in self.robots.values()
                  if state.pose is not None]
        fields = [self.planner.field(x, y) for x, y in starts]
        fields = [field for field in fields if field is not None]
        self.coverage = CoverageMap(self.planner.grid, starts, cell_size=self.coverage_cell,
                                    max_view_angle=self.max_view_angle)
        self.view_points = ViewPoints(self.planner, self.coverage, fields,
                                      spacing=self.spacing, view_range=self.view_range)
        self.messages.append(
            '%d view points over %d coverage cells of %.2f m'
            % (len(self.view_points), self.coverage.total, self.coverage.cell_size))

    def step(self, now):
        """Gi handlingene som skal utføres nå."""
        if self.finished:
            return []
        if not self.ready:
            # Bare de som leter trengs for å starte. En hjelper som mangler skal ikke
            # holde de andre igjen.
            if any(state.pose is None for state in self.robots.values()
                   if not state.helper):
                return []
            self.prepare()

        for state in self.robots.values():
            if state.pose is not None:
                x, y, yaw = state.pose
                self.coverage.mark_seen(x, y, yaw, self.view_range, self.fov)

        actions = []
        all_found = all(marker_id in self.accepted for marker_id in MARKER_NAMES)
        if all_found and self.meeting_done:
            self.finished = True
            for state in self.robots.values():
                state.mode = DONE
                actions.append(('turn', state.name, 0.0))
                actions.append(('cancel', state.name))
            self.messages.append('All markers found and the big fire is put out. Done.')
            return actions

        self.update_meeting(now, actions)
        for state in self.robots.values():
            if state.mode == MEET:
                continue
            if state.helper:
                if state.mode == START:
                    state.mode = PARK
                continue
            if all_found:
                if state.mode != IDLE:
                    self.stop(state, actions)
                    state.mode = IDLE
                continue
            self.update_explorer(state, now, actions)
        return actions

    def send(self, state, goal, now, actions):
        if state.mode == LOOK:
            actions.append(('turn', state.name, 0.0))
        state.goal = goal
        state.goal_time = now
        actions.append(('goal', state.name, goal))

    def stop(self, state, actions):
        if state.mode == LOOK:
            actions.append(('turn', state.name, 0.0))
        if state.goal is not None:
            actions.append(('cancel', state.name))
        state.goal = None
        state.view = None

    def resend_if_ignored(self, state, now, actions):
        """Send målet på nytt hvis roboten ikke har tatt det."""
        if not state.nav_is_for_goal() and now - state.goal_time > self.ack_timeout:
            state.goal_time = now
            actions.append(('goal', state.name, state.goal))

    # Den store brannen

    def update_meeting(self, now, actions):
        if BIG_FIRE_ID not in self.accepted or self.meeting_done:
            return
        target = self.accepted[BIG_FIRE_ID][:2]

        if self.meeting is None:
            if now >= self.meeting_retry:
                self.plan_meeting(target, now, actions)
            return
        if self.waiting is not None:
            self.recall_if_time(now, actions)

        in_place = 0
        for name in self.meeting:
            state = self.robots[name]
            x, y, _ = state.pose
            if math.hypot(x - target[0], y - target[1]) <= self.meet_done_radius:
                in_place += 1
            elif state.mode != MEET:
                # Leter i nærheten mens hjelperen er på vei
                continue
            elif state.nav_is_for_goal() and state.nav[0] == FAILED:
                # Kom ikke fram: planlegg møtet på nytt fra der robotene er
                self.messages.append(name + ' could not reach its place at the big fire')
                self.meeting = None
                self.waiting = None
                self.meeting_retry = now + 1.0
                return
            else:
                self.resend_if_ignored(state, now, actions)

        if in_place < len(self.meeting):
            self.meeting_since = None
            return
        if self.meeting_since is None:
            self.meeting_since = now
            self.messages.append('Both robots are within %.2f m of the big fire'
                                 % self.meet_done_radius)
        elif now - self.meeting_since >= self.meet_hold:
            self.meeting_done = True
            self.waiting = None
            self.messages.append('Big fire meeting done, back to searching')
            for name in self.meeting:
                state = self.robots[name]
                if state.helper:
                    # Blir stående innerst, ute av veien
                    state.mode = PARK
                elif state.mode == MEET:
                    state.mode = EXPLORE
                    state.goal = None
                    state.view = None

    def by_cost_to(self, field, helpers):
        """Gi [(navn, (x, y))] for hjelperne eller de som leter, nærmeste først."""
        robots = []
        for state in self.robots.values():
            if state.pose is None or state.helper != helpers:
                continue
            cost = field.cost_at(state.pose[0], state.pose[1])
            if cost < INF:
                robots.append((cost, state.name, state.pose[:2]))
        robots.sort()
        return [(name, position) for _, name, position in robots]

    def plan_meeting(self, target, now, actions):
        field = place_field(self.planner, target, self.meet_radius)
        places = {}
        helpers = []
        if field is not None:
            helpers = self.by_cost_to(field, helpers=True)
            searchers = self.by_cost_to(field, helpers=False)
            if helpers and searchers:
                # Hjelperen går innerst, siden den blir stående etter møtet
                places = meeting_points(self.planner, target, [helpers[0], searchers[0]],
                                        radius=self.meet_radius,
                                        separation=self.meet_separation,
                                        keep_order=True, field=field)
            else:
                helpers = []
                places = meeting_points(self.planner, target, searchers,
                                        radius=self.meet_radius,
                                        separation=self.meet_separation, field=field)
        if len(places) < 2:
            self.messages.append('Found no places for two robots at the big fire')
            self.meeting_retry = now + 5.0
            # Let videre i mellomtiden
            for state in self.robots.values():
                if state.mode == MEET:
                    state.mode = PARK if state.helper else EXPLORE
                    state.goal = None
            return

        self.meeting = places
        self.fire_field = field
        for name, place in sorted(places.items()):
            state = self.robots[name]
            if helpers and not state.helper:
                # Leter i nærheten til hjelperen nesten er fram
                self.waiting = name
                continue
            self.send(state, place, now, actions)
            state.mode = MEET
            state.view = None
        self.messages.append(
            'Meeting at the big fire: ' + ', '.join(
                '%s to (%.2f, %.2f)' % (name, place[0], place[1])
                for name, place in sorted(places.items())))
        if self.waiting is not None:
            self.fire_cost = [field.cost_at(x, y) for x, y in self.view_points.points]
            self.messages.append(
                '%s searches close by until the helper is almost there' % self.waiting)
            self.recall_if_time(now, actions)

    def helper_cost(self):
        """Gi reisekostnaden mellom den store brannen og hjelperen."""
        costs = [self.fire_field.cost_at(*self.robots[name].pose[:2])
                 for name in self.meeting if self.robots[name].helper]
        return min(costs) if costs else 0.0

    def recall_if_time(self, now, actions):
        """Send ventende robot til brannen når hjelperen er omtrent like nær."""
        state = self.robots[self.waiting]
        all_found = all(marker_id in self.accepted for marker_id in MARKER_NAMES)
        finder_cost = self.fire_field.cost_at(*state.pose[:2])
        if all_found or self.helper_cost() <= finder_cost + self.recall_margin:
            self.call_to_meeting(state, now, actions)

    def call_to_meeting(self, state, now, actions):
        self.waiting = None
        state.view = None
        self.send(state, self.meeting[state.name], now, actions)
        state.mode = MEET
        self.messages.append(state.name + ' goes back to the big fire')

    def parked(self):
        """
        Gi {navn: (x, y)} for roboter som står stille en stund.

        De andre planlegger rundt dem, siden lidaren først ser dem når de er i veien.
        """
        spots = {}
        for state in self.robots.values():
            if (state.pose is not None and state.mode in (PARK, MEET, IDLE, DONE)
                    and state.nav[0] != DRIVING):
                spots[state.name] = state.pose[:2]
        return spots

    # Leting

    def update_explorer(self, state, now, actions):
        if state.mode == LOOK:
            if (state.turned >= 2.0 * math.pi - 0.1 or self.view_points.gain(
                    state.view, self.coverage) < self.local_min_gain):
                actions.append(('turn', state.name, 0.0))
                self.visited.add(state.view)
                state.mode = EXPLORE
                state.goal = None
                state.view = None
            else:
                actions.append(('turn', state.name, state.turn_direction * self.turn_rate))
                return

        if state.mode in (EXPLORE, START) and state.goal is not None:
            status = state.nav[0] if state.nav_is_for_goal() else None
            if status == REACHED:
                self.start_looking(state, actions)
                return
            if status == FAILED:
                self.messages.append(
                    '%s could not reach view point %d, trying another'
                    % (state.name, state.view))
                self.banned.add(state.view)
                state.goal = None
                state.view = None
            else:
                self.resend_if_ignored(state, now, actions)
                # Gi opp punktet når andre roboter har sett det det viser
                if now - state.last_check < 1.0:
                    return
                state.last_check = now
                if self.view_points.gain(state.view, self.coverage) >= self.local_min_gain:
                    return
                state.goal = None
                state.view = None

        if now < state.next_choice:
            return
        self.choose_view_point(state, now, actions)

    def start_looking(self, state, actions):
        """Snu på stedet, mot siden med mest usett gulv."""
        unseen = self.view_points.unseen(state.view, self.coverage)
        if len(unseen) < self.local_min_gain:
            self.visited.add(state.view)
            state.goal = None
            state.view = None
            state.mode = EXPLORE
            return
        x, y, yaw = state.pose
        sum_x = 0.0
        sum_y = 0.0
        for cell in unseen:
            cx, cy = self.coverage.cell_center(cell)
            sum_x += cx - x
            sum_y += cy - y
        bearing = normalize_angle(math.atan2(sum_y, sum_x) - yaw)
        state.turn_direction = 1.0 if bearing >= 0.0 else -1.0
        state.turned = 0.0
        state.mode = LOOK
        actions.append(('turn', state.name, state.turn_direction * self.turn_rate))

    def claimed_by_others(self, state):
        claimed = set()
        for other in self.robots.values():
            if other is state or other.view is None or other.mode not in (EXPLORE, LOOK):
                continue
            claimed.update(self.view_points.views[other.view])
        return claimed

    def goals_of_others(self, state):
        return [other.goal for other in self.robots.values()
                if other is not state and other.goal is not None
                and other.mode in (EXPLORE, LOOK)]

    def choose_view_point(self, state, now, actions):
        x, y, _ = state.pose
        field = self.planner.field(x, y)
        if field is None:
            self.messages.append('%s is not on the map at (%.2f, %.2f)' % (state.name, x, y))
            state.next_choice = now + 2.0
            return

        points = self.view_points
        claimed = self.claimed_by_others(state)
        avoid = self.goals_of_others(state)
        excluded = self.banned | self.visited
        if self.waiting == state.name:
            # Venter på hjelperen: bare punkter i nærheten og ikke lenger fra brannen
            # enn hjelperen, så roboten er tilbake omtrent når den kommer
            budget = self.helper_cost()
            far = {view for view, cost in enumerate(self.fire_cost) if cost > budget}
            view = points.choose(self.coverage, field, claimed, excluded | far,
                                 min_gain=self.local_min_gain, max_cost=self.local_radius,
                                 avoid=avoid, spread=self.spread)
            if view is None:
                self.call_to_meeting(state, now, actions)
            else:
                state.view = view
                self.send(state, points.points[view], now, actions)
                state.mode = EXPLORE
                state.last_check = now
            return
        # Først det som er igjen i nærheten, også små hjørner. Punkter nær andre
        # roboters mål får lavere poeng, så robotene sprer seg.
        view = points.choose(self.coverage, field, claimed, excluded,
                             min_gain=self.local_min_gain, max_cost=self.local_radius,
                             avoid=avoid, spread=self.spread)
        if view is None:
            view = points.choose(self.coverage, field, claimed, excluded,
                                 min_gain=self.min_gain, avoid=avoid, spread=self.spread)
        if view is None:
            # Bedre å se på samme område som en annen robot enn å stå stille
            view = points.choose(self.coverage, field, (), excluded, min_gain=self.min_gain)
        if view is None and not any(other.view is not None
                                    for other in self.robots.values()):
            # Alt er sett, men markører mangler. Søk på nytt.
            self.passes += 1
            self.coverage.reset()
            self.banned.clear()
            self.visited.clear()
            self.messages.append(
                'Searched everything, but %d markers are missing. Starting search %d.'
                % (len(MARKER_NAMES) - len(self.accepted), self.passes))
            view = self.view_points.choose(self.coverage, field, (), (),
                                           min_gain=self.min_gain)
        if view is None:
            if state.mode != IDLE:
                self.stop(state, actions)
                state.mode = IDLE
            state.next_choice = now + 2.0
            return

        state.view = view
        self.send(state, self.view_points.points[view], now, actions)
        state.mode = EXPLORE
        state.last_check = now
