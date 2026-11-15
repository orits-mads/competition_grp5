"""
Tests for mission.py. They need no ROS.

The robots are simulated very roughly: a robot that gets a goal is at the goal on the
next step, or drives along the planned path at a given speed, and a robot that is told
to turn turns at that rate.

Run with pytest from the package folder::

    python3 -m pytest test/test_mission.py

or without pytest::

    PYTHONPATH=. python3 test/test_mission.py
"""
import math

from competition_grp5.mission import DONE
from competition_grp5.mission import EXPLORE
from competition_grp5.mission import IDLE
from competition_grp5.mission import LOOK
from competition_grp5.mission import MEET
from competition_grp5.mission import Mission
from competition_grp5.mission import PARK
# The test folder is on the import path both under pytest and when run directly
from test_exploration import make_planner

DT = 0.2


class FakeRobot:
    """
    Teleports to its goal, and turns while told to.

    With a planner and a speed in m/s it drives along the planned path instead.
    """

    def __init__(self, name, x, y, yaw=0.0, planner=None, speed=None):
        self.name = name
        self.planner = planner
        self.speed = speed
        self.path = []
        self.x = x
        self.y = y
        self.yaw = yaw
        self.turn = 0.0
        self.goal = None
        self.status = 'idle'
        self.goals = []
        # Goals to fail instead of reaching, and goals to ignore
        self.fail = set()
        self.ignore = 0

    def act(self, action):
        kind = action[0]
        if kind == 'goal':
            if self.ignore > 0:
                self.ignore -= 1
                return
            self.goal = (round(action[2][0], 3), round(action[2][1], 3))
            self.goals.append(self.goal)
            self.status = 'driving'
            self.turn = 0.0
            if self.speed is not None:
                self.path = self.planner.path((self.x, self.y), self.goal) or []
        elif kind == 'cancel':
            self.goal = None
            self.status = 'idle'
        elif kind == 'turn':
            self.turn = action[2]

    def move(self):
        if self.status == 'driving':
            if len(self.goals) in self.fail:
                self.status = 'failed'
            elif self.speed is None:
                self.x, self.y = self.goal
                self.status = 'reached'
            else:
                self.drive(self.speed * DT)
        self.yaw = math.atan2(math.sin(self.yaw + self.turn * DT),
                              math.cos(self.yaw + self.turn * DT))

    def drive(self, distance):
        while self.path and distance > 0.0:
            px, py = self.path[0]
            step = math.hypot(px - self.x, py - self.y)
            if step <= distance:
                self.x, self.y = px, py
                self.path.pop(0)
                distance -= step
            else:
                self.x += (px - self.x) * distance / step
                self.y += (py - self.y) * distance / step
                distance = 0.0
        if not self.path:
            self.status = 'reached'


class Team:

    def __init__(self, positions, speed=None, **options):
        planner = make_planner()
        self.robots = {name: FakeRobot(name, x, y, planner=planner, speed=speed)
                       for name, (x, y) in positions.items()}
        self.mission = Mission(planner, sorted(positions), **options)
        self.time = 0.0
        self.feed()

    def feed(self):
        for robot in self.robots.values():
            self.mission.update_pose(robot.name, robot.x, robot.y, robot.yaw)
            self.mission.update_nav(robot.name, robot.status, robot.goal)

    def step(self):
        actions = self.mission.step(self.time)
        for action in actions:
            self.robots[action[1]].act(action)
        for robot in self.robots.values():
            robot.move()
        self.time += DT
        self.feed()
        return actions

    def run(self, seconds, until=None):
        for _ in range(int(seconds / DT)):
            self.step()
            if until is not None and until():
                return True
        return False


TWO = {'tb3_0': (1.0, 1.0), 'tb3_1': (1.0, 3.0)}


def test_waits_for_all_robot_poses():
    planner = make_planner()
    mission = Mission(planner, ['tb3_0', 'tb3_1'])
    mission.update_pose('tb3_0', 1.0, 1.0, 0.0)
    assert mission.step(0.0) == []
    assert not mission.ready
    mission.update_pose('tb3_1', 1.0, 3.0, 0.0)
    actions = mission.step(0.2)
    assert mission.ready
    assert sorted(action[1] for action in actions if action[0] == 'goal') == [
        'tb3_0', 'tb3_1']


def test_robots_search_the_whole_map():
    team = Team(TWO)

    def all_seen():
        return team.mission.coverage.remaining < 0.03 * team.mission.coverage.total

    assert team.run(300.0, until=all_seen)
    coverage = team.mission.coverage
    assert coverage.remaining < 0.03 * coverage.total
    # Both robots took part
    assert all(robot.goals for robot in team.robots.values())


def test_robots_do_not_go_to_the_same_view_point():
    team = Team(TWO)
    for _ in range(50):
        team.step()
        views = [state.view for state in team.mission.robots.values()
                 if state.view is not None]
        assert len(views) == len(set(views))


def test_robots_that_start_together_split_up():
    team = Team({'tb3_0': (1.0, 2.0), 'tb3_1': (1.0, 2.0)})
    team.step()
    goals = [robot.goals[0] for robot in team.robots.values()]
    assert math.dist(goals[0], goals[1]) > 1.5


def test_does_not_go_back_to_a_view_point_it_has_looked_from():
    team = Team(TWO)
    first_pass = {name: [] for name in team.robots}
    while team.mission.passes == 1 and team.time < 300.0:
        for name, robot in team.robots.items():
            first_pass[name] = list(robot.goals)
        team.step()
    assert team.mission.passes == 2
    for goals in first_pass.values():
        assert goals
        assert len(goals) == len(set(goals)), goals


def test_looks_around_at_a_view_point():
    team = Team(TWO)
    looked = team.run(30.0, until=lambda: any(
        state.mode == LOOK for state in team.mission.robots.values()))
    assert looked
    turning = [robot for robot in team.robots.values() if robot.turn != 0.0]
    assert turning


def test_tries_another_view_point_when_one_cannot_be_reached():
    team = Team(TWO)
    team.robots['tb3_0'].fail = {1}
    team.run(1.0)
    robot = team.robots['tb3_0']
    assert len(robot.goals) >= 2
    assert robot.goals[1] != robot.goals[0]
    assert team.mission.banned


def test_sends_a_goal_again_when_it_is_ignored():
    team = Team(TWO, ack_timeout=1.0)
    team.robots['tb3_0'].ignore = 1
    team.run(3.0)
    assert team.robots['tb3_0'].goals


def test_meets_at_the_big_fire_and_goes_back_to_searching():
    team = Team(TWO)
    team.run(5.0)
    target = (4.5, 2.0)
    team.mission.marker_accepted(4, (target[0], target[1], 0.0))
    team.step()
    modes = [state.mode for state in team.mission.robots.values()]
    assert modes == [MEET, MEET]
    for robot in team.robots.values():
        assert robot.turn == 0.0

    assert team.run(10.0, until=lambda: team.mission.meeting_done)
    # Both were within reach of the fire for at least the hold time
    for robot in team.robots.values():
        assert math.hypot(robot.x - target[0], robot.y - target[1]) <= 1.3
    team.step()
    modes = [state.mode for state in team.mission.robots.values()]
    assert all(mode in (EXPLORE, LOOK) for mode in modes)


def meeting_with_robots_held_at(distance):
    """Plan the meeting, then keep both robots this far from the fire, still driving."""
    planner = make_planner()
    mission = Mission(planner, ['tb3_0', 'tb3_1'])
    target = (4.5, 2.0)
    mission.update_pose('tb3_0', 1.0, 1.0, 0.0)
    mission.update_pose('tb3_1', 1.0, 3.0, 0.0)
    mission.step(0.0)
    mission.marker_accepted(4, (target[0], target[1], 0.0))
    time = 0.2
    mission.step(time)
    assert mission.meeting is not None
    for _ in range(20):
        for name, angle in (('tb3_0', 2.5), ('tb3_1', 3.8)):
            mission.update_pose(name, target[0] + distance * math.cos(angle),
                                target[1] + distance * math.sin(angle), 0.0)
            mission.update_nav(name, 'driving', mission.robots[name].goal)
        time += 0.2
        mission.step(time)
        if mission.meeting_done:
            return time
    return None


def test_meeting_is_done_as_soon_as_the_points_are_certain():
    # 1.4 m from the accepted position is always within 2 m of the real marker, so
    # the robots need not drive all the way to their places
    done_after = meeting_with_robots_held_at(1.4)
    assert done_after is not None and done_after < 1.6


def test_meeting_is_not_done_before_the_points_are_certain():
    assert meeting_with_robots_held_at(1.6) is None


def test_meeting_is_planned_again_when_a_robot_cannot_get_there():
    team = Team(TWO)
    team.run(2.0)
    team.mission.marker_accepted(4, (4.5, 2.0, 0.0))
    robot = team.robots['tb3_1']
    robot.fail = {len(robot.goals) + 1}
    assert team.run(15.0, until=lambda: team.mission.meeting_done)


def test_meets_with_only_two_of_three_robots():
    team = Team({'tb3_0': (1.0, 1.0), 'tb3_1': (1.0, 3.0), 'tb3_2': (5.0, 2.0)})
    team.run(2.0)
    team.mission.marker_accepted(4, (4.5, 2.0, 0.0))
    team.step()
    meeting = [state.name for state in team.mission.robots.values() if state.mode == MEET]
    assert len(meeting) == 2
    assert 'tb3_2' in meeting


def test_stops_when_everything_is_done():
    team = Team(TWO)
    team.run(2.0)
    for marker_id in range(5):
        team.mission.marker_accepted(marker_id, (4.5, 2.0, 0.0))
    assert team.run(15.0, until=lambda: team.mission.finished)
    assert all(state.mode == DONE for state in team.mission.robots.values())
    for robot in team.robots.values():
        assert robot.turn == 0.0
        assert robot.status == 'idle'
    assert team.step() == []


def test_stops_searching_when_all_markers_are_found_before_the_meeting():
    team = Team({'tb3_0': (1.0, 1.0), 'tb3_1': (1.0, 3.0), 'tb3_2': (0.5, 2.0)})
    team.run(2.0)
    for marker_id in range(5):
        team.mission.marker_accepted(marker_id, (4.5, 2.0, 0.0))
    team.step()
    states = team.mission.robots
    idle = [state.name for state in states.values() if state.mode == IDLE]
    assert len(idle) == 1


def test_searches_again_when_markers_are_still_missing():
    team = Team(TWO, min_gain=1)
    assert team.run(600.0, until=lambda: team.mission.passes > 1)
    coverage = team.mission.coverage
    assert coverage.remaining > 0.5 * coverage.total


# The helper

# tb3_0 is close to the big fire at (4.5, 2.0), tb3_1 further off, and the helper tb3_2
# waits in the first room
WITH_HELPER = {'tb3_0': (4.0, 1.0), 'tb3_1': (1.0, 3.0), 'tb3_2': (1.0, 1.0)}
FIRE = (4.5, 2.0)


def test_helper_does_not_search():
    team = Team(WITH_HELPER, helpers=['tb3_2'])
    team.run(20.0)
    assert team.robots['tb3_2'].goals == []
    assert team.mission.robots['tb3_2'].mode == PARK
    assert team.robots['tb3_0'].goals and team.robots['tb3_1'].goals


def test_helper_meets_the_robot_closest_to_the_big_fire():
    team = Team(WITH_HELPER, speed=0.25, helpers=['tb3_2'])
    team.run(1.0)
    team.mission.marker_accepted(4, (FIRE[0], FIRE[1], 0.0))
    team.step()
    assert sorted(team.mission.meeting) == ['tb3_0', 'tb3_2']
    assert team.mission.robots['tb3_2'].mode == MEET
    # The helper goes deepest in, since it stays there afterwards
    places = team.mission.meeting
    assert (math.hypot(places['tb3_2'][0] - FIRE[0], places['tb3_2'][1] - FIRE[1])
            < math.hypot(places['tb3_0'][0] - FIRE[0], places['tb3_0'][1] - FIRE[1]))

    assert team.run(60.0, until=lambda: team.mission.meeting_done)
    assert team.mission.robots['tb3_1'].mode != MEET
    team.step()
    assert team.mission.robots['tb3_0'].mode in (EXPLORE, LOOK)
    assert team.mission.robots['tb3_2'].mode == PARK


def test_finder_searches_close_by_while_the_helper_comes():
    team = Team(WITH_HELPER, speed=0.25, helpers=['tb3_2'])
    team.run(1.0)
    finder = team.robots['tb3_0']
    team.mission.marker_accepted(4, (FIRE[0], FIRE[1], 0.0))
    team.step()
    assert team.mission.waiting == 'tb3_0'
    assert team.mission.robots['tb3_0'].mode in (EXPLORE, LOOK)

    searched = len(finder.goals)
    mission = team.mission
    assert team.run(60.0, until=lambda: mission.meeting_done
                    or mission.robots['tb3_0'].mode == MEET)
    assert len(finder.goals) > searched
    if not mission.meeting_done:
        # Called back when the helper was about as far from the fire as the finder
        field = mission.fire_field
        helper = team.robots['tb3_2']
        assert (field.cost_at(helper.x, helper.y)
                <= field.cost_at(finder.x, finder.y) + mission.recall_margin + 0.3)
        assert team.run(60.0, until=lambda: mission.meeting_done)


def test_finder_is_called_back_when_the_helper_is_almost_there():
    # With a long way for the helper, the finder searches and is called back. Every
    # view point it went to was no further from the fire than the helper was then.
    team = Team(WITH_HELPER, speed=0.1, helpers=['tb3_2'])
    team.run(1.0)
    mission = team.mission
    mission.marker_accepted(4, (FIRE[0], FIRE[1], 0.0))
    team.step()
    called_back = False
    for _ in range(int(120.0 / DT)):
        actions = team.step()
        if any(action[0] == 'goal' and action[1] == 'tb3_0' for action in actions):
            if mission.robots['tb3_0'].mode == MEET:
                called_back = True
            elif mission.waiting == 'tb3_0':
                goal = mission.robots['tb3_0'].goal
                assert mission.fire_field.cost_at(*goal) <= mission.helper_cost() + 0.3
        if mission.meeting_done:
            break
    assert mission.meeting_done
    assert called_back


def test_finder_goes_to_the_big_fire_at_once_when_the_helper_is_closer():
    team = Team({'tb3_0': (1.0, 1.0), 'tb3_1': (1.0, 3.0), 'tb3_2': (5.0, 2.0)},
                speed=0.25, helpers=['tb3_2'])
    team.run(1.0)
    team.mission.marker_accepted(4, (FIRE[0], FIRE[1], 0.0))
    team.step()
    assert team.mission.waiting is None
    meeting = sorted(name for name, state in team.mission.robots.items()
                     if state.mode == MEET)
    assert len(meeting) == 2 and 'tb3_2' in meeting


def test_helper_stays_where_it_is_after_the_meeting():
    team = Team(WITH_HELPER, helpers=['tb3_2'])
    team.run(1.0)
    team.mission.marker_accepted(4, (FIRE[0], FIRE[1], 0.0))
    assert team.run(30.0, until=lambda: team.mission.meeting_done)
    helper = team.robots['tb3_2']
    goals = len(helper.goals)
    position = (helper.x, helper.y)
    team.run(30.0)
    assert len(helper.goals) == goals
    assert (helper.x, helper.y) == position


def test_meets_without_the_helper_when_it_cannot_get_there():
    # The helper is shut in the third room
    team = Team({'tb3_0': (4.0, 1.0), 'tb3_1': (1.0, 3.0), 'tb3_2': (7.5, 2.0)},
                helpers=['tb3_2'])
    team.run(1.0)
    team.mission.marker_accepted(4, (FIRE[0], FIRE[1], 0.0))
    team.step()
    assert sorted(team.mission.meeting) == ['tb3_0', 'tb3_1']
    assert team.mission.robots['tb3_2'].mode == PARK


def test_searches_and_meets_without_a_helper_that_never_sends_its_pose():
    # The helper was not spawned, so the mission never gets a pose for it
    planner = make_planner()
    mission = Mission(planner, ['tb3_0', 'tb3_1', 'tb3_2'], helpers=['tb3_2'])
    mission.update_pose('tb3_0', 4.0, 1.0, 0.0)
    mission.update_pose('tb3_1', 1.0, 3.0, 0.0)
    actions = mission.step(0.0)
    assert mission.ready
    assert sorted(action[1] for action in actions if action[0] == 'goal') == [
        'tb3_0', 'tb3_1']
    mission.marker_accepted(4, (FIRE[0], FIRE[1], 0.0))
    mission.step(0.2)
    assert sorted(mission.meeting) == ['tb3_0', 'tb3_1']


def test_robots_standing_still_are_parked():
    team = Team(WITH_HELPER, helpers=['tb3_2'])
    team.step()
    team.robots['tb3_0'].status = 'driving'
    team.robots['tb3_1'].status = 'driving'
    team.feed()
    assert team.mission.parked() == {'tb3_2': (1.0, 1.0)}


if __name__ == '__main__':
    tests = [(name, function) for name, function in sorted(globals().items())
             if name.startswith('test_') and callable(function)]
    failed = 0
    for name, function in tests:
        try:
            function()
            print('ok    ' + name)
        except Exception as error:
            failed += 1
            print('FAIL  ' + name + ': ' + type(error).__name__ + ' ' + str(error))
    print(str(len(tests) - failed) + ' of ' + str(len(tests)) + ' passed')
    raise SystemExit(1 if failed else 0)
