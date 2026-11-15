"""
Tests for path_follower.py. They need no ROS.

The robot is simulated as a point that drives forwards and turns, with limited
acceleration. Run with pytest from the package folder::

    python3 -m pytest test/test_path_follower.py

or without pytest::

    PYTHONPATH=. python3 test/test_path_follower.py
"""
import math

from competition_grp5.distance_transform import GridMap
from competition_grp5.distance_transform import path_length
from competition_grp5.distance_transform import Planner
from competition_grp5.path_follower import DRIVING
from competition_grp5.path_follower import FAILED
from competition_grp5.path_follower import IDLE
from competition_grp5.path_follower import INF
from competition_grp5.path_follower import Navigator
from competition_grp5.path_follower import normalize_angle
from competition_grp5.path_follower import PathFollower
from competition_grp5.path_follower import REACHED
from competition_grp5.path_follower import strip_clearance
from competition_grp5.path_follower import yaw_from_quaternion

RESOLUTION = 0.05
ROBOT_RADIUS = 0.27


def make_planner():
    """
    Two rooms of 3 m x 4 m side by side, joined by a 0.9 m wide doorway.

    The outer walls run along the map edge, the dividing wall is at x = 3 m and the
    doorway is centred on y = 2 m.
    """
    width = 120
    height = 80
    data = [0] * (width * height)

    def fill(col0, col1, row0, row1):
        for row in range(row0, row1):
            for col in range(col0, col1):
                data[row * width + col] = 100

    fill(0, width, 0, 2)
    fill(0, width, height - 2, height)
    fill(0, 2, 0, height)
    fill(width - 2, width, 0, height)
    fill(59, 61, 0, 31)
    fill(59, 61, 49, height)
    grid = GridMap.from_data(data, width, height, RESOLUTION)
    return Planner(grid, robot_radius=ROBOT_RADIUS)


def make_navigator(planner=None, **options):
    return Navigator(planner or make_planner(), PathFollower(), **options)


class Robot:
    """A point robot with the acceleration limits of the simulated TurtleBot."""

    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = 0.0
        self.turn = 0.0
        self.frozen = False

    def step(self, speed, turn, dt):
        if self.frozen:
            return
        self.speed += max(-1.0 * dt, min(1.0 * dt, speed - self.speed))
        self.turn += max(-4.0 * dt, min(4.0 * dt, turn - self.turn))
        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt
        self.yaw = normalize_angle(self.yaw + self.turn * dt)


def drive(navigator, robot, duration, start_time=0.0, front=INF, rear=INF, dt=0.05):
    """
    Run the navigator on the simulated robot until it stops driving or time runs out.

    front and rear are either a fixed clearance or a function of time. Returns the end
    time and the list of (time, speed command, turn command, x, y).
    """
    now = start_time
    log = []
    while now < start_time + duration and navigator.status == DRIVING:
        front_now = front(now) if callable(front) else front
        rear_now = rear(now) if callable(rear) else rear
        speed, turn = navigator.update(now, robot.x, robot.y, robot.yaw, front_now, rear_now)
        log.append((now, speed, turn, robot.x, robot.y))
        robot.step(speed, turn, dt)
        now += dt
    return now, log


def clearance_at(planner, x, y):
    return planner.clearance[planner.grid.index(*planner.grid.world_to_cell(x, y))]


def test_normalize_angle():
    assert abs(normalize_angle(0.5) - 0.5) < 1e-12
    assert abs(normalize_angle(2 * math.pi + 0.5) - 0.5) < 1e-12
    assert abs(normalize_angle(-math.pi - 0.1) - (math.pi - 0.1)) < 1e-12


def test_yaw_from_quaternion():
    for yaw in (0.0, 0.7, -2.5, math.pi / 2):
        z = math.sin(yaw / 2)
        w = math.cos(yaw / 2)
        assert abs(yaw_from_quaternion(0.0, 0.0, z, w) - yaw) < 1e-9


def lidar_scan(points):
    """Return 360 ranges, one per degree, all out of range except the given degrees."""
    ranges = [INF] * 360
    for degree, distance in points.items():
        ranges[degree % 360] = distance
    return ranges


def test_strip_clearance_sees_what_is_in_the_way():
    step = math.radians(1.0)
    # Straight ahead
    assert abs(strip_clearance(lidar_scan({0: 0.5}), 0.0, step, 0.18) - 0.5) < 1e-9
    # 20 degrees to the left at 0.4 m is 0.137 m off the centre line: in the strip
    expected = 0.4 * math.cos(math.radians(20))
    assert abs(strip_clearance(lidar_scan({20: 0.4}), 0.0, step, 0.18) - expected) < 1e-9
    # The nearest of several points counts
    scan = lidar_scan({0: 0.9, 355: 0.6, 10: 0.7})
    expected = 0.6 * math.cos(math.radians(5))
    assert abs(strip_clearance(scan, 0.0, step, 0.18) - expected) < 1e-9


def test_strip_clearance_ignores_what_is_beside_or_behind():
    step = math.radians(1.0)
    # A door frame right beside the robot
    assert strip_clearance(lidar_scan({90: 0.2, 270: 0.2}), 0.0, step, 0.18) == INF
    # 40 degrees to the side at 0.5 m is 0.32 m off the centre line
    assert strip_clearance(lidar_scan({40: 0.5}), 0.0, step, 0.18) == INF
    # Behind the robot
    assert strip_clearance(lidar_scan({180: 0.3}), 0.0, step, 0.18) == INF
    assert abs(strip_clearance(lidar_scan({180: 0.3}), 0.0, step, 0.18,
                               backwards=True) - 0.3) < 1e-9
    assert strip_clearance(lidar_scan({0: 0.3}), 0.0, step, 0.18, backwards=True) == INF


def test_strip_clearance_ignores_bad_readings():
    step = math.radians(1.0)
    ranges = lidar_scan({0: float('nan'), 1: 0.05, 2: 0.0})
    assert strip_clearance(ranges, 0.0, step, 0.18, range_min=0.12) == INF
    # A scan that starts at -pi instead of 0 gives the same answer
    ranges = [INF] * 360
    ranges[180] = 0.5
    assert abs(strip_clearance(ranges, -math.pi, step, 0.18) - 0.5) < 1e-9


def test_follower_turns_on_the_spot_when_facing_away():
    follower = PathFollower()
    follower.set_path([(0.1 * i, 0.0) for i in range(30)])
    speed, turn = follower.update(0.0, 0.0, math.pi)
    assert speed == 0.0
    assert abs(turn) == follower.max_turn
    speed, turn = follower.update(0.0, 0.0, 0.0)
    assert abs(speed - follower.max_speed) < 1e-9
    assert abs(turn) < 1e-9
    # Path to the left of the heading: turn left (positive)
    speed, turn = follower.update(0.0, 0.0, -0.3)
    assert turn > 0.0
    assert 0.0 < speed < follower.max_speed


def test_follower_slows_down_and_stops_at_the_end():
    follower = PathFollower()
    follower.set_path([(0.1 * i, 0.0) for i in range(30)])
    far, _ = follower.update(1.0, 0.0, 0.0)
    near, _ = follower.update(2.75, 0.0, 0.0)
    assert near < far
    assert not follower.done
    assert follower.update(2.85, 0.0, 0.0) == (0.0, 0.0)
    assert follower.done


def test_follower_without_path_is_done():
    follower = PathFollower()
    assert follower.done
    assert follower.update(0.0, 0.0, 0.0) == (0.0, 0.0)
    follower.set_path([])
    assert follower.done


def test_drives_through_the_doorway_to_the_goal():
    planner = make_planner()
    navigator = make_navigator(planner)
    robot = Robot(1.0, 1.0, 0.0)
    goal = (5.0, 3.0)
    assert navigator.set_goal(goal, robot.x, robot.y, 0.0)
    assert navigator.status == DRIVING
    length = path_length(navigator.path)

    end, log = drive(navigator, robot, 120.0)

    assert navigator.status == REACHED
    assert math.hypot(robot.x - goal[0], robot.y - goal[1]) < 0.15
    # Never closer to a wall than the robot's own half width (0.153 m), with margin
    assert min(clearance_at(planner, x, y) for _, _, _, x, y in log) > 0.2
    # Went through the doorway at x = 3 m
    assert any(2.9 < x < 3.1 and 1.55 < y < 2.45 for _, _, _, x, y in log)
    # Not much slower than driving the path at full speed
    assert end < 1.5 * length / navigator.follower.max_speed + 5.0
    # Commands stay within the limits
    assert all(0.0 <= speed <= 0.26 + 1e-9 for _, speed, _, _, _ in log)
    assert all(abs(turn) <= 1.5 + 1e-9 for _, _, turn, _, _ in log)


def test_reaches_goal_from_any_start_heading():
    planner = make_planner()
    for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2, 2.5):
        navigator = make_navigator(planner)
        robot = Robot(5.2, 0.8, yaw)
        assert navigator.set_goal((0.8, 3.2), robot.x, robot.y, 0.0)
        _, log = drive(navigator, robot, 120.0)
        assert navigator.status == REACHED
        assert math.hypot(robot.x - 0.8, robot.y - 3.2) < 0.15
        assert min(clearance_at(planner, x, y) for _, _, _, x, y in log) > 0.2


def test_goal_at_robot_position_is_reached_at_once():
    navigator = make_navigator()
    assert navigator.set_goal((1.0, 1.0), 1.02, 1.01, 0.0)
    assert navigator.update(0.0, 1.02, 1.01, 0.0) == (0.0, 0.0)
    assert navigator.status == REACHED


def test_unreachable_goal_fails():
    navigator = make_navigator()
    # Far outside the map
    assert not navigator.set_goal((50.0, 50.0), 1.0, 1.0, 0.0)
    assert navigator.status == FAILED
    assert navigator.update(0.0, 1.0, 1.0, 0.0) == (0.0, 0.0)


def test_cancel_stops_the_robot():
    navigator = make_navigator()
    assert navigator.status == IDLE
    navigator.set_goal((5.0, 3.0), 1.0, 1.0, 0.0)
    assert navigator.update(0.0, 1.0, 1.0, 0.0)[0] > 0.0
    navigator.cancel()
    assert navigator.status == IDLE
    assert navigator.update(0.1, 1.0, 1.0, 0.0) == (0.0, 0.0)


def test_slows_down_near_an_obstacle():
    navigator = make_navigator()
    navigator.set_goal((2.5, 1.0), 1.0, 1.0, 0.0)
    clear, _ = navigator.update(0.0, 1.0, 1.0, 0.0, front=INF)
    close, _ = navigator.update(0.05, 1.0, 1.0, 0.0, front=0.45)
    assert 0.0 < close < clear


def test_waits_for_an_obstacle_that_moves_away():
    navigator = make_navigator()
    robot = Robot(1.0, 1.0, 0.0)
    navigator.set_goal((2.5, 1.0), robot.x, robot.y, 0.0)
    # Something is in the way for the first two seconds, for instance the other robot
    _, log = drive(navigator, robot, 60.0, front=lambda now: 0.2 if now < 2.0 else INF)
    assert navigator.status == REACHED
    assert all(speed == 0.0 for now, speed, _, _, _ in log if now < 2.0)
    assert all(speed >= 0.0 for _, speed, _, _, _ in log)


def test_backs_up_and_gives_up_when_blocked_for_good():
    navigator = make_navigator()
    robot = Robot(1.5, 1.0, 0.0)
    navigator.set_goal((2.5, 1.0), robot.x, robot.y, 0.0)
    end, log = drive(navigator, robot, 120.0, front=0.2)
    assert navigator.status == FAILED
    assert all(speed <= 0.0 for _, speed, _, _, _ in log)
    assert any(speed < 0.0 for _, speed, _, _, _ in log)
    # Three recoveries of 3 s waiting and 1.5 s backing up, then a last 3 s wait
    assert 14.0 < end < 20.0
    assert navigator.update(end, robot.x, robot.y, robot.yaw) == (0.0, 0.0)


def test_does_not_back_up_into_something():
    navigator = make_navigator()
    robot = Robot(1.5, 1.0, 0.0)
    navigator.set_goal((2.5, 1.0), robot.x, robot.y, 0.0)
    _, log = drive(navigator, robot, 120.0, front=0.2, rear=0.2)
    assert navigator.status == FAILED
    assert all(speed == 0.0 for _, speed, _, _, _ in log)


def test_recovers_when_the_robot_does_not_move():
    navigator = make_navigator()
    robot = Robot(1.0, 1.0, 0.0)
    navigator.set_goal((2.5, 1.0), robot.x, robot.y, 0.0)
    # Wheels spin against something the lidar cannot see for the first 12 seconds
    robot.frozen = True
    now, log = drive(navigator, robot, 12.0)
    assert navigator.status == DRIVING
    assert any(speed < 0.0 for _, speed, _, _, _ in log)
    assert all(speed >= 0.0 for now_, speed, _, _, _ in log if now_ < 7.9)
    robot.frozen = False
    drive(navigator, robot, 60.0, start_time=now)
    assert navigator.status == REACHED


def test_gives_up_when_stuck_for_good():
    navigator = make_navigator()
    robot = Robot(1.0, 1.0, 0.0)
    robot.frozen = True
    navigator.set_goal((2.5, 1.0), robot.x, robot.y, 0.0)
    drive(navigator, robot, 120.0)
    assert navigator.status == FAILED


def test_plans_again_when_pushed_off_the_path():
    planner = make_planner()
    navigator = make_navigator(planner)
    robot = Robot(1.0, 1.0, 0.0)
    navigator.set_goal((5.0, 3.0), robot.x, robot.y, 0.0)
    now, _ = drive(navigator, robot, 3.0)
    old_path = navigator.path
    # Moved a metre sideways, still in free space
    robot.y += 1.0
    now, _ = drive(navigator, robot, 0.2, start_time=now)
    assert navigator.status == DRIVING
    assert navigator.path != old_path
    assert math.hypot(navigator.path[0][0] - robot.x, navigator.path[0][1] - robot.y) < 0.1
    drive(navigator, robot, 120.0, start_time=now)
    assert navigator.status == REACHED


def test_new_goal_replaces_the_old_one():
    navigator = make_navigator()
    robot = Robot(1.0, 1.0, 0.0)
    navigator.set_goal((5.0, 3.0), robot.x, robot.y, 0.0)
    now, _ = drive(navigator, robot, 4.0)
    assert navigator.set_goal((1.0, 3.2), robot.x, robot.y, now)
    drive(navigator, robot, 120.0, start_time=now)
    assert navigator.status == REACHED
    assert math.hypot(robot.x - 1.0, robot.y - 3.2) < 0.15


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
