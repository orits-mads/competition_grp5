"""
Tests for marker_tracking.py. They need no ROS.

Run with pytest from the package folder::

    python3 -m pytest test/test_marker_tracking.py

or without pytest::

    PYTHONPATH=. python3 test/test_marker_tracking.py
"""
import math
import random

from competition_grp5.marker_tracking import MarkerEstimator
from competition_grp5.marker_tracking import median
from competition_grp5.marker_tracking import ReportBook
from competition_grp5.marker_tracking import rotate_vector
from competition_grp5.marker_tracking import transform_point


def close(a, b, tolerance=1e-9):
    return all(abs(x - y) < tolerance for x, y in zip(a, b))


def yaw_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def multiply(a, b):
    """Quaternion product a * b, both as (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def test_rotate_vector():
    assert close(rotate_vector((0.0, 0.0, 0.0, 1.0), (1.0, 2.0, 3.0)), (1.0, 2.0, 3.0))
    # A quarter turn to the left about z takes x to y
    assert close(rotate_vector(yaw_quaternion(math.pi / 2), (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))
    assert close(rotate_vector(yaw_quaternion(math.pi / 2), (0.0, 1.0, 5.0)), (-1.0, 0.0, 5.0))
    # A quarter turn about x takes y to z
    quarter_x = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))
    assert close(rotate_vector(quarter_x, (0.0, 1.0, 0.0)), (0.0, 0.0, 1.0))
    # Rotation keeps the length
    rotation = multiply(yaw_quaternion(0.7), quarter_x)
    rotated = rotate_vector(rotation, (0.3, -1.2, 2.0))
    assert abs(math.hypot(*rotated) - math.hypot(0.3, -1.2, 2.0)) < 1e-9


def test_transform_point():
    assert close(transform_point((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0)),
                 (1.5, 2.0, 3.0))
    # Robot at (2, 1) facing left (+y): a point 3 m ahead of it is at (2, 4)
    assert close(transform_point((2.0, 1.0, 0.0), yaw_quaternion(math.pi / 2),
                                 (3.0, 0.0, 0.0)), (2.0, 4.0, 0.0))


def test_transform_from_camera_optical_frame():
    # The optical frame has z forwards, x to the right and y down. Relative to a frame
    # with x forwards, y left and z up, that is the rotation below.
    optical = (-0.5, 0.5, -0.5, 0.5)
    assert close(rotate_vector(optical, (0.0, 0.0, 1.0)), (1.0, 0.0, 0.0))
    assert close(rotate_vector(optical, (1.0, 0.0, 0.0)), (0.0, -1.0, 0.0))
    assert close(rotate_vector(optical, (0.0, 1.0, 0.0)), (0.0, 0.0, -1.0))
    # Camera 0.1 m above the floor at (1, 1), robot facing +y. A marker 2 m ahead,
    # 0.5 m to the right and 0.15 m above the camera is at (1.5, 3, 0.25) on the map.
    rotation = multiply(yaw_quaternion(math.pi / 2), optical)
    point = transform_point((1.0, 1.0, 0.1), rotation, (0.5, -0.15, 2.0))
    assert close(point, (1.5, 3.0, 0.25))


def test_median():
    assert median([3.0]) == 3.0
    assert median([5.0, 1.0, 3.0]) == 3.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_estimator_waits_for_enough_observations():
    estimator = MarkerEstimator(min_observations=3)
    assert estimator.estimate(2) is None
    assert estimator.add(2, (1.0, 2.0, 0.25), 2.0)
    assert estimator.add(2, (1.1, 2.0, 0.25), 2.0)
    assert estimator.estimate(2) is None
    assert estimator.add(2, (1.2, 2.1, 0.25), 2.0)
    assert close(estimator.estimate(2), (1.1, 2.0, 0.25))
    assert estimator.estimate(3) is None
    assert estimator.count(2) == 3


def test_estimator_throws_away_bad_measurements():
    estimator = MarkerEstimator(max_range=4.0)
    # Ids that are not part of the competition
    assert not estimator.add(5, (1.0, 1.0, 0.0), 1.0)
    assert not estimator.add(-1, (1.0, 1.0, 0.0), 1.0)
    assert not estimator.add(17, (1.0, 1.0, 0.0), 1.0)
    # Too far away, or behind the camera
    assert not estimator.add(0, (1.0, 1.0, 0.0), 4.5)
    assert not estimator.add(0, (1.0, 1.0, 0.0), 0.0)
    assert not estimator.add(0, (1.0, 1.0, 0.0), -1.0)
    # Broken numbers
    assert not estimator.add(0, (float('nan'), 1.0, 0.0), 1.0)
    assert not estimator.add(0, (1.0, float('inf'), 0.0), 1.0)
    assert estimator.count(0) == 0
    for marker_id in range(5):
        assert estimator.add(marker_id, (1.0, 1.0, 0.0), 1.0)


def test_estimator_ignores_an_outlier():
    estimator = MarkerEstimator()
    for _ in range(6):
        estimator.add(1, (3.0, -2.0, 0.25), 1.5)
    estimator.add(1, (9.0, 9.0, 4.0), 1.4)
    assert close(estimator.estimate(1), (3.0, -2.0, 0.25))


def test_estimator_prefers_close_measurements():
    estimator = MarkerEstimator(keep=5)
    # Seen from far away first: the estimate is 0.4 m off
    for _ in range(8):
        estimator.add(0, (2.4, 1.0, 0.25), 3.5)
    assert close(estimator.estimate(0), (2.4, 1.0, 0.25))
    assert estimator.count(0) == 5
    # Three closer and better measurements now outnumber the two far ones left
    for _ in range(3):
        estimator.add(0, (2.0, 1.0, 0.25), 1.0)
    assert close(estimator.estimate(0), (2.0, 1.0, 0.25))
    # A far measurement no longer gets in
    estimator.add(0, (2.4, 1.0, 0.25), 3.9)
    for _ in range(2):
        estimator.add(0, (2.0, 1.0, 0.25), 1.0)
    assert all(distance == 1.0 for distance, _, _, _ in estimator._observations[0])


def test_estimator_with_noisy_measurements_is_close_to_the_truth():
    generator = random.Random(4)
    estimator = MarkerEstimator()
    truth = (4.0, -1.5, 0.25)
    for step in range(60):
        distance = 4.0 - 0.05 * step
        noise = 0.05 * distance
        position = tuple(value + generator.gauss(0.0, noise) for value in truth)
        estimator.add(3, position, distance)
    assert math.dist(estimator.estimate(3), truth) < 0.1


def test_report_book_reports_each_new_estimate_once():
    book = ReportBook()
    assert book.next_report(0.0) is None
    book.add_estimate(2, 'tb3_0', (1.0, 2.0, 0.0))
    report = book.next_report(0.0)
    assert report == (2, 'tb3_0', (1.0, 2.0, 0.0))
    book.reported(*report, 0.0)
    # Not accepted, and nothing new to say
    assert book.result(2, 'tb3_0', (1.0, 2.0, 0.0), False) is False
    assert book.next_report(5.0) is None
    # The same estimate again, or one that barely moved, is not news
    book.add_estimate(2, 'tb3_0', (1.01, 2.0, 0.0))
    assert book.next_report(5.0) is None
    # A better estimate is reported
    book.add_estimate(2, 'tb3_0', (1.3, 2.0, 0.0))
    assert book.next_report(5.0) == (2, 'tb3_0', (1.3, 2.0, 0.0))


def test_report_book_waits_between_reports_of_the_same_marker():
    book = ReportBook(retry_interval=1.0)
    book.add_estimate(0, 'tb3_0', (1.0, 1.0, 0.0))
    book.reported(*book.next_report(10.0), 10.0)
    book.add_estimate(0, 'tb3_0', (1.5, 1.0, 0.0))
    assert book.next_report(10.5) is None
    assert book.next_report(11.1) == (0, 'tb3_0', (1.5, 1.0, 0.0))


def test_report_book_stops_when_a_marker_is_accepted():
    book = ReportBook()
    book.add_estimate(4, 'tb3_1', (3.0, 3.0, 0.0))
    report = book.next_report(0.0)
    book.reported(*report, 0.0)
    assert book.result(*report, True) is True
    assert book.accepted == {4: ((3.0, 3.0, 0.0), 'tb3_1')}
    assert book.missing() == [0, 1, 2, 3]
    # Later estimates of the same marker are ignored, from any robot
    book.add_estimate(4, 'tb3_1', (3.4, 3.0, 0.0))
    book.add_estimate(4, 'tb3_0', (2.9, 3.1, 0.0))
    assert book.next_report(60.0) is None
    # A second answer for the same marker does not count as a new find
    assert book.result(4, 'tb3_0', (2.9, 3.1, 0.0), True) is False
    assert book.accepted[4] == ((3.0, 3.0, 0.0), 'tb3_1')


def test_report_book_uses_estimates_from_both_robots():
    book = ReportBook(retry_interval=1.0)
    book.add_estimate(1, 'tb3_0', (5.0, 5.0, 0.0))
    book.add_estimate(1, 'tb3_1', (5.6, 5.0, 0.0))
    first = book.next_report(0.0)
    book.reported(*first, 0.0)
    book.result(*first, False)
    second = book.next_report(1.5)
    assert second is not None
    assert second[1] != first[1]
    book.reported(*second, 1.5)
    assert book.result(*second, True) is True
    assert book.accepted[1][1] == second[1]


def test_report_book_takes_the_marker_that_waited_longest():
    book = ReportBook(retry_interval=1.0)
    book.add_estimate(0, 'tb3_0', (1.0, 0.0, 0.0))
    book.add_estimate(3, 'tb3_0', (2.0, 0.0, 0.0))
    first = book.next_report(0.0)
    book.reported(*first, 0.0)
    second = book.next_report(0.2)
    assert second[0] != first[0]
    book.reported(*second, 0.2)
    assert {first[0], second[0]} == {0, 3}
    assert book.next_report(0.4) is None


def test_report_book_sends_an_unchanged_estimate_again_after_a_while():
    book = ReportBook(resend_after=10.0)
    book.add_estimate(0, 'tb3_0', (1.0, 1.0, 0.0))
    book.reported(*book.next_report(0.0), 0.0)
    assert book.next_report(9.0) is None
    assert book.next_report(10.5) == (0, 'tb3_0', (1.0, 1.0, 0.0))


def test_report_book_ignores_unknown_ids():
    book = ReportBook()
    book.add_estimate(7, 'tb3_0', (1.0, 1.0, 0.0))
    assert book.next_report(0.0) is None
    assert book.missing() == [0, 1, 2, 3, 4]


if __name__ == '__main__':
    tests = [(name, function) for name, function in sorted(globals().items())
             if name.startswith('test_') and callable(function)]
    failed = 0
    for name, function in tests:
        try:
            function()
            print('ok    ' + name)
        # Fanger alt, så én feil ikke stopper resten av testene
        except Exception as error:  # noqa: B902
            failed += 1
            print('FAIL  ' + name + ': ' + type(error).__name__ + ' ' + str(error))
    print(str(len(tests) - failed) + ' of ' + str(len(tests)) + ' passed')
    raise SystemExit(1 if failed else 0)
