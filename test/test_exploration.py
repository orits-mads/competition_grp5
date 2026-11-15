"""
Tests for exploration.py. They need no ROS.

Run with pytest from the package folder::

    python3 -m pytest test/test_exploration.py

or without pytest::

    PYTHONPATH=. python3 test/test_exploration.py
"""
import math

from competition_grp5.distance_transform import GridMap
from competition_grp5.distance_transform import Planner
from competition_grp5.exploration import CoverageMap
from competition_grp5.exploration import FACE_NORTH
from competition_grp5.exploration import meeting_points
from competition_grp5.exploration import nearest_traversable
from competition_grp5.exploration import ViewPoints

RESOLUTION = 0.05
ROBOT_RADIUS = 0.2


def make_planner():
    """
    Three rooms of 3 m x 4 m side by side, 9 m in all.

    Room 1 (x 0 to 3) and room 2 (x 3 to 6) are joined by a 0.9 m doorway centred on
    y = 2 m. Room 3 (x 6 to 9) is closed off, so nobody can get there.
    """
    width = 180
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
    fill(119, 121, 0, height)
    grid = GridMap.from_data(data, width, height, RESOLUTION)
    return Planner(grid, robot_radius=ROBOT_RADIUS)


START = (1.0, 1.0)


def make_coverage(planner, cell_size=0.25):
    return CoverageMap(planner.grid, [START], cell_size=cell_size)


def cell_at(coverage, x, y):
    return coverage.cell_at(x, y)


def test_coverage_leaves_out_floor_nobody_can_reach():
    planner = make_planner()
    coverage = make_coverage(planner)
    assert coverage.wanted[cell_at(coverage, 1.5, 2.0)]
    assert coverage.wanted[cell_at(coverage, 4.5, 2.0)]
    assert not coverage.wanted[cell_at(coverage, 7.5, 2.0)]
    assert coverage.remaining == coverage.total > 0


def test_visible_stops_at_walls():
    planner = make_planner()
    coverage = make_coverage(planner)
    cells = coverage.visible(1.5, 0.5, 4.0)
    assert cell_at(coverage, 2.5, 0.5) in cells
    # Straight behind the dividing wall
    assert cell_at(coverage, 4.5, 0.5) not in cells
    # Through the doorway
    cells = coverage.visible(1.5, 2.0, 4.0)
    assert cell_at(coverage, 4.5, 2.0) in cells


def test_the_two_sides_of_a_thin_wall_are_different_cells():
    # A wall one grid cell thick in the middle of a coverage square
    width = 40
    height = 20
    data = [0] * (width * height)
    for row in range(height):
        data[row * width + 22] = 100
    grid = GridMap.from_data(data, width, height, RESOLUTION)
    coverage = CoverageMap(grid, [(0.5, 0.5), (1.8, 0.5)], cell_size=0.25)
    left = coverage.cell_at(1.06, 0.5)
    right = coverage.cell_at(1.17, 0.5)
    assert coverage.square_at(1.06, 0.5) == coverage.square_at(1.17, 0.5)
    assert left >= 0 and right >= 0 and left != right
    assert coverage.cell_at(1.12, 0.5) == -1
    coverage.mark_seen(0.5, 0.5, 0.0, 2.0, 1.0)
    assert coverage.seen[left]
    assert not coverage.seen[right]
    assert coverage.occupancy_data()[coverage.square_at(1.1, 0.5)] == 100


def make_room_behind_wall():
    """
    A 6 m x 4 m room split by a wall along y = 2 m, open only at x 4.5 to 6 m.

    The same layout as marker 1 in world w3: the marker hangs on the north side of the
    wall and can only be seen through the opening, almost edge-on.
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
    fill(0, 90, 39, 41)
    grid = GridMap.from_data(data, width, height, RESOLUTION)
    return CoverageMap(grid, [(1.0, 1.0)], cell_size=0.25)


def test_a_wall_seen_from_the_side_is_not_seen():
    coverage = make_room_behind_wall()
    marker = coverage.cell_at(2.1, 2.1)
    assert coverage.need[marker] & FACE_NORTH
    # From the opening, 3 m away and about 85 degrees off to the side
    side = coverage.visible(5.2, 2.35, 4.0)
    assert marker in side
    assert not side[marker] & FACE_NORTH
    coverage.mark_seen(5.2, 2.35, math.pi, 4.0, 1.085)
    assert not coverage.seen[marker]
    # From in front of the wall
    coverage.mark_seen(2.1, 3.5, -0.5 * math.pi, 4.0, 1.085)
    assert coverage.seen[marker]


def test_view_points_go_in_front_of_a_wall():
    coverage = make_room_behind_wall()
    planner = Planner(coverage.grid, robot_radius=ROBOT_RADIUS)
    field = planner.field(1.0, 1.0)
    views = ViewPoints(planner, coverage, [field], spacing=0.5, view_range=2.5)
    marker = coverage.cell_at(2.1, 2.1)
    # Look around from every view point below the wall and in the opening
    for view, (x, y) in enumerate(views.points):
        if y < 2.0 or x > 4.5:
            coverage.mark_seen(x, y, 0.0, 2.5, 2.0 * math.pi)
    assert not coverage.seen[marker]
    # Some view point in the room above still shows it, so the search goes in
    assert any(marker in views.unseen(view, coverage) for view in range(len(views)))


def test_visible_respects_range():
    planner = make_planner()
    coverage = make_coverage(planner)
    cells = coverage.visible(0.5, 2.0, 1.0)
    assert cell_at(coverage, 1.25, 2.0) in cells
    assert cell_at(coverage, 2.25, 2.0) not in cells


def test_mark_seen_only_marks_what_is_in_front():
    planner = make_planner()
    coverage = make_coverage(planner)
    new = coverage.mark_seen(1.5, 2.0, 0.0, 2.0, 1.0)
    assert new > 0
    assert coverage.remaining == coverage.total - new
    assert coverage.seen[cell_at(coverage, 2.5, 2.0)]
    assert not coverage.seen[cell_at(coverage, 0.5, 2.0)]
    # Seeing the same again is nothing new
    assert coverage.mark_seen(1.5, 2.0, 0.0, 2.0, 1.0) == 0


def test_reset_forgets_what_was_seen():
    planner = make_planner()
    coverage = make_coverage(planner)
    coverage.mark_seen(1.5, 2.0, 0.0, 2.0, 1.0)
    coverage.reset()
    assert coverage.remaining == coverage.total
    assert not any(coverage.seen)


def test_occupancy_data_marks_seen_unseen_and_unwanted():
    planner = make_planner()
    coverage = make_coverage(planner)
    coverage.mark_seen(1.5, 2.0, 0.0, 2.0, 1.0)
    data = coverage.occupancy_data()
    assert len(data) == coverage.width * coverage.height
    assert data[coverage.square_at(2.5, 2.0)] == 0
    assert data[coverage.square_at(4.5, 2.0)] == 100
    assert data[coverage.square_at(7.5, 2.0)] == -1


def make_view_points(planner, coverage, view_range=2.5):
    field = planner.field(*START)
    return ViewPoints(planner, coverage, [field], spacing=0.5, view_range=view_range)


def test_view_points_are_reachable_and_cover_the_floor():
    planner = make_planner()
    coverage = make_coverage(planner)
    views = make_view_points(planner, coverage)
    assert len(views) > 10
    for x, y in views.points:
        assert x < 6.0, 'view point in the closed room'
        index = planner.grid.index(*planner.grid.world_to_cell(x, y))
        assert planner.traversable[index]
    covered = set()
    for view in views.views:
        covered.update(view)
    wanted = [cell for cell, is_wanted in enumerate(coverage.wanted) if is_wanted]
    assert len(covered) >= 0.98 * len(wanted)


def test_choose_goes_where_there_is_something_new_to_see():
    planner = make_planner()
    coverage = make_coverage(planner)
    views = make_view_points(planner, coverage)
    # Everything in room 1 has been seen
    for cell in range(len(coverage.seen)):
        if coverage.cell_center(cell)[0] < 3.0:
            coverage.seen[cell] = 1
    field = planner.field(1.0, 2.0)
    view = views.choose(coverage, field)
    assert view is not None
    assert views.gain(view, coverage) >= 4
    # The chosen point must show part of room 2
    assert any(coverage.cell_center(cell)[0] > 3.0 for cell in views.views[view]
               if not coverage.seen[cell])


def test_choose_prefers_the_nearer_of_two_equal_view_points():
    planner = make_planner()
    coverage = make_coverage(planner)
    views = make_view_points(planner, coverage)
    field = planner.field(1.0, 2.0)
    view = views.choose(coverage, field)
    x, y = views.points[view]
    assert x < 3.0, 'chose a view point in the far room before the near one'


def test_choose_leaves_out_claimed_cells_and_banned_points():
    planner = make_planner()
    coverage = make_coverage(planner)
    views = make_view_points(planner, coverage)
    field = planner.field(1.0, 2.0)
    everything = set(cell for view in views.views for cell in view)
    assert views.choose(coverage, field, claimed=everything) is None
    assert views.choose(coverage, field, banned=set(range(len(views)))) is None
    first = views.choose(coverage, field)
    second = views.choose(coverage, field, banned={first})
    assert second is not None and second != first


def test_choose_only_looks_within_max_cost():
    planner = make_planner()
    coverage = make_coverage(planner)
    views = make_view_points(planner, coverage)
    field = planner.field(1.0, 2.0)
    view = views.choose(coverage, field, max_cost=1.0)
    assert view is not None
    assert field.cost[views.cells[view]] <= 1.0
    assert views.choose(coverage, field, max_cost=-1.0) is None


def test_choose_spreads_away_from_where_others_are_going():
    planner = make_planner()
    coverage = make_coverage(planner)
    views = make_view_points(planner, coverage)
    field = planner.field(1.0, 2.0)
    first = views.choose(coverage, field)
    taken = views.points[first]
    second = views.choose(coverage, field, claimed=set(views.views[first]),
                          banned={first}, avoid=[taken], spread=3.0)
    assert second is not None
    without = views.choose(coverage, field, claimed=set(views.views[first]),
                           banned={first})
    assert (math.dist(views.points[second], taken)
            >= math.dist(views.points[without], taken))
    assert math.dist(views.points[second], taken) > 1.0


def test_choose_returns_none_when_everything_is_seen():
    planner = make_planner()
    coverage = make_coverage(planner)
    views = make_view_points(planner, coverage)
    coverage.seen = bytearray([1] * len(coverage.seen))
    assert views.choose(coverage, planner.field(1.0, 2.0)) is None


def test_nearest_traversable_comes_out_of_a_wall():
    planner = make_planner()
    cell = nearest_traversable(planner, 3.0, 0.8, 1.0)
    assert cell is not None
    assert planner.traversable[planner.grid.index(*cell)]
    assert nearest_traversable(planner, 7.5, 2.0, 0.1) is not None
    # Inside the thick part of the wall, with a radius too small to get out
    assert nearest_traversable(planner, 3.0, 0.8, 0.05) is None


def check_places(planner, target, places, radius, separation):
    for x, y in places.values():
        assert math.hypot(x - target[0], y - target[1]) <= radius + 1e-9
        index = planner.grid.index(*planner.grid.world_to_cell(x, y))
        assert planner.traversable[index]
    points = list(places.values())
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            assert math.dist(points[i], points[j]) >= separation - 1e-9


def test_meeting_points_for_two_robots():
    planner = make_planner()
    target = (4.5, 2.0)
    places = meeting_points(planner, target, [('a', (1.0, 1.0)), ('b', (1.0, 3.0))],
                            radius=1.2, separation=0.6)
    assert set(places) == {'a', 'b'}
    check_places(planner, target, places, 1.2, 0.6)


def test_meeting_points_picks_the_two_nearest_robots():
    planner = make_planner()
    target = (4.5, 2.0)
    robots = [('far', (0.5, 0.5)), ('near', (5.0, 1.0)), ('middle', (3.5, 3.0))]
    places = meeting_points(planner, target, robots, radius=1.2, separation=0.6)
    assert set(places) == {'near', 'middle'}
    check_places(planner, target, places, 1.2, 0.6)


def test_meeting_points_in_the_given_order():
    planner = make_planner()
    target = (4.5, 2.0)
    robots = [('far', (0.5, 0.5)), ('near', (5.0, 1.0)), ('middle', (3.5, 3.0))]
    places = meeting_points(planner, target, robots, radius=1.2, separation=0.6,
                            keep_order=True)
    assert set(places) == {'far', 'near'}
    check_places(planner, target, places, 1.2, 0.6)
    # The first one given goes deepest in, although the other one is closer
    depth = {name: math.hypot(x - target[0], y - target[1])
             for name, (x, y) in places.items()}
    assert depth['far'] <= 1.2 - 0.6 + 0.05
    assert depth['far'] < depth['near']


def test_meeting_points_from_the_same_side_do_not_overlap():
    planner = make_planner()
    target = (5.0, 2.0)
    robots = [('a', (1.0, 2.0)), ('b', (0.5, 2.0))]
    places = meeting_points(planner, target, robots, radius=1.2, separation=0.6)
    assert set(places) == {'a', 'b'}
    check_places(planner, target, places, 1.2, 0.6)
    # The robot that gets there first goes further in, so the other need not pass it
    assert (math.dist(places['a'], target) < math.dist(places['b'], target))


def test_meeting_points_at_a_marker_on_a_wall():
    planner = make_planner()
    target = (3.0, 0.8)
    places = meeting_points(planner, target, [('a', (1.0, 1.0)), ('b', (5.0, 3.0))],
                            radius=1.2, separation=0.6)
    assert set(places) == {'a', 'b'}
    check_places(planner, target, places, 1.2, 0.6)


def test_meeting_points_without_a_way_there():
    planner = make_planner()
    places = meeting_points(planner, (7.5, 2.0), [('a', (1.0, 1.0)), ('b', (1.0, 3.0))],
                            radius=1.2, separation=0.6)
    assert places == {}


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
