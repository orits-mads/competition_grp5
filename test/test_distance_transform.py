"""
Tests for distance_transform.py. They need no ROS.

Run with pytest from the package folder::

    python3 -m pytest test/test_distance_transform.py

or without pytest::

    PYTHONPATH=. python3 test/test_distance_transform.py
"""
import math
import random
from types import SimpleNamespace

from competition_grp5.distance_transform import descend
from competition_grp5.distance_transform import distance_transform
from competition_grp5.distance_transform import GridMap
from competition_grp5.distance_transform import INF
from competition_grp5.distance_transform import NEIGHBOURS
from competition_grp5.distance_transform import obstacle_distance
from competition_grp5.distance_transform import path_length
from competition_grp5.distance_transform import Planner
from competition_grp5.distance_transform import SQRT2


def make_grid(picture, resolution=1.0, origin_x=0.0, origin_y=0.0):
    """
    Build a GridMap from a drawing where '#' is a wall and anything else is free.

    The last line of the drawing is row 0, so the drawing looks like the map seen from
    above with x to the right and y upwards.
    """
    lines = picture[::-1]
    width = len(lines[0])
    occupied = bytearray()
    for line in lines:
        assert len(line) == width
        occupied.extend(1 if char == '#' else 0 for char in line)
    return GridMap(width, len(lines), resolution, origin_x, origin_y, occupied)


def free_cells(grid):
    return bytearray(0 if value else 1 for value in grid.occupied)


def reference_transform(grid, traversable, sources):
    """
    Slow textbook distance transform: sweep the whole grid until nothing changes.

    Written independently of the wavefront version to check it against.
    """
    width = grid.width
    cost = [INF] * (width * grid.height)
    for col, row in sources:
        cost[row * width + col] = 0.0
    changed = True
    while changed:
        changed = False
        for row in range(grid.height):
            for col in range(width):
                index = row * width + col
                if not traversable[index]:
                    continue
                for d_col, d_row, length in NEIGHBOURS:
                    n_col = col + d_col
                    n_row = row + d_row
                    if not grid.in_bounds(n_col, n_row):
                        continue
                    if not traversable[n_row * width + n_col]:
                        continue
                    if d_col != 0 and d_row != 0:
                        if not traversable[row * width + n_col]:
                            continue
                        if not traversable[n_row * width + col]:
                            continue
                    candidate = cost[n_row * width + n_col] + length * grid.resolution
                    if candidate < cost[index] - 1e-12:
                        cost[index] = candidate
                        changed = True
    return cost


def test_world_and_cell_conversion():
    grid = make_grid(['....', '....', '....'], resolution=0.5, origin_x=-1.0, origin_y=2.0)
    assert grid.world_to_cell(-1.0, 2.0) == (0, 0)
    assert grid.world_to_cell(-0.51, 2.49) == (0, 0)
    assert grid.world_to_cell(0.6, 3.1) == (3, 2)
    assert grid.cell_to_world(0, 0) == (-0.75, 2.25)
    # A cell centre converts back to the same cell
    for col in range(grid.width):
        for row in range(grid.height):
            assert grid.world_to_cell(*grid.cell_to_world(col, row)) == (col, row)
    assert not grid.in_bounds(*grid.world_to_cell(-1.01, 2.0))
    assert not grid.in_bounds(*grid.world_to_cell(1.0, 2.0))


def test_from_data_thresholds_and_unknown():
    data = [0, 49, 50, 100, -1, 0]
    grid = GridMap.from_data(data, 3, 2, 0.1)
    assert list(grid.occupied) == [0, 0, 1, 1, 1, 0]
    grid = GridMap.from_data(data, 3, 2, 0.1, unknown_is_occupied=False)
    assert list(grid.occupied) == [0, 0, 1, 1, 0, 0]


def test_from_data_merges_cells_and_keeps_walls():
    # 4 x 4 cells of 1 cm with a single wall cell at col 2, row 1
    width = 4
    data = [0] * 16
    data[1 * width + 2] = 100
    grid = GridMap.from_data(data, 4, 4, 0.01, origin_x=-1.0, origin_y=-2.0, cell_size=0.02)
    assert (grid.width, grid.height) == (2, 2)
    assert abs(grid.resolution - 0.02) < 1e-12
    assert (grid.origin_x, grid.origin_y) == (-1.0, -2.0)
    # The wall cell lands in merged cell (1, 0), every other merged cell stays free
    assert list(grid.occupied) == [0, 1, 0, 0]


def test_from_data_merge_with_uneven_size():
    # 5 cells wide merged by 2 gives 3 columns, the last one covering a single cell
    data = [0, 0, 0, 0, 100]
    grid = GridMap.from_data(data, 5, 1, 0.01, cell_size=0.02)
    assert (grid.width, grid.height) == (3, 1)
    assert list(grid.occupied) == [0, 0, 1]


def test_from_occupancy_grid_message():
    msg = SimpleNamespace(
        data=[0, 100, 0, 0],
        info=SimpleNamespace(
            width=2, height=2, resolution=0.05,
            origin=SimpleNamespace(position=SimpleNamespace(x=-3.0, y=4.0))))
    grid = GridMap.from_occupancy_grid(msg)
    assert (grid.width, grid.height, grid.resolution) == (2, 2, 0.05)
    assert (grid.origin_x, grid.origin_y) == (-3.0, 4.0)
    assert list(grid.occupied) == [0, 1, 0, 0]


def test_transform_on_empty_grid():
    grid = make_grid(['.....'] * 5, resolution=0.5)
    cost = distance_transform(grid, free_cells(grid), [(0, 0)])
    assert cost[grid.index(0, 0)] == 0.0
    assert abs(cost[grid.index(4, 0)] - 4 * 0.5) < 1e-9
    assert abs(cost[grid.index(0, 3)] - 3 * 0.5) < 1e-9
    assert abs(cost[grid.index(4, 4)] - 4 * SQRT2 * 0.5) < 1e-9
    # Two steps along x and one diagonal step
    assert abs(cost[grid.index(3, 1)] - (2 + SQRT2) * 0.5) < 1e-9


def test_transform_goes_around_a_wall():
    grid = make_grid([
        '.......',
        '...#...',
        '...#...',
        '...#...',
        '.......',
    ])
    cost = distance_transform(grid, free_cells(grid), [(1, 2)])
    # Straight through the wall would be 4. Getting past the wall end at (3, 4) takes
    # one diagonal and two straight steps, because the diagonal past the wall corner
    # is not allowed, and the way back down on the other side costs the same.
    assert abs(cost[grid.index(5, 2)] - (2 * SQRT2 + 4)) < 1e-9
    assert cost[grid.index(3, 2)] == INF


def test_transform_marks_closed_room_unreachable():
    grid = make_grid([
        '.......',
        '..###..',
        '..#.#..',
        '..###..',
        '.......',
    ])
    traversable = free_cells(grid)
    cost = distance_transform(grid, traversable, [(0, 0)])
    assert cost[grid.index(3, 2)] == INF
    assert cost[grid.index(6, 4)] < INF
    assert descend(grid, traversable, cost, (3, 2)) is None


def test_transform_does_not_cut_corners():
    grid = make_grid([
        '.#',
        '#.',
    ])
    cost = distance_transform(grid, free_cells(grid), [(1, 0)])
    # The only way from (1, 0) to (0, 1) is diagonally between the two wall cells
    assert cost[grid.index(0, 1)] == INF

    grid = make_grid([
        '..',
        '#.',
    ])
    cost = distance_transform(grid, free_cells(grid), [(1, 0)])
    # Round the wall corner takes two straight steps, not one diagonal step
    assert abs(cost[grid.index(0, 1)] - 2.0) < 1e-9


def test_transform_with_several_sources():
    grid = make_grid(['.........'])
    cost = distance_transform(grid, free_cells(grid), [(0, 0), (8, 0)])
    assert [round(value) for value in cost] == [0, 1, 2, 3, 4, 3, 2, 1, 0]


def test_transform_matches_slow_reference_on_random_maps():
    generator = random.Random(160)
    for _ in range(25):
        width = generator.randint(3, 14)
        height = generator.randint(3, 14)
        occupied = bytearray(
            1 if generator.random() < 0.3 else 0 for _ in range(width * height))
        grid = GridMap(width, height, 0.1, 0.0, 0.0, occupied)
        traversable = free_cells(grid)
        free = [i for i in range(width * height) if traversable[i]]
        if not free:
            continue
        source = generator.choice(free)
        sources = [(source % width, source // width)]
        fast = distance_transform(grid, traversable, sources)
        slow = reference_transform(grid, traversable, sources)
        for a, b in zip(fast, slow):
            assert a == b or abs(a - b) < 1e-9


def test_descend_gives_a_valid_path_of_the_right_length():
    generator = random.Random(5)
    checked = 0
    for _ in range(25):
        width = generator.randint(5, 14)
        height = generator.randint(5, 14)
        occupied = bytearray(
            1 if generator.random() < 0.25 else 0 for _ in range(width * height))
        grid = GridMap(width, height, 0.1, 0.0, 0.0, occupied)
        traversable = free_cells(grid)
        free = [i for i in range(width * height) if traversable[i]]
        if len(free) < 2:
            continue
        goal = generator.choice(free)
        start = generator.choice(free)
        goal_cell = (goal % width, goal // width)
        start_cell = (start % width, start // width)
        cost = distance_transform(grid, traversable, [goal_cell])
        path = descend(grid, traversable, cost, start_cell)
        if cost[start] == INF:
            assert path is None
            continue
        checked += 1
        assert path[0] == start_cell
        assert path[-1] == goal_cell
        for (c0, r0), (c1, r1) in zip(path, path[1:]):
            # Each step goes to one of the 8 neighbours, stays on free cells and
            # does not squeeze diagonally past a wall corner
            assert max(abs(c1 - c0), abs(r1 - r0)) == 1
            assert traversable[grid.index(c1, r1)]
            assert traversable[grid.index(c1, r0)]
            assert traversable[grid.index(c0, r1)]
        points = [grid.cell_to_world(col, row) for col, row in path]
        assert abs(path_length(points) - cost[start]) < 1e-9
    assert checked > 5


def test_obstacle_distance_is_euclidean():
    grid = make_grid([
        '.......',
        '.......',
        '.......',
        '...#...',
        '.......',
        '.......',
        '.......',
    ], resolution=0.5)
    distance = obstacle_distance(grid)
    for col in range(7):
        for row in range(7):
            expected = 0.5 * math.hypot(col - 3, row - 3)
            assert abs(distance[grid.index(col, row)] - expected) < 1e-9


def test_obstacle_distance_respects_max_distance():
    grid = make_grid(['#......'])
    distance = obstacle_distance(grid, max_distance=3.0)
    assert distance[:4] == [0.0, 1.0, 2.0, 3.0]
    assert distance[4:] == [INF, INF, INF]


def test_obstacle_distance_without_obstacles():
    grid = make_grid(['...', '...'])
    assert obstacle_distance(grid) == [INF] * 6


def test_planner_keeps_robot_radius_from_walls():
    grid = make_grid([
        '##########',
        '..........',
        '..........',
        '..........',
        '..........',
        '..........',
        '##########',
    ], resolution=0.1)
    planner = Planner(grid, robot_radius=0.15, soft_radius=0.15, wall_weight=0.0)
    for row, allowed in enumerate([0, 0, 1, 1, 1, 0, 0]):
        assert planner.traversable[grid.index(4, row)] == allowed


def test_planner_closes_gap_that_is_too_narrow():
    picture = [
        '....#....',
        '....#....',
        '.........',
        '.........',
        '.........',
        '....#....',
        '....#....',
    ]
    grid = make_grid(picture, resolution=0.1)
    start = grid.cell_to_world(0, 3)
    goal = grid.cell_to_world(8, 3)
    # The gap is 3 cells: the middle cell is 0.2 m from the wall cells on both sides
    assert Planner(grid, robot_radius=0.15, wall_weight=0.0).path(start, goal) is not None
    assert Planner(grid, robot_radius=0.25, wall_weight=0.0).path(start, goal) is None


def test_planner_path_runs_from_start_to_goal():
    grid = make_grid([
        '..........',
        '..........',
        '.....#....',
        '.....#....',
        '.....#....',
        '.....#....',
        '..........',
        '..........',
    ], resolution=0.1)
    planner = Planner(grid, robot_radius=0.05, soft_radius=0.05, wall_weight=0.0)
    start = (0.12, 0.43)
    goal = (0.88, 0.41)
    path = planner.path(start, goal)
    assert path[0] == grid.cell_to_world(*grid.world_to_cell(*start))
    assert path[-1] == grid.cell_to_world(*grid.world_to_cell(*goal))
    for x, y in path:
        assert planner.traversable[grid.index(*grid.world_to_cell(x, y))]
    # Longer than the straight line because of the wall, but not absurdly so
    assert 0.76 < path_length(path) < 1.3


def test_wall_weight_moves_path_to_middle_of_corridor():
    # A wide corridor with start and goal close to the lower wall
    picture = ['#' * 30] + ['.' * 30] * 11 + ['#' * 30]
    grid = make_grid(picture, resolution=0.1)
    start = grid.cell_to_world(1, 2)
    goal = grid.cell_to_world(28, 2)

    shortest = Planner(grid, robot_radius=0.1, soft_radius=0.6, wall_weight=0.0)
    path = shortest.path(start, goal)
    assert max(grid.world_to_cell(x, y)[1] for x, y in path) == 2

    centred = Planner(grid, robot_radius=0.1, soft_radius=0.6, wall_weight=5.0)
    path = centred.path(start, goal)
    assert max(grid.world_to_cell(x, y)[1] for x, y in path) >= 4
    assert path[0] == start
    assert path[-1] == goal


def test_snap_moves_point_out_of_forbidden_zone():
    grid = make_grid([
        '##########',
        '..........',
        '..........',
        '..........',
        '..........',
        '..........',
        '##########',
    ], resolution=0.1)
    planner = Planner(grid, robot_radius=0.15, soft_radius=0.15, wall_weight=0.0)
    # Row 1 is free but too close to the lower wall, row 2 is the nearest allowed row
    assert planner.snap(*grid.cell_to_world(4, 1)) == (4, 2)
    assert planner.snap(*grid.cell_to_world(4, 3)) == (4, 3)
    # A point inside the wall itself is also moved out
    assert planner.snap(*grid.cell_to_world(4, 0)) == (4, 2)
    # Outside the map there is nothing to snap to
    assert planner.snap(-5.0, 0.3) is None
    # Too far from any allowed cell
    tight = Planner(grid, robot_radius=0.15, soft_radius=0.15, wall_weight=0.0,
                    snap_radius=0.05)
    assert tight.snap(*grid.cell_to_world(4, 1)) is None


def test_snap_does_not_jump_through_a_wall():
    # The point is next to a thin wall. The nearest allowed cell in a straight line is
    # on the other side, but the robot is on this side.
    grid = make_grid([
        '....#.........',
        '....#.........',
        '....#.........',
        '....#.........',
        '....#.........',
    ], resolution=0.1)
    planner = Planner(grid, robot_radius=0.25, soft_radius=0.25, wall_weight=0.0,
                      snap_radius=1.0)
    # Left of the wall only col 0 and 1 are far enough from it, right of it col 7 and up
    assert planner.traversable[grid.index(1, 2)]
    assert not planner.traversable[grid.index(2, 2)]
    assert planner.traversable[grid.index(7, 2)]
    assert not planner.traversable[grid.index(6, 2)]
    assert planner.snap(*grid.cell_to_world(3, 2)) == (1, 2)
    assert planner.snap(*grid.cell_to_world(5, 2)) == (7, 2)


def test_field_answers_many_lookups_and_is_symmetric():
    grid = make_grid([
        '............',
        '............',
        '......#.....',
        '......#.....',
        '......#.....',
        '............',
        '............',
    ], resolution=0.1)
    planner = Planner(grid, robot_radius=0.05, soft_radius=0.3, wall_weight=2.0)
    a = grid.cell_to_world(1, 3)
    b = grid.cell_to_world(10, 3)
    c = grid.cell_to_world(1, 6)
    field_a = planner.field(*a)
    field_b = planner.field(*b)
    assert field_a.source == a
    assert field_a.cost_at(*a) == 0.0
    assert field_a.cost_at(*c) < field_a.cost_at(*b)
    assert abs(field_a.cost_at(*b) - field_b.cost_at(*a)) < 1e-9
    # The cost is never lower than the length of the path it belongs to
    path = field_b.path_from(*a)
    assert path[0] == a and path[-1] == b
    assert field_b.cost_at(*a) >= path_length(path) - 1e-9
    assert planner.field(-9.0, -9.0) is None
    assert field_a.cost_at(-9.0, -9.0) == INF
    assert field_a.path_from(-9.0, -9.0) is None


def test_path_to_unreachable_room_is_none():
    grid = make_grid([
        '..........',
        '..#####...',
        '..#...#...',
        '..#...#...',
        '..#####...',
        '..........',
    ], resolution=0.1)
    planner = Planner(grid, robot_radius=0.05, soft_radius=0.05, wall_weight=0.0)
    assert planner.path(grid.cell_to_world(0, 0), grid.cell_to_world(4, 2)) is None
    assert planner.path(grid.cell_to_world(4, 2), grid.cell_to_world(5, 3)) is not None


def test_keep_out_sends_the_path_around_a_robot_standing_still():
    grid = make_grid(['....................'] * 12, resolution=0.1)
    planner = Planner(grid, robot_radius=0.05, soft_radius=0.05, wall_weight=0.0)
    start = (0.15, 0.55)
    goal = (1.85, 0.55)
    straight = planner.path(start, goal)
    assert path_length(straight) < 1.75

    robot = (1.0, 0.55)
    planner.set_keep_out([robot], 0.3)
    around = planner.path(start, goal)
    for x, y in around:
        assert math.hypot(x - robot[0], y - robot[1]) > 0.3
    assert path_length(around) > path_length(straight)

    # A new list replaces the old one
    planner.set_keep_out([], 0.3)
    assert path_length(planner.path(start, goal)) == path_length(straight)


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
