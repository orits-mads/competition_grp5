# competition_grp5

DAT160 s&r competition, group 5. Robots explore a Gazebo world, find ArUco markers and report them to the scoring node.

Competition description: https://frdedynamics.github.io/hvl_robotics_website/courses/dat160/competition

## Setup

**Build:**
   ```bash
   cd ~/ros2_ws
   source /opt/ros/foxy/setup.bash
   colcon build --symlink-install
   source install/setup.bash
   ```

## Running

```bash
ros2 launch competition_grp5 competition.launch.py world:=w1
```

| Argument | Default | Description |
|---|---|---|
| `world` | `w1` | `w1` to `w5` |
| `rviz` | `false` | Start rviz (slows Gazebo down) |
| `scoring` | `true` | Start the scoring node |
| `explore` | `true` | `false` leaves the robots still, to drive by hand |
| `helper` | `true` | Start the helper robot `tb3_2` |


## Architecture

One central node, `coordinator`, decides **where** each robot goes. Each robot's own `robot_controller` decides **how** to get there.

```
marker_detector ──marker_estimates──▶ coordinator ──goal_pose──▶ robot_controller ──cmd_vel──▶ robot
   (per robot)                        (one, central) ◀─nav_status──    (per robot)
                                           │
                                           └──/set_marker_position──▶ scoring node
```

Robots: `tb3_0` and `tb3_1` search. The helper `tb3_2` waits at (0, 0) until the big fire is found.

| Trigger | Decided by | Decision | Action |
|---|---|---|---|
| A searcher is idle (`nav_status` is `reached` or `failed`) | coordinator | Pick the view point with the most unseen map per metre of driving. Points close by come first, and points near another robot's goal score less | Send `goal_pose` to that robot |
| A robot arrives at a view point | coordinator | Has it seen everything visible from here? | Turn on the spot, then mark the point as used |
| A new goal arrives | robot_controller | Plan the shortest safe path on the blueprint map (distance transform) | Drive along the path |
| Obstacle ahead, or the robot is stuck | robot_controller | Wait for the obstacle to move, then back up and plan again | Report `failed` after too many tries in the same place |
| Another robot stands still | coordinator | Treat it as an obstacle that is not on the map | Send its position on `keep_out`, so paths go around it |
| The camera sees an ArUco marker | marker_detector | Convert the detection to a map position | Publish on `marker_estimates` |
| New marker estimate | coordinator | Has this marker been accepted already? | If not, report it to the scoring node |
| Marker 4 (big fire) accepted | coordinator | Choose the meeting places: the helper deepest in, the closest searcher next to it | The helper drives there. The searcher looks at nearby view points until the helper is about as close, then joins it |
| Both robots near the big fire for 1 s | coordinator | The meeting is done | The searcher goes back to searching, the helper stays |
| Whole map seen, markers still missing | coordinator | Start the search over | Forget what has been seen and which view points were used |
| All 5 markers accepted and meeting done | coordinator | The mission is finished | Stop all robots |

The ROS nodes are thin wrappers. The logic is in modules without ROS imports, so it can be tested on its own:

```
semester_project_grp5/
├── robot_controller.py    ROS node
├── distance_transform.py  path planning
├── path_follower.py       steering, obstacle stop, recovery
├── marker_detector.py     ROS node
├── coordinator.py         ROS node
├── mission.py             who searches where, big fire meeting
├── exploration.py         coverage, view points, meeting places
└── marker_tracking.py     marker estimates and reports
```

`worlds/` holds the competition worlds with the `gazebo_ros_state` plugin that the scoring node needs.
