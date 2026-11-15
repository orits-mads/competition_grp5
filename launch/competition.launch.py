import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

package_name = 'competition_grp5'

challenge_package = 'multi_robot_challenge_23'

# Poengnoden lytter på /tb3_0/odom til /tb3_4/odom, men leser startposisjon bare for
# tb3_0 og tb3_1
robot_namespaces = ['tb3_0', 'tb3_1']

# Hjelperen (se mission.py). Poengnoden bruker (0, 0) som start for alle roboter unntatt
# tb3_0 og tb3_1, og starter klokken når en robot er 0.1 m fra start. Derfor må
# hjelperen starte i (0, 0).
helper_namespace = 'tb3_2'
helper_position = ['0.0', '0.0']

# Startposisjon [x, y] for tb3_0 og tb3_1 i hver verden, fra rescue_robots_wX.launch.py
start_positions = {
    'w1': [['0.0', '-1.0'], ['0.0', '1.0']],
    'w2': [['-1.0', '1.0'], ['-1.0', '-1.0']],
    'w3': [['-1.0', '0.5'], ['-1.0', '-1.5']],
    'w4': [['-0.5', '0.7'], ['-0.5', '-1.3']],
    'w5': [['0.0', '-0.5'], ['0.0', '-2.5']],
}


def robots_to_start(context):
    """
    Returner [(navnerom, [x, y])] for robotene som skal startes, hjelperen sist.

    Kall den før Gazebo-launchfilen tas med. I Foxy lekker argumentene fra en inkludert
    launchfil, så etterpå er 'world' stien til verdensfilen.
    """
    world = context.launch_configurations['world']
    robots = list(zip(robot_namespaces, start_positions[world]))
    if use_helper(context):
        robots.append((helper_namespace, helper_position))
    return robots


def all_namespaces(context):
    """Returner navnerommene til alle robotene, hjelperen sist."""
    return robot_namespaces + ([helper_namespace] if use_helper(context) else [])


def use_helper(context):
    return context.launch_configurations['helper'].lower() in ('true', '1', 'yes')


def start_simulation(context):
    """
    Start Gazebo, kartserveren, robotene og rviz for valgt verden.

    Som rescue_robots_wX.launch.py, men verdensfilene har gazebo_ros_state-pluginen
    som poengnoden trenger.
    """
    world = context.launch_configurations['world']
    if world not in start_positions:
        raise RuntimeError('Unknown world "' + world + '", use one of ' + str(list(start_positions)))

    challenge_share = get_package_share_directory(challenge_package)
    world_file_path = os.path.join(
        get_package_share_directory(package_name), 'worlds', 'dat160_' + world + '.world')
    map_file_path = os.path.join(challenge_share, 'maps', 'map_dat160_' + world + '.yaml')
    rviz_config_file_path = os.path.join(challenge_share, 'rviz', 'model.rviz')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world_file_path}.items()
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_file_path, 'topic_name': 'map', 'frame_id': 'map'}],
    )
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': True},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )

    # Starter roboten og ArUco-deteksjonen dens
    robots = []
    for namespace, position in robots_to_start(context):
        robots.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(challenge_share, 'launch', 'spawn_robot.launch.py')),
            launch_arguments={
                'namespace': namespace,
                'x': position[0],
                'y': position[1],
                'yaw': '0.0',
                'use_sim_time': 'true',
            }.items()
        ))

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file_path],
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return [gazebo, map_server, lifecycle_manager, *robots, rviz]


def start_our_nodes(context):
    """Start robot_controller og marker_detector for hver robot, og koordinatoren."""
    use_sim_time = {'use_sim_time': True}

    # Noder per robot, i robotens navnerom
    robot_nodes = []
    for namespace in all_namespaces(context):
        robot_nodes.append(Node(
            package=package_name,
            executable='robot_controller',
            namespace=namespace,
            output='screen',
            parameters=[use_sim_time],
        ))
        # tf ligger på globale /tf og /tf_static. I Foxy bruker en tf-lytter relative
        # navn, som i et navnerom blir /tb3_0/tf, der map-rammen mangler. Derfor
        # remappingen.
        robot_nodes.append(Node(
            package=package_name,
            executable='marker_detector',
            namespace=namespace,
            output='screen',
            parameters=[use_sim_time],
            remappings=[('tf', '/tf'), ('tf_static', '/tf_static')],
        ))

    # Noder for hele laget
    coordinator_parameters = {'robot_namespaces': robot_namespaces,
                          'explore': LaunchConfiguration('explore'),
                          'min_gain': 6, 'local_min_gain': 3}
    if use_helper(context):
        coordinator_parameters['helper_namespace'] = helper_namespace
    coordinator = Node(
        package=package_name,
        executable='coordinator',
        output='screen',
        parameters=[use_sim_time, coordinator_parameters],
    )
    return [*robot_nodes, coordinator]


def generate_launch_description():
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='w1',
        description='Competition world to start: w1, w2, w3, w4 or w5'
    )
    scoring_arg = DeclareLaunchArgument(
        'scoring',
        default_value='true',
        description='Start the scoring node'
    )
    # Av som standard: rviz halverer omtrent farten til Gazebo, og poengnoden teller
    # ekte sekunder
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Start rviz'
    )
    explore_arg = DeclareLaunchArgument(
        'explore',
        default_value='true',
        description='Let the coordinator send the robots out to search. Set to false to '
                    'drive them by hand with goal_pose.'
    )
    helper_arg = DeclareLaunchArgument(
        'helper',
        default_value='true',
        description='Start a third robot, ' + helper_namespace + ', that only helps at '
                    'the big fire'
    )

    # Gazebo, kartserver, robotene med ArUco-deteksjon, og rviz
    simulation = OpaqueFunction(function=start_simulation)

    scoring = Node(
        package='scoring',
        executable='scoring',
        output='screen',
        condition=IfCondition(LaunchConfiguration('scoring')),
    )

    our_nodes = OpaqueFunction(function=start_our_nodes)

    return LaunchDescription([
        world_arg,
        scoring_arg,
        rviz_arg,
        explore_arg,
        helper_arg,
        simulation,
        scoring,
        our_nodes,
    ])
