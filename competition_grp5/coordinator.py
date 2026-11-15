"""
Koordinator for laget.

Sender robotene ut for å lete, rapporterer markørene de finner, og samler to roboter ved den store brannen. 
Kjører én gang for hele laget. Logikken ligger i mission.py, exploration.py og marker_tracking.py.

Abonnerer på:
    /map                       nav_msgs/OccupancyGrid              konkurransekartet
    /<robot>/odom              nav_msgs/Odometry                   robotposisjoner
    /<robot>/nav_status        std_msgs/String                     fra robot_controller
    /<robot>/marker_estimates  ros2_aruco_interfaces/ArucoMarkers  fra marker_detector
Kaller:
    /set_marker_position       scoring_interfaces/SetMarkerPosition  poengnoden
Publiserer:
    /<robot>/goal_pose         geometry_msgs/PoseStamped           mål for robot_controller
    /<robot>/cancel_goal       std_msgs/Empty
    /<robot>/cmd_vel           geometry_msgs/Twist                 bare for å snu på stedet
    /<robot>/keep_out          geometry_msgs/PoseArray             roboter som står stille
    /found_markers             ros2_aruco_interfaces/ArucoMarkers  godkjente markører
    /coverage                  nav_msgs/OccupancyGrid              0 sett, 100 ikke sett

Parameteren robot_namespaces gir robotene som leter, og helper_namespace hjelperen. 
Med explore satt til false rapporterer noden bare markører og styrer ingen roboter.
"""
from functools import partial
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from ros2_aruco_interfaces.msg import ArucoMarkers
from scoring_interfaces.srv import SetMarkerPosition
from std_msgs.msg import Empty
from std_msgs.msg import String

from competition_grp5.distance_transform import GridMap
from competition_grp5.distance_transform import INF
from competition_grp5.distance_transform import Planner
from competition_grp5.marker_tracking import MARKER_NAMES
from competition_grp5.marker_tracking import ReportBook
from competition_grp5.mission import Mission
from competition_grp5.path_follower import yaw_from_quaternion


def parse_nav_status(text):
    """Del 'driving 2.000 1.000' i ('driving', (2.0, 1.0)). 'idle' gir ('idle', None)."""
    parts = text.split()
    if len(parts) == 3:
        try:
            return parts[0], (float(parts[1]), float(parts[2]))
        except ValueError:
            pass
    return text.strip(), None


class Coordinator(Node):

    def __init__(self):
        super().__init__('coordinator')

        self.robots = list(
            self.declare_parameter('robot_namespaces', ['tb3_0', 'tb3_1']).value)
        # Hjelperen (se mission.py). Tom for ingen.
        helper = self.declare_parameter('helper_namespace', '').value
        self.helpers = [helper] if helper else []
        self.robots += [name for name in self.helpers if name not in self.robots]
        # Slutt å vente på svar fra poengnoden etter så lang tid, s
        self.call_timeout = self.declare_parameter('call_timeout', 5.0).value
        self.explore = self.declare_parameter('explore', True).value
        # Planlegger, må være lik som i robot_controller
        self.cell_size = self.declare_parameter('cell_size', 0.05).value
        self.robot_radius = self.declare_parameter('robot_radius', 0.27).value
        self.soft_radius = self.declare_parameter('soft_radius', 0.5).value
        self.wall_weight = self.declare_parameter('wall_weight', 2.0).value
        # Søk, se mission.py
        self.view_range = self.declare_parameter('view_range', 2.5).value
        self.camera_fov = self.declare_parameter('camera_fov', 1.085).value
        # 1.05 rad er 60 grader, se exploration.py
        self.max_view_angle = self.declare_parameter('max_view_angle', 1.05).value
        self.view_spacing = self.declare_parameter('view_spacing', 0.5).value
        self.coverage_cell = self.declare_parameter('coverage_cell', 0.25).value
        self.min_gain = self.declare_parameter('min_gain', 4).value
        self.local_min_gain = self.declare_parameter('local_min_gain', 2).value
        self.turn_rate = self.declare_parameter('turn_rate', 0.8).value
        self.meet_radius = self.declare_parameter('meet_radius', 1.2).value
        self.meet_done_radius = self.declare_parameter('meet_done_radius', 1.45).value
        self.meet_hold = self.declare_parameter('meet_hold', 1.0).value
        self.local_radius = self.declare_parameter('local_radius', 3.0).value
        self.spread = self.declare_parameter('spread', 1.5).value
        self.recall_margin = self.declare_parameter('recall_margin', 0.5).value

        self.book = ReportBook()
        # Rapporten som venter på svar: (markør-id, robot, posisjon)
        self.in_flight = None
        self.in_flight_since = 0.0
        self.all_found_logged = False
        self.not_ready_warned = -INF

        self.mission = None
        self.map_info = None
        self.poses = {}

        for robot in self.robots:
            self.create_subscription(
                ArucoMarkers, '/' + robot + '/marker_estimates',
                partial(self.clbk_estimates, robot), 10)

        self.scoring_client = self.create_client(SetMarkerPosition, '/set_marker_position')
        self.found_pub = self.create_publisher(ArucoMarkers, '/found_markers', 10)

        self.create_timer(0.2, self.report_next)
        self.create_timer(1.0, self.publish_found)

        if self.explore:
            map_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
                history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
                durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
                depth=1,
            )
            self.create_subscription(OccupancyGrid, '/map', self.clbk_map, map_qos)
            self.goal_pubs = {}
            self.cancel_pubs = {}
            self.cmd_vel_pubs = {}
            self.keep_out_pubs = {}
            self.keep_out_sent = {}
            for robot in self.robots:
                self.create_subscription(
                    Odometry, '/' + robot + '/odom', partial(self.clbk_odom, robot), 10)
                self.create_subscription(
                    String, '/' + robot + '/nav_status', partial(self.clbk_nav, robot), 10)
                self.goal_pubs[robot] = self.create_publisher(
                    PoseStamped, '/' + robot + '/goal_pose', 10)
                self.cancel_pubs[robot] = self.create_publisher(
                    Empty, '/' + robot + '/cancel_goal', 10)
                self.cmd_vel_pubs[robot] = self.create_publisher(
                    Twist, '/' + robot + '/cmd_vel', 10)
                self.keep_out_pubs[robot] = self.create_publisher(
                    PoseArray, '/' + robot + '/keep_out', 10)
            self.coverage_pub = self.create_publisher(OccupancyGrid, '/coverage', 1)
            self.create_timer(0.2, self.mission_step)
            self.create_timer(1.0, self.publish_coverage)
            self.create_timer(2.0, self.repeat_keep_out)

        self.get_logger().info('Coordinating ' + ', '.join(self.robots) + (
            ', with ' + ', '.join(self.helpers) + ' only helping at the big fire'
            if self.helpers else ''))

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # Leting og møte

    def clbk_map(self, msg):
        if self.mission is not None:
            return
        started = time.perf_counter()
        grid = GridMap.from_occupancy_grid(msg, cell_size=self.cell_size)
        planner = Planner(grid, robot_radius=self.robot_radius,
                          soft_radius=self.soft_radius, wall_weight=self.wall_weight)
        self.mission = Mission(planner, self.robots, view_range=self.view_range,
                               fov=self.camera_fov, spacing=self.view_spacing,
                               coverage_cell=self.coverage_cell, min_gain=self.min_gain,
                               local_min_gain=self.local_min_gain,
                               turn_rate=self.turn_rate, meet_radius=self.meet_radius,
                               meet_done_radius=self.meet_done_radius,
                               meet_hold=self.meet_hold, local_radius=self.local_radius,
                               spread=self.spread, helpers=self.helpers,
                               recall_margin=self.recall_margin,
                               max_view_angle=self.max_view_angle)
        for marker_id, (position, _) in self.book.accepted.items():
            self.mission.marker_accepted(marker_id, position)
        for robot, pose in self.poses.items():
            self.mission.update_pose(robot, *pose)
        self.map_info = msg.info
        self.get_logger().info(
            'Search planner ready in %.2f s' % (time.perf_counter() - started))

    def clbk_odom(self, robot, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
        self.poses[robot] = (position.x, position.y, yaw)
        if self.mission is not None:
            self.mission.update_pose(robot, position.x, position.y, yaw)

    def clbk_nav(self, robot, msg):
        if self.mission is not None:
            status, goal = parse_nav_status(msg.data)
            self.mission.update_nav(robot, status, goal)

    def mission_step(self):
        if self.mission is None:
            return
        was_ready = self.mission.ready
        started = time.perf_counter()
        actions = self.mission.step(self.now())
        if self.mission.ready and not was_ready:
            self.get_logger().info(
                'Search prepared in %.2f s' % (time.perf_counter() - started))
        for message in self.mission.take_messages():
            self.get_logger().info(message)
        for action in actions:
            kind, robot = action[0], action[1]
            if kind == 'goal':
                msg = PoseStamped()
                msg.header.frame_id = 'map'
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.pose.position.x = action[2][0]
                msg.pose.position.y = action[2][1]
                msg.pose.orientation.w = 1.0
                self.goal_pubs[robot].publish(msg)
            elif kind == 'cancel':
                self.cancel_pubs[robot].publish(Empty())
            elif kind == 'turn':
                msg = Twist()
                msg.angular.z = float(action[2])
                self.cmd_vel_pubs[robot].publish(msg)
        self.publish_keep_out()

    def publish_keep_out(self, repeat=False):
        """Fortell hver robot hvor de andre står stille, når det endrer seg."""
        if self.mission is None or not self.mission.ready:
            return
        parked = self.mission.parked()
        for robot in self.robots:
            points = sorted((round(x, 1), round(y, 1))
                            for name, (x, y) in parked.items() if name != robot)
            if not repeat and points == self.keep_out_sent.get(robot):
                continue
            self.keep_out_sent[robot] = points
            msg = PoseArray()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()
            for x, y in points:
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.orientation.w = 1.0
                msg.poses.append(pose)
            self.keep_out_pubs[robot].publish(msg)

    def repeat_keep_out(self):
        # I tilfelle en robot_controller startet sent
        self.publish_keep_out(repeat=True)

    def publish_coverage(self):
        if self.mission is None or not self.mission.ready:
            return
        coverage = self.mission.coverage
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = coverage.cell_size
        msg.info.width = coverage.width
        msg.info.height = coverage.height
        msg.info.origin = self.map_info.origin
        msg.data = coverage.occupancy_data()
        self.coverage_pub.publish(msg)

    # Rapportering av markører

    def clbk_estimates(self, robot, msg):
        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            position = (pose.position.x, pose.position.y, pose.position.z)
            self.book.add_estimate(int(marker_id), robot, position)

    def report_next(self):
        """Send én rapport til poengnoden hvis det er noe nytt å rapportere."""
        now = self.now()
        if self.in_flight is not None:
            if now - self.in_flight_since < self.call_timeout:
                return
            self.get_logger().warn('No answer from the scoring node, trying again')
            self.in_flight = None
        report = self.book.next_report(now)
        if report is None:
            return
        if not self.scoring_client.service_is_ready():
            if now - self.not_ready_warned > 5.0:
                self.not_ready_warned = now
                self.get_logger().warn(
                    'Marker %d is waiting to be reported, but the scoring service '
                    '/set_marker_position is not available' % report[0])
            return

        marker_id, robot, position = report
        request = SetMarkerPosition.Request()
        request.marker_id = marker_id
        request.marker_position.x = position[0]
        request.marker_position.y = position[1]
        request.marker_position.z = position[2]

        self.book.reported(marker_id, robot, position, now)
        self.in_flight = report
        self.in_flight_since = now
        future = self.scoring_client.call_async(request)
        future.add_done_callback(partial(self.clbk_scoring, report))

    def clbk_scoring(self, report, future):
        if self.in_flight is report:
            self.in_flight = None
        try:
            accepted = future.result().accepted
        except Exception as error:
            self.get_logger().warn('Call to the scoring node failed: ' + str(error))
            return

        marker_id, robot, position = report
        if self.book.result(marker_id, robot, position, accepted):
            self.get_logger().info(
                'Marker %d (%s) accepted at (%.2f, %.2f), found by %s. %d of %d found.'
                % (marker_id, MARKER_NAMES[marker_id], position[0], position[1], robot,
                   len(self.book.accepted), len(MARKER_NAMES)))
            if self.mission is not None:
                self.mission.marker_accepted(marker_id, position)
            self.publish_found()
            if not self.book.missing() and not self.all_found_logged:
                self.all_found_logged = True
                self.get_logger().info('All markers found')
        elif marker_id not in self.book.accepted:
            self.get_logger().info(
                'Marker %d at (%.2f, %.2f) from %s was not accepted yet'
                % (marker_id, position[0], position[1], robot))

    def publish_found(self):
        msg = ArucoMarkers()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for marker_id, (position, _) in sorted(self.book.accepted.items()):
            pose = Pose()
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = position[2]
            pose.orientation.w = 1.0
            msg.marker_ids.append(marker_id)
            msg.poses.append(pose)
        self.found_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    coordinator = Coordinator()

    rclpy.spin(coordinator)

    coordinator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
