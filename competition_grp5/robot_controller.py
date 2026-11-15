"""
Kjører én robot til målpunkter på kartet.

Kjører én gang per robot, i robotens navnerom. Logikken ligger i distance_transform.py
og path_follower.py. Noden kobler den bare til ROS.

Abonnerer på:
    /map          nav_msgs/OccupancyGrid     konkurransekartet
    odom          nav_msgs/Odometry          robotens posisjon
    scan          sensor_msgs/LaserScan      lidar, for å stoppe for hindringer
    goal_pose     geometry_msgs/PoseStamped  kjør hit (retningen brukes ikke)
    cancel_goal   std_msgs/Empty             stopp og glem målet
    keep_out      geometry_msgs/PoseArray    roboter som står stille, fra koordinatoren

Publiserer:
    cmd_vel       geometry_msgs/Twist        bare under kjøring, så koordinatoren kan
                                             snu roboten når denne noden er ledig
    nav_status    std_msgs/String            'idle', eller tilstand og mål, for eksempel
                                             'driving 2.500 1.000'. Sendes ved endring og
                                             hvert sekund.
    plan          nav_msgs/Path              planlagt bane, for rviz

Posisjonen fra odom brukes rett som posisjon på kartet. spawn_robot.launch.py publiserer
map -> <navnerom>/odom som null-transform, og simulert odometri er i verdenskoordinater.
"""
import time

from competition_grp5.distance_transform import GridMap
from competition_grp5.distance_transform import Planner
from competition_grp5.path_follower import DRIVING
from competition_grp5.path_follower import FAILED
from competition_grp5.path_follower import IDLE
from competition_grp5.path_follower import Navigator
from competition_grp5.path_follower import PathFollower
from competition_grp5.path_follower import strip_clearance
from competition_grp5.path_follower import yaw_from_quaternion
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from std_msgs.msg import String


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # Planlegger
        self.cell_size = self.declare_parameter('cell_size', 0.05).value
        self.robot_radius = self.declare_parameter('robot_radius', 0.27).value
        self.soft_radius = self.declare_parameter('soft_radius', 0.5).value
        self.wall_weight = self.declare_parameter('wall_weight', 2.0).value
        # Kjøring
        self.max_speed = self.declare_parameter('max_speed', 0.26).value
        self.max_turn = self.declare_parameter('max_turn', 1.5).value
        self.lookahead = self.declare_parameter('lookahead', 0.35).value
        self.goal_tolerance = self.declare_parameter('goal_tolerance', 0.12).value
        # Hindringer, målt fra lidaren (se Navigator)
        self.stop_distance = self.declare_parameter('stop_distance', 0.30).value
        self.slow_distance = self.declare_parameter('slow_distance', 0.60).value
        self.strip_half_width = self.declare_parameter('strip_half_width', 0.18).value
        # Avstand fra senteret til en robot som står stille: dens radius pluss robot_radius
        self.keep_out_radius = self.declare_parameter('keep_out_radius', 0.45).value

        self.navigator = None
        self.pose = None
        self.front = 0.0
        self.rear = 0.0
        self.scan_time = None
        self.pending_goal = None
        self.published_status = None
        self.was_driving = False
        self.keep_out = []

        # Kartserveren sender kartet én gang og tar vare på det for sene abonnenter
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, '/map', self.clbk_map, map_qos)
        self.create_subscription(Odometry, 'odom', self.clbk_odom, 10)
        self.create_subscription(LaserScan, 'scan', self.clbk_scan, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, 'goal_pose', self.clbk_goal, 10)
        self.create_subscription(Empty, 'cancel_goal', self.clbk_cancel, 10)
        self.create_subscription(PoseArray, 'keep_out', self.clbk_keep_out, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'nav_status', 10)
        self.plan_pub = self.create_publisher(Path, 'plan', 1)

        self.create_timer(0.05, self.control_loop)
        self.create_timer(1.0, self.publish_status)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def clbk_map(self, msg):
        # Veggklokke, fordi simulert tid kan stå stille mens Gazebo starter
        started = time.perf_counter()
        grid = GridMap.from_occupancy_grid(msg, cell_size=self.cell_size)
        planner = Planner(grid, robot_radius=self.robot_radius,
                          soft_radius=self.soft_radius, wall_weight=self.wall_weight)
        follower = PathFollower(max_speed=self.max_speed, max_turn=self.max_turn,
                                lookahead=self.lookahead,
                                goal_tolerance=self.goal_tolerance)
        self.navigator = Navigator(planner, follower, stop_distance=self.stop_distance,
                                   slow_distance=self.slow_distance)
        planner.set_keep_out(self.keep_out, self.keep_out_radius)
        self.get_logger().info(
            'Map ready: %d x %d cells of %.2f m, %d of them allowed for the robot centre'
            % (grid.width, grid.height, grid.resolution, sum(planner.traversable)))
        self.get_logger().info(
            'Building the planner took %.2f s' % (time.perf_counter() - started))

    def clbk_odom(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
        self.pose = (position.x, position.y, yaw)

    def clbk_keep_out(self, msg):
        points = [(round(pose.position.x, 1), round(pose.position.y, 1))
                  for pose in msg.poses]
        if points == self.keep_out:
            return
        self.keep_out = points
        if self.navigator is None:
            return
        self.navigator.planner.set_keep_out(points, self.keep_out_radius)
        self.get_logger().info(
            'Planning around robots standing at ' + (', '.join(
                '(%.1f, %.1f)' % point for point in points) or 'nowhere'))
        # Planlegg på nytt hvis banen går gjennom en av dem
        navigator = self.navigator
        if navigator.status == DRIVING and self.pose is not None and any(
                (px - x) ** 2 + (py - y) ** 2 <= self.keep_out_radius ** 2
                for px, py in navigator.path for x, y in points):
            # Stå stille mens den nye banen planlegges, ellers kjører roboten videre på
            # den gamle
            self.cmd_vel_pub.publish(Twist())
            x, y, _ = self.pose
            if navigator.set_goal(navigator.goal, x, y, self.now()):
                self.get_logger().info('Planned a new path around a robot standing still')
                self.publish_plan()

    def clbk_scan(self, msg):
        self.front = strip_clearance(
            msg.ranges, msg.angle_min, msg.angle_increment, self.strip_half_width,
            range_min=msg.range_min)
        self.rear = strip_clearance(
            msg.ranges, msg.angle_min, msg.angle_increment, self.strip_half_width,
            range_min=msg.range_min, backwards=True)
        self.scan_time = self.now()

    def clbk_goal(self, msg):
        self.pending_goal = (msg.pose.position.x, msg.pose.position.y)
        # Stå stille mens den nye banen planlegges
        if self.was_driving:
            self.cmd_vel_pub.publish(Twist())

    def clbk_cancel(self, msg):
        self.pending_goal = None
        if self.navigator is not None:
            self.navigator.cancel()

    def start_pending_goal(self):
        """Planlegg banen til ventende mål når kart, posisjon og skann er kommet."""
        if self.navigator is None or self.pose is None or self.scan_time is None:
            return
        goal = self.pending_goal
        self.pending_goal = None
        x, y, _ = self.pose
        started = time.perf_counter()
        if self.navigator.set_goal(goal, x, y, self.now()):
            self.get_logger().info(
                'Driving to (%.2f, %.2f), path of %d points, planned in %.2f s'
                % (goal[0], goal[1], len(self.navigator.path),
                   time.perf_counter() - started))
            self.publish_plan()
        else:
            self.get_logger().warn(
                'No path from (%.2f, %.2f) to (%.2f, %.2f). If the way goes through a '
                'narrow opening, try a smaller robot_radius.' % (x, y, goal[0], goal[1]))

    def control_loop(self):
        if self.pending_goal is not None:
            self.start_pending_goal()
        if self.navigator is None or self.pose is None:
            return

        now = self.now()
        front = self.front
        rear = self.rear
        # Uten ferskt skann regnes begge retninger som blokkert
        if self.scan_time is None or now - self.scan_time > 1.0:
            front = 0.0
            rear = 0.0

        x, y, yaw = self.pose
        path_before = self.navigator.path
        speed, turn = self.navigator.update(now, x, y, yaw, front, rear)
        status = self.navigator.status

        if status == DRIVING:
            cmd = Twist()
            cmd.linear.x = speed
            cmd.angular.z = turn
            self.cmd_vel_pub.publish(cmd)
            if self.navigator.path is not path_before:
                self.get_logger().info('Planned a new path after getting blocked or off track')
                self.publish_plan()
        elif self.was_driving:
            # Akkurat stoppet: send én stoppkommando, og la så cmd_vel være
            self.cmd_vel_pub.publish(Twist())
            if status == FAILED:
                self.get_logger().warn('Gave up on the goal')
            elif status != IDLE:
                self.get_logger().info('Goal ' + status)
        self.was_driving = status == DRIVING

        if self.status_text() != self.published_status:
            self.publish_status()

    def status_text(self):
        """Gi tilstanden sammen med målet, så de to aldri blandes."""
        if self.navigator is None or self.navigator.goal is None:
            return IDLE
        goal = self.navigator.goal
        return '%s %.3f %.3f' % (self.navigator.status, goal[0], goal[1])

    def publish_status(self):
        msg = String()
        msg.data = self.status_text()
        self.published_status = msg.data
        self.status_pub.publish(msg)

    def publish_plan(self):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.navigator.path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.plan_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    robot_controller = RobotController()

    rclpy.spin(robot_controller)

    robot_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
