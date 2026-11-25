"""
Gjør ArUco-deteksjonene til én robot om til markørposisjoner på kartet.

Kjører én gang per robot, i robotens navnerom. Logikken ligger i marker_tracking.py.

Abonnerer på:
    aruco_markers     ros2_aruco_interfaces/ArucoMarkers  deteksjoner i kamerarammen,
                                                           fra kursets aruco_node
Publiserer:
    marker_estimates  ros2_aruco_interfaces/ArucoMarkers  én stabil posisjon per markør
                                                           i map-rammen

Hver deteksjon flyttes til map med kameraposisjonen fra da bildet ble tatt, ikke nå.
Roboten kan ha snudd i mellomtiden, og en liten vridning flytter en markør langt unna
mye. Mangler tf den posisjonen, brukes den siste.

aruco_node gir midten av markørplaten, 0.25 m over gulvet. Poengnoden sammenligner med
foten av platen og teller høydeforskjellen som feil, så høyden settes til gulvet.

AI: Claude wrote this file
"""
from collections import deque
import math

from competition_grp5.marker_tracking import MARKER_NAMES
from competition_grp5.marker_tracking import MarkerEstimator
from competition_grp5.marker_tracking import transform_point
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class MarkerDetector(Node):

    def __init__(self):
        super().__init__('marker_detector')

        namespace = self.get_namespace().strip('/')
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.camera_frame = self.declare_parameter('camera_frame', '').value
        if not self.camera_frame:
            self.camera_frame = namespace + '/camera_rgb_optical_frame'
        # Deteksjoner lenger unna enn dette brukes ikke, m
        max_range = self.declare_parameter('max_range', 4.0).value
        min_observations = self.declare_parameter('min_observations', 3).value
        # Rapporter markører på gulvhøyde i stedet for målt høyde
        self.use_floor_height = self.declare_parameter('use_floor_height', True).value
        self.floor_height = self.declare_parameter('floor_height', 0.0).value
        # Hvor lenge vi venter på kameraposisjonen til et bilde, s
        self.max_wait = self.declare_parameter('max_wait', 0.5).value
        # Logg hva kameraet ser, for feilsøking
        self.verbose = self.declare_parameter('verbose', True).value

        self.estimator = MarkerEstimator(max_range=max_range, min_observations=min_observations)
        # For loggen: markører sett og estimert minst én gang, og per markør-id
        # [deteksjoner, brukt, nærmeste avstand, siste posisjon] siden forrige oppsummering
        self.first_seen = set()
        self.estimated = set()
        self.recent = {}
        self.dropped = 0
        self.fallbacks = 0
        self.waiting = deque(maxlen=30)
        self.last_warning = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(ArucoMarkers, 'aruco_markers', self.clbk_aruco, 10)
        self.estimates_pub = self.create_publisher(ArucoMarkers, 'marker_estimates', 10)

        self.create_timer(0.05, self.process_waiting)
        if self.verbose:
            self.create_timer(5.0, self.log_summary)

    def name(self, marker_id):
        return '%d (%s)' % (marker_id, MARKER_NAMES.get(marker_id, 'not a competition id'))

    def log_summary(self):
        """Logg hva kameraet har sett de siste sekundene."""
        if not self.recent and not self.dropped and not self.fallbacks:
            return
        parts = []
        for marker_id, (count, used, closest, position) in sorted(self.recent.items()):
            parts.append(
                'marker %s %d times, %d used, closest %.2f m, last at (%.2f, %.2f, %.2f)'
                % (self.name(marker_id), count, used, closest, position[0], position[1],
                   position[2]))
        if self.fallbacks:
            parts.append('%d images used the latest camera pose' % self.fallbacks)
        if self.dropped:
            parts.append('%d images dropped for lack of a camera pose' % self.dropped)
        self.get_logger().info('Last 5 s: ' + '; '.join(parts))
        self.recent = {}
        self.dropped = 0
        self.fallbacks = 0

    def note_detection(self, marker_id, distance, on_map, used):
        if not self.verbose:
            return
        if marker_id not in self.first_seen:
            self.first_seen.add(marker_id)
            self.get_logger().info(
                'Sees marker %s for the first time, %.2f m away, at (%.2f, %.2f), height '
                '%.2f m%s' % (self.name(marker_id), distance, on_map[0], on_map[1], on_map[2],
                              '' if used else ', not used (too far or not a competition id)'))
        entry = self.recent.setdefault(marker_id, [0, 0, math.inf, on_map])
        entry[0] += 1
        entry[1] += 1 if used else 0
        entry[2] = min(entry[2], distance)
        entry[3] = on_map

    def clbk_aruco(self, msg):
        # Kameraposisjonen kan mangle ennå, så timeren tar seg av køen
        self.waiting.append(msg)

    def process_waiting(self):
        while self.waiting:
            msg = self.waiting[0]
            stamp = Time.from_msg(msg.header.stamp)
            if self.tf_buffer.can_transform(self.map_frame, self.camera_frame, stamp):
                self.waiting.popleft()
                self.handle_detections(msg, stamp)
                continue
            age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
            if abs(age) <= self.max_wait:
                # Venter fortsatt på posisjonen til det eldste bildet
                return
            self.waiting.popleft()
            reason = self.transform_error(stamp)
            # Posisjonen mangler. Den siste er litt feil hvis roboten har flyttet seg,
            # men mye bedre enn ingenting.
            latest = Time()
            if self.tf_buffer.can_transform(self.map_frame, self.camera_frame, latest):
                self.fallbacks += 1
                self.warn_transform(
                    'No camera pose for the time of the image (image %.2f s old, %s). '
                    'Using the latest pose instead.' % (age, reason))
                self.handle_detections(msg, latest)
            else:
                self.dropped += 1
                self.warn_transform(
                    'Dropped a marker detection: no transform from %s to %s at all (%s). '
                    'The image itself is in frame "%s".'
                    % (self.camera_frame, self.map_frame, self.transform_error(latest),
                       msg.header.frame_id))

    def transform_error(self, stamp):
        """Gi tf sin forklaring på hvorfor kameraposisjonen mangler."""
        try:
            self.tf_buffer.lookup_transform(self.map_frame, self.camera_frame, stamp)
        except TransformException as error:
            return str(error).strip()[:300]
        return 'no error from tf'

    def handle_detections(self, msg, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.camera_frame, stamp).transform
        except TransformException as error:
            self.dropped += 1
            self.warn_transform('Dropped a marker detection: ' + str(error).strip()[:300])
            return
        translation = (transform.translation.x, transform.translation.y,
                       transform.translation.z)
        rotation = (transform.rotation.x, transform.rotation.y,
                    transform.rotation.z, transform.rotation.w)

        seen = []
        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            marker_id = int(marker_id)
            in_camera = (pose.position.x, pose.position.y, pose.position.z)
            distance = math.sqrt(sum(value * value for value in in_camera))
            on_map = transform_point(translation, rotation, in_camera)
            used = self.estimator.add(marker_id, on_map, distance)
            self.note_detection(marker_id, distance, on_map, used)
            if used and marker_id not in seen:
                seen.append(marker_id)

        estimates = ArucoMarkers()
        estimates.header.frame_id = self.map_frame
        estimates.header.stamp = msg.header.stamp
        for marker_id in seen:
            position = self.estimator.estimate(marker_id)
            if position is None:
                continue
            if self.verbose and marker_id not in self.estimated:
                self.estimated.add(marker_id)
                self.get_logger().info(
                    'Marker %s located at (%.2f, %.2f) from %d detections, sending it to '
                    'the coordinator' % (self.name(marker_id), position[0], position[1],
                                         self.estimator.count(marker_id)))
            pose = Pose()
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = self.floor_height if self.use_floor_height else position[2]
            pose.orientation.w = 1.0
            estimates.marker_ids.append(marker_id)
            estimates.poses.append(pose)
        if estimates.marker_ids:
            self.estimates_pub.publish(estimates)

    def warn_transform(self, text):
        """Logg et problem med kameraposisjonen, maks én gang hvert 5. sekund."""
        now = self.get_clock().now()
        if self.last_warning is not None and (now - self.last_warning).nanoseconds < 5e9:
            return
        self.last_warning = now
        self.get_logger().warn(text)


def main(args=None):
    rclpy.init(args=args)

    marker_detector = MarkerDetector()

    rclpy.spin(marker_detector)

    marker_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
