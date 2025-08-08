import cv2
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ultralytics import YOLO

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative


class BallFollower(Node):
    def __init__(self):
        super().__init__('ball_follower_pid')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.model = YOLO("runs/detect/train4/weights/best.pt")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("khong mo duoc camera.")
            exit()

        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz

        # PID điều khiển góc
        self.pid_angle = PIDController(kp=1.2, ki=0.0, kd=0.1)

        # FSM
        self.state = "SEARCH"
        self.current_target = 0
        self.total_targets = 3
        self.target_locked_time = None

    def send_velocity(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.publisher_.publish(msg)

    def stop(self):
        self.send_velocity(0.0, 0.0)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Khong doc duoc khung hinh.")
            return

        frame_center_x = frame.shape[1] // 2
        results = self.model.predict(source=frame, conf=0.5, stream=True)
        centers = []
        radii = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max((x2 - x1) // 2, (y2 - y1) // 2)
                centers.append((cx, cy))
                radii.append(radius)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        # Nếu thấy bóng
        if len(centers) > 0:
            sorted_balls = sorted(zip(centers, radii), key=lambda x: x[0][0])
            if self.current_target < len(sorted_balls):
                target_cx, target_cy = sorted_balls[self.current_target][0]
                target_r = sorted_balls[self.current_target][1]
                error_x = target_cx - frame_center_x
                angle_error = error_x / frame.shape[1] * 1.5

                if self.state in ["SEARCH", "APPROACH"]:
                    self.state = "APPROACH"
                    w = -self.pid_angle.compute(angle_error)

                    if target_r < 80:
                        v = 0.2
                    else:
                        self.state = "STOP_REACHED"
                        self.target_locked_time = time.time()
                        v = 0.0
                        w = 0.0
                    self.send_velocity(v, w)

        # Không thấy bóng → SEARCH
        else:
            if self.state != "STOP_REACHED" and self.state != "BACK_OFF":
                self.state = "SEARCH"
                self.send_velocity(0.0, 0.3)  # xoay tìm bóng

        # Xử lý STOP_REACHED
        if self.state == "STOP_REACHED":
            self.stop()
            if time.time() - self.target_locked_time > 1.0:
                self.state = "BACK_OFF"
                self.target_locked_time = time.time()

        # Xử lý BACK_OFF
        elif self.state == "BACK_OFF":
            self.send_velocity(-0.2, 0.0)
            if time.time() - self.target_locked_time > 2.0:
                self.stop()
                self.current_target += 1
                if self.current_target >= self.total_targets:
                    self.state = "DONE"
                else:
                    self.state = "SEARCH"

        # Khi DONE
        elif self.state == "DONE":
            self.stop()

        cv2.putText(frame, f"State: {self.state}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Target: {self.current_target+1}/{self.total_targets}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Ball Tracking PID", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = BallFollower()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
