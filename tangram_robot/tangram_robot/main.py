# ROS Libraries
from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from planning.ik import IKPlanner

class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_place')

        self.pick_subs = []
        for i in range(7):
            self.pick_subs.append(self.create_subscription(PoseStamped, f'/tangram/pick_{i}_pose', lambda x, i=i: self.pick_callback(x, i), 1))

        self.place_subs = []
        for i in range(7):
            self.place_subs.append(self.create_subscription(PoseStamped, f'/tangram/place_{i}_pose', lambda x, i=i: self.place_callback(x, i), 1))

        self.tangrams = [[None, None] for _ in range(7)]

        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.pick_pose = None
        self.joint_state = None

        self.ik_planner = IKPlanner()

        self.currently_picking = False
        self.create_timer(5, self.pick_place)

        # Entries should be of type either JointState or String('toggle_grip')
        self.job_queue = []

        self.get_logger().info('pick and place is gonna start in 5 seconds rip') # REDUCE BY CHANGING THE TIMER

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def pick_callback(self, msg, i):
        if abs(msg.pose.position.x) > 1 or abs(msg.pose.position.y) > 1 or abs(msg.pose.position.z) > 1:
            return

        self.tangrams[i][0] = msg.pose

    def place_callback(self, msg, i):
        if abs(msg.pose.position.x) > 1 or abs(msg.pose.position.y) > 1 or abs(msg.pose.position.z) > 1:
            return

        self.tangrams[i][1] = msg.pose

    def pick_place(self):
        self.get_logger().info('attempting to pick place')
        if self.currently_picking:
            self.get_logger().info('got rejected')
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return

        self.get_logger().info(f'picking')
        self.currently_picking = True # DO NOT MOVE THIS, WEIRD THINGS HAPPEN IF REMOVED (TREAT THIS AS A LOCK FOR THE JOB QUEUE)
        for i in range(len(self.tangrams)):
            pick_pose = self.tangrams[i][0]
            place_pose = self.tangrams[i][1]

            if pick_pose is None or place_pose is None:
                self.get_logger().info(f'skipping pick place for {i}')
                continue

            # NOTE: WE FIXED THE Z VALUES TO THE CONSTANT BECAUSE IT ALWAYS WORKS. THE ARUCO DETECTION IS NOISY SO THE Z VALUE DERIVED FROM THAT IS COOKED TOO
            self.pick_pose = (pick_pose.position.x, pick_pose.position.y, 0.035, pick_pose.orientation.x, pick_pose.orientation.y, pick_pose.orientation.z, pick_pose.orientation.w)
            self.place_pose = (place_pose.position.x, place_pose.position.y, 0.035, place_pose.orientation.x, place_pose.orientation.y, place_pose.orientation.z, place_pose.orientation.w)

            self.get_logger().info(f'pick pose: {self.pick_pose[:3]}')
            self.get_logger().info(f'place pose: {self.place_pose[:3]}')

            # 1) move to pre-pick position (pick + some z offset)
            ik = self.ik_planner.compute_ik(self.joint_state, self.pick_pose[0], self.pick_pose[1], self.pick_pose[2] + 0.03, qx=self.pick_pose[3], qy=self.pick_pose[4], qz=self.pick_pose[5], qw=self.pick_pose[6])
            if ik is None:
                self.get_logger().error('Failed to compute IK for pick position, skipping this piece')
                continue
            self.job_queue.append(ik)

            # 2) lower to pick position
            ik = self.ik_planner.compute_ik(self.joint_state, self.pick_pose[0], self.pick_pose[1], self.pick_pose[2], qx=self.pick_pose[3], qy=self.pick_pose[4], qz=self.pick_pose[5], qw=self.pick_pose[6])
            if ik is None:
                self.get_logger().error('Failed to compute IK for pick position, skipping this piece')
                continue
            self.job_queue.append(ik)

            # 3) start suction
            self.job_queue.append('toggle_grip')
            
            # 4) move back to pre-pick position
            ik = self.ik_planner.compute_ik(self.joint_state, self.pick_pose[0], self.pick_pose[1], self.pick_pose[2] + 0.03, qx=self.pick_pose[3], qy=self.pick_pose[4], qz=self.pick_pose[5], qw=self.pick_pose[6])
            if ik is None:
                self.get_logger().error('Failed to compute IK for pick position, skipping this piece')
                continue
            self.job_queue.append(ik)

            # 5) move to pre-place position
            ik = self.ik_planner.compute_ik(self.joint_state, self.place_pose[0], self.place_pose[1], self.place_pose[2] + 0.03, qx=self.place_pose[3], qy=self.place_pose[4], qz=self.place_pose[5], qw=self.place_pose[6])
            if ik is None:
                self.get_logger().error('Failed to compute IK for place position, skipping this piece')
                continue
            self.job_queue.append(ik)

            # 6) move to place position
            ik = self.ik_planner.compute_ik(self.joint_state, self.place_pose[0], self.place_pose[1], self.place_pose[2], qx=self.place_pose[3], qy=self.place_pose[4], qz=self.place_pose[5], qw=self.place_pose[6])
            if ik is None:
                self.get_logger().error('Failed to compute IK for place position, skipping this piece')
                continue
            self.job_queue.append(ik)

            # 7) stop suction
            self.job_queue.append('toggle_grip')

            # 8) move back to pre-place position
            ik = self.ik_planner.compute_ik(self.joint_state, self.place_pose[0], self.place_pose[1], self.place_pose[2] + 0.03, qx=self.place_pose[3], qy=self.place_pose[4], qz=self.place_pose[5], qw=self.place_pose[6])
            if ik is None:
                self.get_logger().error('Failed to compute IK for place position, skipping this piece')
                continue
            self.job_queue.append(ik)

        self.execute_jobs()
        # self.currently_picking = False # DO NOT UNCOMMENT THIS!!! THIS LOCK IS RELEASED WHEN JOB QUEUE IS EMPTY

    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            self.currently_picking = False
            # rclpy.shutdown()
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)
        input('press enter to confirm')

        if isinstance(next_job, JointState):

            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error("Failed to plan to position")
                return

            self.get_logger().info("Planned to position")

            self._execute_joint_trajectory(traj.joint_trajectory)
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()  # Proceed to next job

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        # wait for 2 seconds
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

            
    def _execute_joint_trajectory(self, joint_traj):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        print(send_future)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
