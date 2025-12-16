import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

import asyncio
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from google import genai
from google.genai import types

class TangramPlannerNode(Node):

    def __init__(self):
        super().__init__('tangram_planner')

        self.pick_publishers = []
        for i in range(7):
            pub = self.create_publisher(PoseStamped, f'/tangram/pick_{i}_pose', 1)
            self.pick_publishers.append(pub)

        self.rect_pub = self.create_publisher(Image, '/tangram/rectified_image', 1)
        self.masked_pub = self.create_publisher(Image, '/tangram/masked_image', 1)
        self.output_pub = self.create_publisher(Image, '/tangram/output_image', 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.cam_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 1)
        self.bridge = CvBridge()

        self.piece_transforms = {'blue': None, 'light blue': None, 'green': None, 'yellow': None, 'purple': None, 'hot pink': None, 'red': None}
        self.num_frames = 0

        self.timer = self.create_timer(1, self.publish_pick_poses)
        self.process_image_lock = False

    # Configuration
    MCP_SERVER_URL = "http://localhost:8000/sse"
    API_KEY = os.environ.get("GEMINI_API_KEY")

    def solve_callback(self, request, response):
        self.get_logger().info("Starting planning")

        # TODO: get instruction somehow
        instruction = "Please arrange these shapes into your best approximation of a house."

        try:
            blueprint = asyncio.run(self.run_planning_cycle(instruction))

            msg = String()
            msg.data = json.dumps(blueprint)
            self.publisher_.publish(msg)

            response.success = True
            response.message = "Blueprint generated and published."
        except Exception as e:
            self.get_logger().error(f"Planning failed: {e}")
            response.success = False
            response.message = str(e)

        return response

    async def run_planning_cycle(self, user_instruction):
        client = genai.Client(api_key=API_KEY)

        print(f"connecting at {MCP_SERVER_URL}...")
        async with sse_client(MCP_SERVER_URL) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()

                tools = await session.list_tools()

                print(tools)
                funcs = []
                for tool in tools.tools:
                    funcs.append({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    })

                chat = client.chats.create(model="gemini-3-pro-preview",
                                        config=types.GenerateContentConfig(
                                            temperature=1,
                                            tools=[{"function_declarations": funcs}]
                                        )
                                    )

                system_prompt = """
    you are looking at a set of tangram pieces. try to arrange them to form your best approxmation of whatever task the user is interested in, using all MCP tools available to you! You may do/undo moves as you wish, as long as the final configuration is an approximation to whatever the user wants, with no overlaps between pieces. After you're done, try to give each tangram piece a buffer (you can use the too_close function to help see which pieces need to be moved). For this, you should start with the outermost pieces, move them apart and then adjust to inner ones.

    - You have access to tools like 'move_polygon', 'rotate_polygon', and 'get_observation'.
    - **DO NOT** write Python code or markdown blocks.
    - **DO NOT** simulate the moves in text.
    - **DIRECTLY CALL** the functions to move the pieces.
    - Execute one or two moves, then observe, then move again.

    get_observation allows you to view the board as an image
    get_observation_points tells you the vertices of all shapes
    intersections tells you whether your pieces are colliding with each other (which is not allowed)
    too_close tells you which pieces are too close to each other (within 10 pixels). This is more optional to follow, but recommended. I suggest that you don't use it until after you're done with your design!

    actively make moves. don't get stuck thinking too hard when you can try things and see what works, physically. it's a puzzle you have to interact with!

    When finished, just output "done" without any other punctuation.

    DO NOT REFER TO ANY EXTERNAL SOURCES, OR MAKE NEW FILES.
                """

                print("sending response")

                response = chat.send_message(f"{system_prompt}\n\nTask: {user_instruction}")

                print("response received")

                while True:
                    if response.function_calls:
                        for call in response.function_calls:
                            print(f"Gemini calling: {call.name} with {call.args}")

                            result = await session.call_tool(call.name, arguments=call.args)

                            response = chat.send_message(
                                types.Part.from_function_response(
                                    name=call.name,
                                    response={"result": result.content}
                                )
                            )
                    else:
                        candidate = response.candidates[0]
                        if response.text is None:
                            print("\n!!! RESPONSE BLOCKED OR EMPTY !!!")
                            print(f"Finish Reason: {candidate.finish_reason}")
                            print(f"Safety Ratings: {candidate.safety_ratings}")

                            response = await chat.send_message(
                                "The previous response was blocked. Please try again, but be more concise and avoid harmful keywords."
                            )
                            continue

                        print("Gemini Response:", response.text)
                        if response.text and ("done" in response.text.lower() or "finished" in response.text.lower()):
                            break

                        response = chat.send_message("Continue or confirm completion.")

                final_state_result = await session.call_tool("get_polygon_poses")
                final_blueprint = json.loads(final_state_result.content[0].text)

                return final_blueprint

def main(args=None):
    rclpy.init(args=args)
    node = TangramPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


# class RealSenseSubscriber(Node):
    # def pose_stamped_to_transform_stamped(self, pose, child_frame_id):
    #     transform = TransformStamped()
    #     transform.header = pose.header
    #     transform.child_frame_id = child_frame_id
    #     transform.transform.translation.x = pose.pose.position.x
    #     transform.transform.translation.y = pose.pose.position.y
    #     transform.transform.translation.z = pose.pose.position.z
    #     transform.transform.rotation.x = pose.pose.orientation.x
    #     transform.transform.rotation.y = pose.pose.orientation.y
    #     transform.transform.rotation.z = pose.pose.orientation.z
    #     transform.transform.rotation.w = pose.pose.orientation.w
    #     return transform

#     def transform_stamped_to_pose_stamped(self, transform, msg):
#         pose_stamped = PoseStamped()
#         pose_stamped.header = msg.header
#         pose_stamped.header.frame_id = 'base_link'
#         pose_stamped.pose.position.x = transform.translation.x
#         pose_stamped.pose.position.y = transform.translation.y
#         pose_stamped.pose.position.z = transform.translation.z
#         pose_stamped.pose.orientation.x = transform.rotation.x
#         pose_stamped.pose.orientation.y = transform.rotation.y
#         pose_stamped.pose.orientation.z = transform.rotation.z
#         pose_stamped.pose.orientation.w = transform.rotation.w
#         return pose_stamped

#     def z_axis_rot(self, theta):
#         return np.array([[np.cos(theta), -np.sin(theta), 0],
#                                    [np.sin(theta),  np.cos(theta), 0],
#                                    [0,              0,             1]])

#     def final_quat(self, rot_mat):
#         rot = np.array([[1, 0, 0],
#                          [0, -1, 0],
#                          [0, 0, -1]]) @ rot_mat
#         return R.from_matrix(rot).as_quat()

#     def publish_pick_poses(self):
#         for color in self.piece_transforms:
#             if self.piece_transforms[color] is not None:
#                 self.pick_publishers[list(self.piece_transforms.keys()).index(color)].publish(self.piece_transforms[color])

#     def image_callback(self, msg):
#         tangram = extract_corners_from_image(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'), self)
#         if tangram is None:
#             return

#         if len(tangram.pieces) == 0 or not tangram.pieces[0].meters:
#             return

#         self.num_frames += 1
#         for p in range(len(tangram.pieces)):
#             piece = tangram.pieces[p]
#             p_col = piece.color

#             z_axis_quat = self.z_axis_rot(piece.pose[2] + np.pi)
#             final_quat = self.final_quat(z_axis_quat)

#             transform_to_marker = TransformStamped()
#             transform_to_marker.header = msg.header
#             transform_to_marker.header.frame_id = 'ar_marker_0'
#             transform_to_marker.child_frame_id = f'tangram_pick_{p_col}'
#             # NOTE: THESE OFFSETS ARE BASED ON THE TRANSLATION FROM EEF TO WRIST 3
#             transform_to_marker.transform.translation.x = float(piece.pose[0]) - 0.066 * np.sin(piece.pose[2])
#             transform_to_marker.transform.translation.y = float(piece.pose[1]) - 0.066 * np.cos(piece.pose[2])
#             transform_to_marker.transform.translation.z = 0.255
#             transform_to_marker.transform.rotation.x = final_quat[0]
#             transform_to_marker.transform.rotation.y = final_quat[1]
#             transform_to_marker.transform.rotation.z = final_quat[2]
#             transform_to_marker.transform.rotation.w = final_quat[3]

#             self.tf_broadcaster.sendTransform(transform_to_marker)

#             try:
#                 transform = self.tf_buffer.lookup_transform('base_link', f'tangram_pick_{p_col}', rclpy.time.Time()).transform
#             except:
#                 self.get_logger().info('still waiting for buffer transform')
#                 continue

#             if self.piece_transforms[p_col] is not None and self.num_frames > 200:
#                 pick_pose = self.piece_transforms[p_col]
#             else:
#                 pick_pose = self.transform_stamped_to_pose_stamped(transform, msg)
#                 self.piece_transforms[p_col] = pick_pose

# def main(args=None):
#     rclpy.init(args=args)
#     node = RealSenseSubscriber()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
