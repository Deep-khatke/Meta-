import time, json, urllib.request
import roslibpy

urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/api/project/new?name=ROS&template=rover", method="POST"))
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/api/simulation/control",
    data=json.dumps({"action": "play"}).encode(), headers={"Content-Type": "application/json"}, method="POST"))

ros = roslibpy.Ros(host="localhost", port=9090)
ros.run(timeout=10)
print("connected:", ros.is_connected)

# Discovery via rosapi, like Foxglove/rqt do.
topics = roslibpy.Service(ros, "/rosapi/topics", "rosapi/Topics").call(roslibpy.ServiceRequest(), timeout=5)
print("topics:", dict(zip(topics["topics"], topics["types"])))

got = {}
def mk(name):
    def cb(msg):
        got.setdefault(name, []).append(msg)
    return cb
for t, ty in [("/odom", "nav_msgs/msg/Odometry"), ("/scan", "sensor_msgs/msg/LaserScan"),
              ("/tf", "tf2_msgs/msg/TFMessage"), ("/imu", "sensor_msgs/msg/Imu"), ("/clock", "rosgraph_msgs/msg/Clock")]:
    roslibpy.Topic(ros, t, ty).subscribe(mk(t))
time.sleep(2)
for t, v in got.items():
    print("%-7s %3d msgs" % (t, len(v)))
o = got["/odom"][-1]
x0 = o["pose"]["pose"]["position"]["x"]
print("odom header:", o["header"])
print("scan beams:", len(got["/scan"][-1]["ranges"]) if "/scan" in got else 0)

# Drive via /cmd_vel at 10 Hz for 2.5 s, exactly like teleop_twist_keyboard.
cmd = roslibpy.Topic(ros, "/cmd_vel", "geometry_msgs/msg/Twist")
cmd.advertise()
for _ in range(25):
    cmd.publish(roslibpy.Message({"linear": {"x": 0.8, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": 0}}))
    time.sleep(0.1)
time.sleep(0.3)
x1 = got["/odom"][-1]["pose"]["pose"]["position"]["x"]
print("cmd_vel moved robot: x %.2f -> %.2f (dx=%.2f m)" % (x0, x1, x1 - x0))

# Nav2-style goal
goal = roslibpy.Topic(ros, "/goal_pose", "geometry_msgs/msg/PoseStamped")
goal.advertise()
goal.publish(roslibpy.Message({"header": {"frame_id": "map"},
    "pose": {"position": {"x": 0.0, "y": 2.0, "z": 0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}}))
time.sleep(6)
p = got["/odom"][-1]["pose"]["pose"]["position"]
print("goal_pose (0,2) -> robot now at (%.2f, %.2f)" % (p["x"], p["y"]))
print("bridge status:", json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/ros2/bridge/status").read())["cmd_vel_received"], "cmd_vel msgs received")
ros.terminate()
