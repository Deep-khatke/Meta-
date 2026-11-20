import React, { useState } from 'react';
import {
  FileCode,
  Download,
  Copy,
  Check,
  FileJson,
  FileText,
  Cpu
} from 'lucide-react';
import { Project } from '../../../types';

interface FilesTabProps {
  project: Project;
  onShowToast?: (msg: string) => void;
}

export const FilesTab: React.FC<FilesTabProps> = ({
  project,
  onShowToast,
}) => {
  const [selectedFile, setSelectedFile] = useState<string>('scene.json');
  const [copied, setCopied] = useState<boolean>(false);

  const files = [
    { name: 'scene.json', type: 'json', desc: 'Current MuJoCo 3D Scene Specification' },
    { name: 'robot.urdf', type: 'xml', desc: 'Unified Robot Description Format (URDF)' },
    { name: 'policy.onnx', type: 'onnx', desc: 'Exported Neural Policy Network (PPO)' },
    { name: 'ros2_bridge_node.py', type: 'python', desc: 'ROS 2 Telemetry & CmdVel Driver Node' },
  ];

  const getFileContent = (filename: string): string => {
    switch (filename) {
      case 'scene.json':
        return JSON.stringify(project || {}, null, 2);
      case 'robot.urdf':
        return `<robot name="${project?.robot?.name || 'roboforge_robot'}">
  <link name="base_link">
    <inertial>
      <mass value="${project?.robot?.chassis_mass || 5.0}"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.08"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.36 0.24 0.16"/>
      </geometry>
    </visual>
  </link>
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>`;
      case 'policy.onnx':
        return `// ONNX Model Header: v1.14.0\n// Producer: RoboForge PPO Neural Trainer (MuJoCo 3.17)\n// Inputs: [batch, 15] (LiDAR 8 + Pose 4 + Vel 3)\n// Outputs: [batch, 4] (Actuator Targets)`;
      case 'ros2_bridge_node.py':
        return `#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class RoboForgeBridge(Node):
    def __init__(self):
        super().__init__('roboforge_bridge_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.get_logger().info('RoboForge ROS 2 Bridge Connected to ws://localhost:9090')

    def odom_callback(self, msg):
        pass

def main():
    rclpy.init()
    node = RoboForgeBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()`;
      default:
        return '';
    }
  };

  const currentContent = getFileContent(selectedFile);

  const handleCopy = () => {
    navigator.clipboard.writeText(currentContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([currentContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedFile;
    a.click();
    URL.revokeObjectURL(url);
    onShowToast?.(`Downloaded ${selectedFile}`);
  };

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        backgroundColor: '#101113',
        color: '#F5F5F5',
        overflow: 'hidden',
      }}
    >
      {/* File List Pane */}
      <div
        style={{
          width: 220,
          borderRight: '1px solid #2B2C32',
          background: '#141518',
          display: 'flex',
          flexDirection: 'column',
          padding: '12px 10px',
          gap: 4,
          flexShrink: 0,
        }}
      >
        <div style={{ fontSize: 10, fontWeight: 700, color: '#71717A', textTransform: 'uppercase', padding: '0 8px 6px 8px' }}>
          Project Artifacts
        </div>

        {files.map((f) => (
          <button
            key={f.name}
            onClick={() => setSelectedFile(f.name)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '7px 10px',
              borderRadius: 6,
              background: selectedFile === f.name ? '#24252B' : 'transparent',
              border: selectedFile === f.name ? '1px solid #383A42' : '1px solid transparent',
              color: selectedFile === f.name ? '#FFFFFF' : '#A1A1AA',
              fontSize: 11.5,
              fontWeight: selectedFile === f.name ? 600 : 400,
              textAlign: 'left',
              cursor: 'pointer',
              transition: 'all 0.1s ease',
            }}
          >
            <FileCode size={13} color={selectedFile === f.name ? '#38BDF8' : '#71717A'} />
            <span style={{ fontFamily: 'monospace' }}>{f.name}</span>
          </button>
        ))}
      </div>

      {/* Code Editor / Preview Pane */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Editor Header Bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '8px 14px',
            background: '#16171B',
            borderBottom: '1px solid #26272D',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: '#F4F4F5', fontFamily: 'monospace' }}>
              {selectedFile}
            </span>
            <span style={{ fontSize: 10, color: '#71717A' }}>
              {files.find((f) => f.name === selectedFile)?.desc}
            </span>
          </div>

          <div style={{ display: 'flex', gap: 6 }}>
            <button
              onClick={handleCopy}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 9px',
                borderRadius: 5,
                background: '#222329',
                border: '1px solid #33343C',
                color: copied ? '#4ADE80' : '#D4D4D8',
                fontSize: 11,
                cursor: 'pointer',
              }}
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>

            <button
              onClick={handleDownload}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 9px',
                borderRadius: 5,
                background: '#D35432',
                border: 'none',
                color: '#FFFFFF',
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              <Download size={12} />
              <span>Download</span>
            </button>
          </div>
        </div>

        {/* Code Content */}
        <div style={{ flex: 1, overflow: 'auto', padding: '12px 16px', background: '#0D0E11' }}>
          <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: 12, lineHeight: 1.55, color: '#E4E4E7' }}>
            <code>{currentContent}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};

