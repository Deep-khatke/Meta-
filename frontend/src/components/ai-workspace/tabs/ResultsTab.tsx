import React from 'react';
import {
  Download,
  FileCode,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
  Layers,
  Activity,
  Cpu
} from 'lucide-react';
import { Project, SimulationState } from '../../../types';

interface ResultsTabProps {
  project: Project;
  simState: SimulationState;
  onShowToast?: (msg: string) => void;
}

export const ResultsTab: React.FC<ResultsTabProps> = ({
  project,
  simState,
  onShowToast,
}) => {
  const handleExportONNX = () => {
    const dummy = `// ONNX Model export metadata\n{\n  "ir_version": 8,\n  "producer_name": "RoboForge-MuJoCo",\n  "graph": { "name": "PPO_Policy_Network", "inputs": 15, "outputs": 4 }\n}`;
    const blob = new Blob([dummy], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${project?.name || 'roboforge'}_policy.onnx`;
    a.click();
    URL.revokeObjectURL(url);
    onShowToast?.('Exported trained policy to ONNX format');
  };

  const handleExportROS2 = () => {
    const pyCode = `#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class RoboForgeDeployNode(Node):
    def __init__(self):
        super().__init__('roboforge_policy_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.get_logger().info('RoboForge Autonomous Policy Node Initialized')

    def scan_cb(self, msg):
        pass

    def odom_cb(self, msg):
        pass

def main(args=None):
    rclpy.init(args=args)
    node = RoboForgeDeployNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
`;
    const blob = new Blob([pyCode], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `roboforge_ros2_node.py`;
    a.click();
    URL.revokeObjectURL(url);
    onShowToast?.('Exported deployable ROS 2 Python Node');
  };

  const benchmarks = [
    { name: 'Nominal Baseline (MuJoCo 3.17)', friction: '1.0x', payload: '0 kg', noise: '0.0', success: '96.8%', error: '0.042m', status: 'pass' },
    { name: 'Domain Randomized: High Friction', friction: '1.8x', payload: '0 kg', noise: '0.0', success: '94.2%', error: '0.058m', status: 'pass' },
    { name: 'Domain Randomized: Low Friction', friction: '0.4x', payload: '0 kg', noise: '0.0', success: '89.6%', error: '0.081m', status: 'pass' },
    { name: 'Payload Drift Stress Test', friction: '1.0x', payload: '+2.5 kg', noise: '0.0', success: '91.4%', error: '0.067m', status: 'pass' },
    { name: 'Sensor Gaussian Noise Injection', friction: '1.0x', payload: '0 kg', noise: 'σ=0.05m', success: '93.5%', error: '0.054m', status: 'pass' },
  ];

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        overflowY: 'auto',
        padding: '18px 20px',
        backgroundColor: '#101113',
        color: '#F5F5F5',
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
      }}
    >
      {/* Export Bar */}
      <div
        style={{
          borderRadius: 12,
          padding: '16px 18px',
          background: 'linear-gradient(135deg, #1C1D22 0%, #16171B 100%)',
          border: '1px solid #303138',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
        }}
      >
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: '#FFFFFF' }}>Sim-to-Real Deployment & Results</div>
          <div style={{ fontSize: 11.5, color: '#A1A1AA', marginTop: 2 }}>
            Empirical robustness verification across randomized dynamics & hardware deployment targets
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={handleExportONNX}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 12px',
              borderRadius: 6,
              background: '#24252B',
              border: '1px solid #383A42',
              color: '#38BDF8',
              fontSize: 11.5,
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            <Download size={12} />
            <span>Export ONNX</span>
          </button>

          <button
            onClick={handleExportROS2}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 12px',
              borderRadius: 6,
              background: 'linear-gradient(135deg, #D35432 0%, #A8381C 100%)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              color: '#FFFFFF',
              fontSize: 11.5,
              fontWeight: 700,
              cursor: 'pointer',
              boxShadow: '0 2px 10px rgba(211, 84, 50, 0.4)',
            }}
          >
            <FileCode size={12} />
            <span>Export ROS 2 Node</span>
          </button>
        </div>
      </div>

      {/* Sim-to-Real Robustness Matrix Table */}
      <div
        style={{
          background: '#16171B',
          border: '1px solid #2B2C32',
          borderRadius: 10,
          overflow: 'hidden',
        }}
      >
        <div style={{ padding: '12px 16px', borderBottom: '1px solid #2B2C32', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 12.5, fontWeight: 700, color: '#FFFFFF' }}>
            Sim-to-Real Robustness Scorecard
          </span>
          <span style={{ fontSize: 11, color: '#4ADE80', fontWeight: 600 }}>5/5 Benchmark Suites Passed</span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11.5, textAlign: 'left' }}>
            <thead>
              <tr style={{ background: '#1A1B20', borderBottom: '1px solid #26272E', color: '#A1A1AA' }}>
                <th style={{ padding: '8px 14px' }}>Scenario</th>
                <th style={{ padding: '8px 12px' }}>Friction</th>
                <th style={{ padding: '8px 12px' }}>Payload</th>
                <th style={{ padding: '8px 12px' }}>Sensor Noise</th>
                <th style={{ padding: '8px 12px' }}>Success Rate</th>
                <th style={{ padding: '8px 12px' }}>Mean Error</th>
                <th style={{ padding: '8px 14px', textAlign: 'right' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((b, idx) => (
                <tr
                  key={idx}
                  style={{
                    borderBottom: '1px solid #222329',
                    backgroundColor: idx % 2 === 0 ? 'transparent' : 'rgba(255, 255, 255, 0.015)',
                  }}
                >
                  <td style={{ padding: '9px 14px', fontWeight: 600, color: '#E4E4E7' }}>{b.name}</td>
                  <td style={{ padding: '9px 12px', color: '#A1A1AA' }}>{b.friction}</td>
                  <td style={{ padding: '9px 12px', color: '#A1A1AA' }}>{b.payload}</td>
                  <td style={{ padding: '9px 12px', color: '#A1A1AA' }}>{b.noise}</td>
                  <td style={{ padding: '9px 12px', color: '#4ADE80', fontWeight: 700 }}>{b.success}</td>
                  <td style={{ padding: '9px 12px', color: '#38BDF8', fontFamily: 'monospace' }}>{b.error}</td>
                  <td style={{ padding: '9px 14px', textAlign: 'right' }}>
                    <span
                      style={{
                        padding: '2px 7px',
                        borderRadius: 100,
                        background: 'rgba(74, 222, 128, 0.15)',
                        color: '#4ADE80',
                        fontSize: 10,
                        fontWeight: 700,
                        textTransform: 'uppercase',
                      }}
                    >
                      {b.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

