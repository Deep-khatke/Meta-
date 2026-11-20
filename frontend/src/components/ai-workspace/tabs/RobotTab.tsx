import React from 'react';
import {
  Wrench,
  Cpu,
  Layers,
  Radio,
  ExternalLink,
  ShieldCheck,
  CheckCircle2,
  Sliders,
  Plane,
  Car,
  Compass
} from 'lucide-react';
import { Project, SimulationState } from '../../../types';

interface RobotTabProps {
  project: Project;
  simState: SimulationState;
  onOpenInEngineeringStudio: () => void;
}

export const RobotTab: React.FC<RobotTabProps> = ({
  project,
  simState,
  onOpenInEngineeringStudio,
}) => {
  const generic = project?.generic_robot;
  const standardRobot = project?.robot;
  const activeObjects = project?.objects || [];

  // Detect which robot is in the scene
  const droneObj = activeObjects.find((o) => o.name.toLowerCase().includes('drone') || o.id.startsWith('drone'));
  const hasGeneric = Boolean(generic && generic.joints && generic.joints.length > 0);
  const isDrone = Boolean(droneObj);
  const robotName = hasGeneric
    ? (generic?.name || 'Articulated Manipulator')
    : isDrone
    ? (droneObj?.name || 'ScoutDrone-01')
    : (standardRobot?.name || 'ForgeBot-DD1');

  const robotType = hasGeneric
    ? '6-DOF Articulated Manipulator'
    : isDrone
    ? 'Quadcopter UAV (Aerodynamic 4-Rotor Dynamics)'
    : 'Differential-Drive Ground Rover';

  const mass = hasGeneric
    ? generic?.links?.reduce((acc, l) => acc + (l.inertial?.mass || 1.0), 0) || 4.5
    : isDrone
    ? (droneObj?.mass || 1.2)
    : (standardRobot?.chassis_mass || 8.5);

  const dof = hasGeneric
    ? generic?.joints?.length || 6
    : isDrone
    ? 6
    : 2;

  const joints = hasGeneric
    ? generic?.joints || []
    : isDrone
    ? [
        { name: 'rotor_0', type: 'continuous', limit: { lower: -100, upper: 100 } },
        { name: 'rotor_1', type: 'continuous', limit: { lower: -100, upper: 100 } },
        { name: 'rotor_2', type: 'continuous', limit: { lower: -100, upper: 100 } },
        { name: 'rotor_3', type: 'continuous', limit: { lower: -100, upper: 100 } },
      ]
    : [
        { name: 'left_wheel_joint', type: 'continuous', limit: { lower: -100, upper: 100 } },
        { name: 'right_wheel_joint', type: 'continuous', limit: { lower: -100, upper: 100 } },
      ];

  const sensors = [
    { name: 'Simulated 2D/3D Raycast LiDAR', spec: '16 Beams / 360° FOV (8.0m max range)', active: true },
    { name: '6-DOF Inertial Measurement Unit (IMU)', spec: '3-Axis Gyro + 3-Axis Accelerometer', active: true },
    { name: 'Onboard RGB Vision Camera', spec: '65° Horizontal FOV / Forward Nose Mount', active: true },
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
      {/* Header Card */}
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
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: 10,
              background: 'linear-gradient(135deg, #D35432 0%, #9C381C 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 14px rgba(211, 84, 50, 0.35)',
            }}
          >
            {isDrone ? <Plane size={22} color="#FFF" /> : hasGeneric ? <Wrench size={22} color="#FFF" /> : <Car size={22} color="#FFF" />}
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#FFFFFF' }}>{robotName}</div>
            <div style={{ fontSize: 11.5, color: '#A1A1AA', marginTop: 2 }}>{robotType}</div>
          </div>
        </div>

        <button
          onClick={onOpenInEngineeringStudio}
          title="Open Robot Studio in Engineering mode to actuate joints and tune limits"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 14px',
            borderRadius: 6,
            background: 'rgba(211, 84, 50, 0.15)',
            border: '1px solid rgba(211, 84, 50, 0.4)',
            color: '#D35432',
            fontSize: 11.5,
            fontWeight: 700,
            cursor: 'pointer',
          }}
        >
          <span>Inspect in Robot Studio</span>
          <ExternalLink size={12} />
        </button>
      </div>

      {/* Primary Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
        <div
          style={{
            padding: '12px 14px',
            background: '#16171B',
            border: '1px solid #2B2C32',
            borderRadius: 8,
          }}
        >
          <div style={{ fontSize: 10, color: '#71717A', textTransform: 'uppercase', fontWeight: 700 }}>Total Mass</div>
          <div style={{ fontSize: 18, fontWeight: 800, color: '#F4F4F5', marginTop: 3 }}>{mass.toFixed(1)} kg</div>
          <div style={{ fontSize: 10, color: '#38BDF8', marginTop: 2 }}>Auto Inertia Tensor</div>
        </div>

        <div
          style={{
            padding: '12px 14px',
            background: '#16171B',
            border: '1px solid #2B2C32',
            borderRadius: 8,
          }}
        >
          <div style={{ fontSize: 10, color: '#71717A', textTransform: 'uppercase', fontWeight: 700 }}>Degrees of Freedom</div>
          <div style={{ fontSize: 18, fontWeight: 800, color: '#F4F4F5', marginTop: 3 }}>{dof} DoF</div>
          <div style={{ fontSize: 10, color: '#4ADE80', marginTop: 2 }}>{joints.length} Actuated Joints</div>
        </div>

        <div
          style={{
            padding: '12px 14px',
            background: '#16171B',
            border: '1px solid #2B2C32',
            borderRadius: 8,
          }}
        >
          <div style={{ fontSize: 10, color: '#71717A', textTransform: 'uppercase', fontWeight: 700 }}>Physics Core</div>
          <div style={{ fontSize: 18, fontWeight: 800, color: '#F4F4F5', marginTop: 3 }}>MuJoCo 3.17</div>
          <div style={{ fontSize: 10, color: '#F59E0B', marginTop: 2 }}>implicitfast solver</div>
        </div>
      </div>

      {/* Joint Kinematics Breakdown */}
      <div
        style={{
          background: '#16171B',
          border: '1px solid #2B2C32',
          borderRadius: 10,
          padding: '14px 16px',
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 700, color: '#FFFFFF', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
          <Sliders size={13} color="#D35432" />
          <span>Joint Topologies & Limits</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {joints.map((j: any, idx: number) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 10px',
                background: '#1E1F24',
                borderRadius: 6,
                fontSize: 11.5,
              }}
            >
              <span style={{ fontWeight: 600, color: '#E4E4E7' }}>{j.name}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: 10, color: '#A1A1AA', textTransform: 'capitalize' }}>{j.type}</span>
                <span style={{ fontSize: 10, color: '#71717A', fontFamily: 'monospace' }}>
                  {j.limit ? `[${j.limit.lower ?? -3.14}, ${j.limit.upper ?? 3.14}]` : '±∞'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Attached Sensors Suite */}
      <div
        style={{
          background: '#16171B',
          border: '1px solid #2B2C32',
          borderRadius: 10,
          padding: '14px 16px',
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 700, color: '#FFFFFF', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
          <Radio size={13} color="#38BDF8" />
          <span>Active Sensor Telemetry</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {sensors.map((s, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '7px 10px',
                background: '#1E1F24',
                borderRadius: 6,
                fontSize: 11.5,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <CheckCircle2 size={13} color="#4ADE80" />
                <span style={{ fontWeight: 600, color: '#E4E4E7' }}>{s.name}</span>
              </div>
              <span style={{ fontSize: 10.5, color: '#94A3B8' }}>{s.spec}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

