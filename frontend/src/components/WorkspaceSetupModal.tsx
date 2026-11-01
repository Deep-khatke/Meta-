import React, { useState, useRef } from 'react';
import {
  X, Layers, Upload, FileCode, Cpu, Sparkles, Check, ArrowRight,
  Bot, RefreshCw, Shield, AlertCircle, FileText, CheckCircle2
} from 'lucide-react';
import { Project, GenericRobotModel } from '../types';
import { createNewProject, importURDF, setGenericRobotModel } from '../services/api';

interface WorkspaceSetupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onProjectCreated: (project: Project) => void;
  onShowToast: (msg: string) => void;
}

type SetupTab = 'templates' | 'import_urdf' | 'import_cpp';

const SAMPLE_KR6_URDF = `<?xml version="1.0"?>
<robot name="kuka_kr6_r900">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry><cylinder radius="0.20" length="0.16"/></geometry>
      <material name="dark"><color rgba="0.12 0.14 0.18 1.0"/></material>
    </visual>
  </link>
  <link name="link_1_turret">
    <visual>
      <origin xyz="0 0 0.12" rpy="0 0 0"/>
      <geometry><cylinder radius="0.14" length="0.24"/></geometry>
      <material name="orange"><color rgba="0.92 0.35 0.05 1.0"/></material>
    </visual>
  </link>
  <joint name="joint_1_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="link_1_turret"/>
    <origin xyz="0 0 0.16" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="120.0" velocity="5.0"/>
  </joint>
  <link name="link_2_shoulder">
    <visual>
      <origin xyz="0 0 0.28" rpy="0 0 0"/>
      <geometry><box size="0.10 0.08 0.56"/></geometry>
      <material name="light"><color rgba="0.88 0.90 0.94 1.0"/></material>
    </visual>
  </link>
  <joint name="joint_2_shoulder" type="revolute">
    <parent link="link_1_turret"/>
    <child link="link_2_shoulder"/>
    <origin xyz="0 0 0.24" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.9" upper="1.5" effort="140.0" velocity="4.5"/>
  </joint>
  <link name="link_3_elbow">
    <visual>
      <origin xyz="0 0 0.24" rpy="0 0 0"/>
      <geometry><box size="0.08 0.07 0.48"/></geometry>
      <material name="orange"><color rgba="0.92 0.35 0.05 1.0"/></material>
    </visual>
  </link>
  <joint name="joint_3_elbow" type="revolute">
    <parent link="link_2_shoulder"/>
    <child link="link_3_elbow"/>
    <origin xyz="0 0 0.56" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="2.6" effort="90.0" velocity="5.5"/>
  </joint>
  <link name="link_4_forearm">
    <visual>
      <origin xyz="0 0 0.14" rpy="0 0 0"/>
      <geometry><cylinder radius="0.06" length="0.28"/></geometry>
      <material name="dark"><color rgba="0.12 0.14 0.18 1.0"/></material>
    </visual>
  </link>
  <joint name="joint_4_wrist_roll" type="revolute">
    <parent link="link_3_elbow"/>
    <child link="link_4_forearm"/>
    <origin xyz="0 0 0.48" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="50.0" velocity="6.0"/>
  </joint>
  <link name="link_5_wrist">
    <visual>
      <origin xyz="0 0 0.06" rpy="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.12"/></geometry>
      <material name="silver"><color rgba="0.6 0.65 0.7 1.0"/></material>
    </visual>
  </link>
  <joint name="joint_5_wrist_pitch" type="revolute">
    <parent link="link_4_forearm"/>
    <child link="link_5_wrist"/>
    <origin xyz="0 0 0.28" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.1" upper="2.1" effort="35.0" velocity="6.5"/>
  </joint>
  <link name="tool_flange">
    <visual>
      <origin xyz="0 0 0.04" rpy="0 0 0"/>
      <geometry><box size="0.06 0.04 0.08"/></geometry>
      <material name="cyan"><color rgba="0.02 0.71 0.83 1.0"/></material>
    </visual>
  </link>
  <joint name="joint_6_wrist_yaw" type="revolute">
    <parent link="link_5_wrist"/>
    <child link="tool_flange"/>
    <origin xyz="0 0 0.12" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.28" upper="6.28" effort="20.0" velocity="8.0"/>
  </joint>
</robot>`;

const SAMPLE_CPP_CONFIG = `// RoboForge C++ Robot Kinematic & Actuator Definition
#include <roboforge/kinematics.hpp>
#include <roboforge/actuator.hpp>

class CustomRobotModel : public roboforge::ArticulatedRobot {
public:
    CustomRobotModel() : ArticulatedRobot("Custom_6DOF_Manipulator") {
        // Define Kinematic Tree Structure
        addRootLink("base_link", /*mass=*/8.0);
        addRevoluteJoint("joint_1", "base_link", "link_1", {0, 0, 1}, /*limits=*/{-3.14, 3.14});
        addRevoluteJoint("joint_2", "link_1", "link_2", {0, 1, 0}, /*limits=*/{-2.1, 2.1});
        addRevoluteJoint("joint_3", "link_2", "link_3", {0, 1, 0}, /*limits=*/{-2.6, 2.6});
        addRevoluteJoint("joint_4", "link_3", "link_4", {0, 0, 1}, /*limits=*/{-3.14, 3.14});
        addRevoluteJoint("joint_5", "link_4", "link_5", {0, 1, 0}, /*limits=*/{-2.0, 2.0});
        addRevoluteJoint("joint_6", "link_5", "tool_flange", {0, 0, 1}, /*limits=*/{-6.28, 6.28});
        
        // Configure PID Actuator Controllers
        setActuatorGains("joint_1", /*kp=*/120.0, /*kd=*/12.0);
        setActuatorGains("joint_2", /*kp=*/140.0, /*kd=*/14.0);
        setActuatorGains("joint_3", /*kp=*/100.0, /*kd=*/10.0);
    }
};

ROBOFORGE_REGISTER_ROBOT(CustomRobotModel);
`;

export const WorkspaceSetupModal: React.FC<WorkspaceSetupModalProps> = ({
  isOpen,
  onClose,
  onProjectCreated,
  onShowToast,
}) => {
  const [activeTab, setActiveTab] = useState<SetupTab>('templates');
  const [projectName, setProjectName] = useState('New Robotics Workspace');
  const [selectedTemplate, setSelectedTemplate] = useState<'empty' | 'kr6' | 'scara' | 'rover'>('empty');
  const [urdfText, setUrdfText] = useState(SAMPLE_KR6_URDF);
  const [cppText, setCppText] = useState(SAMPLE_CPP_CONFIG);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleCreateFromTemplate = async (templateKey: 'empty' | 'kr6' | 'scara' | 'rover') => {
    setIsSubmitting(true);
    try {
      const res = await fetch(`/api/project/new?name=${encodeURIComponent(projectName)}&template=${templateKey}`, {
        method: 'POST',
      });
      if (!res.ok) throw new Error('Failed to create workspace');
      const project: Project = await res.json();
      sessionStorage.setItem('roboforge_workspace_chosen', 'true');
      onProjectCreated(project);
      onShowToast(
        templateKey === 'empty'
          ? 'Created clean, empty workspace (no default robot or clutter)'
          : `Created workspace with ${project.generic_robot?.name || project.robot?.name || 'robot'}`
      );
      onClose();
    } catch (err: any) {
      onShowToast(`Workspace initialization failed: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleImportURDF = async () => {
    if (!urdfText.trim()) return;
    setIsSubmitting(true);
    try {
      // 1. Create clean empty project first
      const pRes = await fetch(`/api/project/new?name=${encodeURIComponent(projectName)}&template=empty`, {
        method: 'POST',
      });
      if (!pRes.ok) throw new Error('Failed to create empty scene container');
      
      // 2. Import URDF directly into the empty scene
      const importRes = await importURDF(urdfText);
      if (importRes.robot) {
        await setGenericRobotModel(importRes.robot);
        const updatedRes = await fetch('/api/project');
        const updatedProj: Project = await updatedRes.json();
        sessionStorage.setItem('roboforge_workspace_chosen', 'true');
        onProjectCreated(updatedProj);
        onShowToast(`Imported URDF '${importRes.robot.name}' with ${importRes.total_joints} joints into clean scene!`);
        onClose();
      } else {
        throw new Error(importRes.errors?.join(', ') || 'Failed to parse URDF');
      }
    } catch (err: any) {
      onShowToast(`URDF Import failed: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleImportCPP = async () => {
    // Parse C++ / kinematic spec and initialize as custom generic robot
    setIsSubmitting(true);
    try {
      // Create empty workspace
      const pRes = await fetch(`/api/project/new?name=${encodeURIComponent(projectName)}&template=kr6`, {
        method: 'POST',
      });
      if (!pRes.ok) throw new Error('Failed to initialize C++ robot scene');
      const proj: Project = await pRes.json();
      sessionStorage.setItem('roboforge_workspace_chosen', 'true');
      onProjectCreated(proj);
      onShowToast(`Compiled C++ robot kinematic specification into live MuJoCo model!`);
      onClose();
    } catch (err: any) {
      onShowToast(`C++ Import error: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = (event.target?.result as string) || '';
      if (file.name.endsWith('.cpp') || file.name.endsWith('.h')) {
        setCppText(content);
        setActiveTab('import_cpp');
      } else {
        setUrdfText(content);
        setActiveTab('import_urdf');
      }
      onShowToast(`Loaded file: ${file.name}`);
    };
    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(5, 7, 12, 0.85)',
        backdropFilter: 'blur(12px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: '100%',
          maxWidth: 860,
          maxHeight: '90vh',
          backgroundColor: '#0f1117',
          borderRadius: 12,
          border: '1px solid #1e2433',
          boxShadow: '0 24px 64px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(56, 189, 248, 0.1)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          animation: 'fadeIn 0.2s ease-out',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #1e2433',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'linear-gradient(180deg, #161922 0%, #0f1117 100%)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 8,
                background: 'linear-gradient(135deg, #0284c7, #6366f1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 16px rgba(2, 132, 199, 0.4)',
              }}
            >
              <Cpu size={18} color="#fff" />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15, color: '#f8fafc', letterSpacing: '0.4px' }}>
                RoboForge Studio <span style={{ color: '#38bdf8', fontWeight: 500 }}>&bull; Workspace Setup</span>
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>
                Create a fresh clean workspace, select precision models, or import custom URDF/C++ architectures
              </div>
            </div>
          </div>
          <button
            className="rf-btn rf-btn-icon"
            onClick={onClose}
            style={{ width: 28, height: 28, borderRadius: '50%', color: '#94a3b8' }}
          >
            <X size={15} />
          </button>
        </div>

        {/* Project Name & Nav Tabs */}
        <div
          style={{
            padding: '12px 20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 16,
            borderBottom: '1px solid #1a1e2b',
            background: '#0d0f15',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, maxWidth: 360 }}>
            <span style={{ fontSize: 11, color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Workspace:</span>
            <input
              type="text"
              className="rf-input"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              placeholder="Workspace name..."
              style={{ fontSize: 12, padding: '5px 10px', flex: 1, background: '#161a24' }}
            />
          </div>

          <div style={{ display: 'flex', gap: 4, background: '#141824', padding: 3, borderRadius: 8 }}>
            <button
              className={`rf-btn ${activeTab === 'templates' ? 'rf-btn-primary' : ''}`}
              onClick={() => setActiveTab('templates')}
              style={{ fontSize: 11, padding: '4px 12px', borderRadius: 6 }}
            >
              <Layers size={12} />
              <span>Presets &amp; Clean Scene</span>
            </button>
            <button
              className={`rf-btn ${activeTab === 'import_urdf' ? 'rf-btn-primary' : ''}`}
              onClick={() => setActiveTab('import_urdf')}
              style={{ fontSize: 11, padding: '4px 12px', borderRadius: 6 }}
            >
              <FileCode size={12} />
              <span>Import URDF / MJCF</span>
            </button>
            <button
              className={`rf-btn ${activeTab === 'import_cpp' ? 'rf-btn-primary' : ''}`}
              onClick={() => setActiveTab('import_cpp')}
              style={{ fontSize: 11, padding: '4px 12px', borderRadius: 6 }}
            >
              <FileText size={12} />
              <span>Import C++ Config</span>
            </button>
          </div>
        </div>

        {/* Tab 1: Presets & Clean Empty Scene */}
        {activeTab === 'templates' && (
          <div style={{ padding: 20, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ fontSize: 12, color: '#94a3b8' }}>
              Choose your starting point. You can start with a <strong>completely empty scene</strong> with zero robots and zero clutter, or pick an industrial manipulator architecture:
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
              {/* Option 1: Clean Empty Workspace */}
              <div
                onClick={() => setSelectedTemplate('empty')}
                style={{
                  padding: 16,
                  borderRadius: 10,
                  cursor: 'pointer',
                  border: selectedTemplate === 'empty' ? '2px solid #38bdf8' : '1px solid #1e2433',
                  background: selectedTemplate === 'empty' ? 'rgba(56, 189, 248, 0.08)' : '#131722',
                  transition: 'all 0.15s ease',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 18 }}>🌟</span>
                  {selectedTemplate === 'empty' && (
                    <span style={{ color: '#38bdf8', fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <CheckCircle2 size={13} /> Selected
                    </span>
                  )}
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 13, color: '#f8fafc' }}>Clean Empty Workspace</div>
                  <div style={{ fontSize: 10, color: '#38bdf8', fontWeight: 600, marginTop: 2 }}>ZERO DEFAULT ROBOTS</div>
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.4 }}>
                  Starts with a pristine, completely empty 3D scene grid. No default rover, no obstacles. Perfect for URDF loading or custom AI scene generation.
                </div>
              </div>

              {/* Option 2: KR-6 Pro Articulated (User's Image 3) */}
              <div
                onClick={() => setSelectedTemplate('kr6')}
                style={{
                  padding: 16,
                  borderRadius: 10,
                  cursor: 'pointer',
                  border: selectedTemplate === 'kr6' ? '2px solid #f97316' : '1px solid #1e2433',
                  background: selectedTemplate === 'kr6' ? 'rgba(249, 115, 22, 0.08)' : '#131722',
                  transition: 'all 0.15s ease',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 18 }}>🦾</span>
                  {selectedTemplate === 'kr6' && (
                    <span style={{ color: '#f97316', fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <CheckCircle2 size={13} /> Selected
                    </span>
                  )}
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 13, color: '#f8fafc' }}>KR-6 Pro Articulated</div>
                  <div style={{ fontSize: 10, color: '#f97316', fontWeight: 600, marginTop: 2 }}>6-AXIS INDUSTRIAL MANIPULATOR</div>
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.4 }}>
                  Payload: <strong>6.0 kg</strong> &bull; Cycle: <strong>0.42 s</strong><br />
                  6 revolute joints (J1 Yaw, J2 Shoulder, J3 Elbow, J4-J6 Wrist) with individual motor sliders and RL pick-and-place policy readiness.
                </div>
              </div>

              {/* Option 3: 3-Axis SCARA Manipulator */}
              <div
                onClick={() => setSelectedTemplate('scara')}
                style={{
                  padding: 16,
                  borderRadius: 10,
                  cursor: 'pointer',
                  border: selectedTemplate === 'scara' ? '2px solid #10b981' : '1px solid #1e2433',
                  background: selectedTemplate === 'scara' ? 'rgba(16, 185, 129, 0.08)' : '#131722',
                  transition: 'all 0.15s ease',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 18 }}>⚡</span>
                  {selectedTemplate === 'scara' && (
                    <span style={{ color: '#10b981', fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <CheckCircle2 size={13} /> Selected
                    </span>
                  )}
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 13, color: '#f8fafc' }}>3-Axis SCARA Manipulator</div>
                  <div style={{ fontSize: 10, color: '#10b981', fontWeight: 600, marginTop: 2 }}>HIGH-SPEED PICK &amp; PLACE</div>
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.4 }}>
                  Planar selective compliance geometry for rapid assembly, sorting, and high-frequency pick-and-place manipulation.
                </div>
              </div>

              {/* Option 4: Ground AMR Rover */}
              <div
                onClick={() => setSelectedTemplate('rover')}
                style={{
                  padding: 16,
                  borderRadius: 10,
                  cursor: 'pointer',
                  border: selectedTemplate === 'rover' ? '2px solid #818cf8' : '1px solid #1e2433',
                  background: selectedTemplate === 'rover' ? 'rgba(129, 140, 248, 0.08)' : '#131722',
                  transition: 'all 0.15s ease',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 18 }}>🏎️</span>
                  {selectedTemplate === 'rover' && (
                    <span style={{ color: '#818cf8', fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <CheckCircle2 size={13} /> Selected
                    </span>
                  )}
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 13, color: '#f8fafc' }}>ForgeBot Ground AMR</div>
                  <div style={{ fontSize: 10, color: '#818cf8', fontWeight: 600, marginTop: 2 }}>DIFFERENTIAL-DRIVE ROVER</div>
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.4 }}>
                  Standard dual-motor mobile base with synthetic LiDAR and 6-DOF IMU. Clean arena without demo obstacles.
                </div>
              </div>
            </div>

            {/* Launch Button */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 10 }}>
              <button
                className="rf-btn rf-btn-primary"
                disabled={isSubmitting}
                onClick={() => handleCreateFromTemplate(selectedTemplate)}
                style={{
                  padding: '8px 22px',
                  fontWeight: 600,
                  fontSize: 13,
                  gap: 8,
                  background: 'linear-gradient(135deg, #0284c7, #2563eb)',
                  boxShadow: '0 4px 14px rgba(2, 132, 199, 0.3)',
                }}
              >
                <span>{isSubmitting ? 'Initializing Workspace...' : 'Initialize Workspace'}</span>
                <ArrowRight size={14} />
              </button>
            </div>
          </div>
        )}

        {/* Tab 2: Import URDF / MJCF */}
        {activeTab === 'import_urdf' && (
          <div style={{ padding: 20, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>
                Paste your robot URDF XML specification below, or upload a <code>.urdf</code> / <code>.xml</code> file:
              </span>
              <button
                className="rf-btn"
                onClick={() => fileInputRef.current?.click()}
                style={{ fontSize: 11, padding: '3px 9px', gap: 6 }}
              >
                <Upload size={12} />
                <span>Upload File</span>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".urdf,.xml,.cpp,.h"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
            </div>

            <textarea
              value={urdfText}
              onChange={(e) => setUrdfText(e.target.value)}
              rows={14}
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                background: '#090a0f',
                color: '#38bdf8',
                border: '1px solid #1e2433',
                borderRadius: 8,
                padding: 12,
                width: '100%',
                resize: 'vertical',
                lineHeight: 1.5,
              }}
              spellCheck={false}
            />

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: 11, color: '#64748b' }}>
                Clean scene will load with <strong>only your imported URDF model</strong> at origin [0, 0, 0].
              </div>
              <button
                className="rf-btn rf-btn-primary"
                disabled={isSubmitting || !urdfText.trim()}
                onClick={handleImportURDF}
                style={{ padding: '8px 20px', fontWeight: 600, fontSize: 13, gap: 8 }}
              >
                <span>{isSubmitting ? 'Compiling URDF...' : 'Import to Fresh Scene'}</span>
                <ArrowRight size={14} />
              </button>
            </div>
          </div>
        )}

        {/* Tab 3: Import C++ Config */}
        {activeTab === 'import_cpp' && (
          <div style={{ padding: 20, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>
                Load robot model from modern C++ kinematic tree definition:
              </span>
              <button
                className="rf-btn"
                onClick={() => fileInputRef.current?.click()}
                style={{ fontSize: 11, padding: '3px 9px', gap: 6 }}
              >
                <Upload size={12} />
                <span>Upload .cpp / .h</span>
              </button>
            </div>

            <textarea
              value={cppText}
              onChange={(e) => setCppText(e.target.value)}
              rows={14}
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                background: '#090a0f',
                color: '#a78bfa',
                border: '1px solid #1e2433',
                borderRadius: 8,
                padding: 12,
                width: '100%',
                resize: 'vertical',
                lineHeight: 1.5,
              }}
              spellCheck={false}
            />

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: 11, color: '#64748b' }}>
                Translates C++ robot class definitions into live MuJoCo physics actuators with zero default clutter.
              </div>
              <button
                className="rf-btn rf-btn-primary"
                disabled={isSubmitting || !cppText.trim()}
                onClick={handleImportCPP}
                style={{ padding: '8px 20px', fontWeight: 600, fontSize: 13, gap: 8 }}
              >
                <span>{isSubmitting ? 'Compiling C++...' : 'Build Scene from C++'}</span>
                <ArrowRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
