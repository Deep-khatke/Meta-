import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import * as THREE from 'three';
import { OrbitControls } from 'three-stdlib';
import { AurixScrollShowcase } from '../components/AurixScrollShowcase';
import {
  BrainCircuit, Sparkles, Shield, Cpu, Activity, ArrowRight,
  BookOpen, Code2, Layers, CheckCircle2, ChevronRight, X, Copy,
  Play, Pause, RotateCcw, Eye, Orbit, Zap, Box, Compass, RefreshCw
} from 'lucide-react';

// ─── Design Tokens (Teenage Engineering & Industrial Studio) ─────────────────
const C = {
  ivory: '#F1EEE7',
  ivoryLight: '#FBF9F5',
  ivoryDark: '#E4DFC7',
  graphite: '#17191A',
  graphiteLight: '#24282B',
  black: '#202222',
  red: '#C4472D',
  redGlow: 'rgba(196,71,45,0.35)',
  cyan: '#2DD4BF',
  muted: '#858580',
  aluminium: '#C8C9C7',
  gold: '#D4AF37',
};

// ─── Robot Metadata ───────────────────────────────────────────────────────────
interface RobotMeta {
  id: string;
  name: string;
  category: string;
  dof: string;
  payload: string;
  cycleTime: string;
  description: string;
  features: string[];
}

const ROBOT_CATALOG: RobotMeta[] = [
  {
    id: 'kuka',
    name: 'KR-6 Pro Articulated',
    category: '6-Axis Industrial Manipulator',
    dof: '6-DOF Revolute',
    payload: '6.0 kg',
    cycleTime: '0.42 s',
    description: 'Autonomous dual-object bi-directional transfer. Picks Object 1 and places it across in the opposite direction, then picks Object 2 and transfers it in the other direction.',
    features: ['Dual-Object Bi-Directional Pick & Place', 'Synchronized Opposing Transfer Arcs', 'True Contact Physics & Kinematic Clamping'],
  },
  {
    id: 'scara',
    name: 'SR-850 Assembly SCARA',
    category: '4-Axis Planar Selective Compliance',
    dof: '4-DOF (2R + 1P + 1R)',
    payload: '5.0 kg',
    cycleTime: '0.24 s',
    description: 'Precision SCARA executing continuous 6-object micro-component pick & place from Feeder Pallet to PCB Sockets with 4-DOF inverse kinematics.',
    features: ['Continuous 6-Object Pick & Place Cycle', 'Real-Time 4-DOF SCARA Kinematics', 'Silicone Vacuum Suction & Contact Sensing'],
  },
  {
    id: 'delta',
    name: 'D3-1200 FlexPicker Delta',
    category: '3-Axis Parallel Delta Kinematics',
    dof: '3 Parallel Arm Pairs',
    payload: '3.0 kg',
    cycleTime: '0.15 s',
    description: 'High-speed Delta robot executing rapid in-flight pick & place from a continuous moving conveyor belt to an outfeed matrix packaging tray with 15G Cartesian dynamics.',
    features: ['Dynamic Moving Conveyor Part Tracking', 'High-G Parabolic Pick & Place Transfer', 'Overhead Industrial Gantry & Rigid Linkages'],
  },
  {
    id: 'mobile',
    name: 'AMR-4 Industrial Rover',
    category: 'Autonomous Mobile Robot (AGV)',
    dof: '3-DOF Planar (X, Y, θ)',
    payload: '50.0 kg',
    cycleTime: '2.2 m/s',
    description: 'Heavy-duty AGV with steel chassis, rolling drive wheels, front/rear caster swivels, and 360° spinning LiDAR scanner with live laser fan.',
    features: ['True Wheel Rolling Physics', 'Active 360° Planar LiDAR Ray Fan', 'Safety Perimeter Status LED Ring'],
  },
];

// ─── Shared PBR Materials Factory ─────────────────────────────────────────────
function createStudioMaterials(isWireframe = false) {
  return {
    castIronDark: new THREE.MeshStandardMaterial({
      color: 0x1f2224, roughness: 0.38, metalness: 0.82, wireframe: isWireframe
    }),
    castAluminiumLight: new THREE.MeshStandardMaterial({
      color: 0xd2d2cd, roughness: 0.22, metalness: 0.88, wireframe: isWireframe
    }),
    ceramicShell: new THREE.MeshStandardMaterial({
      color: 0xf4f3ee, roughness: 0.25, metalness: 0.08, wireframe: isWireframe
    }),
    vermilionAccent: new THREE.MeshStandardMaterial({
      color: 0xc4472d, roughness: 0.25, metalness: 0.45, wireframe: isWireframe
    }),
    chromeShaft: new THREE.MeshStandardMaterial({
      color: 0xf5f5f5, roughness: 0.1, metalness: 0.95, wireframe: isWireframe
    }),
    machinedBrass: new THREE.MeshStandardMaterial({
      color: 0xd4af37, roughness: 0.22, metalness: 0.85, wireframe: isWireframe
    }),
    rubberTread: new THREE.MeshStandardMaterial({
      color: 0x141517, roughness: 0.85, metalness: 0.05, wireframe: isWireframe
    }),
    laserCyan: new THREE.MeshBasicMaterial({
      color: 0x2dd4bf, transparent: true, opacity: 0.65
    }),
    laserRed: new THREE.MeshBasicMaterial({
      color: 0xc4472d, transparent: true, opacity: 0.75
    }),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Accurate 6-DOF Articulated Robot Arm (Universal Robots / KUKA Architecture)
// ─────────────────────────────────────────────────────────────────────────────
function buildAccurateKUKAArm(scene: THREE.Scene, isWireframe = false) {
  const m = createStudioMaterials(isWireframe);
  const root = new THREE.Group();

  // 1. Heavy Base Mounting Plate with 4 Bolt Mounts
  const baseFlange = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.25, 0.06, 36), m.castIronDark);
  baseFlange.position.y = 0.03;
  baseFlange.castShadow = true;
  root.add(baseFlange);

  // 4 Corner Mounting Bolt Details
  for (let b = 0; b < 4; b++) {
    const ang = (b / 4) * Math.PI * 2 + Math.PI / 4;
    const bolt = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.02, 12), m.chromeShaft);
    bolt.position.set(Math.cos(ang) * 0.19, 0.065, Math.sin(ang) * 0.19);
    root.add(bolt);
  }

  const baseAccentRing = new THREE.Mesh(new THREE.CylinderGeometry(0.23, 0.23, 0.012, 36), m.vermilionAccent);
  baseAccentRing.position.y = 0.065;
  root.add(baseAccentRing);

  // 2. Joint 1: Turntable (Yaw around Y-axis)
  const j1Turntable = new THREE.Group();
  j1Turntable.position.y = 0.07;
  root.add(j1Turntable);

  const j1Housing = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.175, 0.14, 32), m.castIronDark);
  j1Housing.position.y = 0.07;
  j1Housing.castShadow = true;
  j1Turntable.add(j1Housing);

  // Shoulder Fork Housing (Single-sided heavy cast bracket)
  const shoulderFork = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.18, 0.22), m.castAluminiumLight);
  shoulderFork.position.set(0, 0.18, 0.03);
  shoulderFork.castShadow = true;
  j1Turntable.add(shoulderFork);

  // 3. Joint 2: Shoulder Pitch (Pitch around Z-axis)
  const j2Pivot = new THREE.Group();
  j2Pivot.position.set(0, 0.22, 0.03);
  j1Turntable.add(j2Pivot);

  const j2BearingCap = new THREE.Mesh(new THREE.CylinderGeometry(0.095, 0.095, 0.22, 28), m.castIronDark);
  j2BearingCap.rotation.x = Math.PI / 2;
  j2BearingCap.castShadow = true;
  j2Pivot.add(j2BearingCap);

  const j2VermilionRing = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.015, 28), m.vermilionAccent);
  j2VermilionRing.rotation.x = Math.PI / 2;
  j2VermilionRing.position.z = 0.112;
  j2Pivot.add(j2VermilionRing);

  // Link 1: Upper Arm Bicep (Extruded structural casting)
  const l1Bicep = new THREE.Group();
  j2Pivot.add(l1Bicep);

  const l1Body = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.075, 0.46, 24), m.ceramicShell);
  l1Body.position.y = 0.23;
  l1Body.castShadow = true;
  l1Bicep.add(l1Body);

  // Backside Cable Guide Conduit
  const l1Conduit = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.44, 0.04), m.castIronDark);
  l1Conduit.position.set(-0.06, 0.23, 0);
  l1Bicep.add(l1Conduit);

  // 4. Joint 3: Elbow Pitch (Pitch around Z-axis)
  const j3Pivot = new THREE.Group();
  j3Pivot.position.set(0, 0.46, 0);
  l1Bicep.add(j3Pivot);

  const j3Housing = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.18, 24), m.castIronDark);
  j3Housing.rotation.x = Math.PI / 2;
  j3Housing.castShadow = true;
  j3Pivot.add(j3Housing);

  const j3Accent = new THREE.Mesh(new THREE.CylinderGeometry(0.084, 0.084, 0.015, 24), m.vermilionAccent);
  j3Accent.rotation.x = Math.PI / 2;
  j3Accent.position.z = 0.092;
  j3Pivot.add(j3Accent);

  // Link 2: Forearm Link
  const l2Forearm = new THREE.Group();
  j3Pivot.add(l2Forearm);

  const l2Body = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, 0.38, 24), m.ceramicShell);
  l2Body.position.y = 0.19;
  l2Body.castShadow = true;
  l2Forearm.add(l2Body);

  // Aluminum Service Panel on Forearm
  const l2Plate = new THREE.Mesh(new THREE.BoxGeometry(0.095, 0.18, 0.08), m.castAluminiumLight);
  l2Plate.position.y = 0.18;
  l2Forearm.add(l2Plate);

  // 5. Joint 4: Wrist 1 (Pitch around Z-axis)
  const j4Pivot = new THREE.Group();
  j4Pivot.position.set(0, 0.38, 0);
  l2Forearm.add(j4Pivot);

  const j4Bearing = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.14, 20), m.castIronDark);
  j4Bearing.rotation.x = Math.PI / 2;
  j4Pivot.add(j4Bearing);

  // 6. Joint 5: Wrist 2 (Yaw around Y-axis)
  const j5Pivot = new THREE.Group();
  j5Pivot.position.set(0, 0.06, 0);
  j4Pivot.add(j5Pivot);

  const j5Housing = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.08), m.castAluminiumLight);
  j5Housing.position.y = 0.04;
  j5Pivot.add(j5Housing);

  // 7. Joint 6: Wrist 3 / Tool Flange (Roll around centerline)
  const j6Flange = new THREE.Group();
  j6Flange.position.set(0, 0.08, 0);
  j5Pivot.add(j6Flange);

  const flangePlate = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.048, 0.02, 24), m.chromeShaft);
  flangePlate.position.y = 0.01;
  j6Flange.add(flangePlate);

  // 8. End-Effector: Robotiq-Style 2-Finger Adaptive Gripper
  const gripperPalm = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.04, 0.05), m.castIronDark);
  gripperPalm.position.y = 0.04;
  gripperPalm.castShadow = true;
  j6Flange.add(gripperPalm);

  // Optical Contact / Laser Proximity Sensor Dot
  const gripperLed = new THREE.Mesh(new THREE.SphereGeometry(0.007, 12, 12), m.laserRed);
  gripperLed.position.set(0, 0.045, 0.026);
  j6Flange.add(gripperLed);

  // Precise Kinematic Grip Center (located right between the rubber contact pads)
  const gripCenter = new THREE.Group();
  gripCenter.position.set(0, 0.13, 0);
  j6Flange.add(gripCenter);

  // Finger Left Linkage
  const fingerLeft = new THREE.Group();
  fingerLeft.position.set(0.034, 0.06, 0);
  j6Flange.add(fingerLeft);

  const fingerLBar = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.08, 0.022), m.castAluminiumLight);
  fingerLBar.position.y = 0.04;
  fingerLBar.castShadow = true;
  fingerLeft.add(fingerLBar);

  const padLeft = new THREE.Mesh(new THREE.BoxGeometry(0.014, 0.03, 0.024), m.vermilionAccent);
  padLeft.position.set(0.002, 0.07, 0);
  fingerLeft.add(padLeft);

  // Finger Right Linkage
  const fingerRight = new THREE.Group();
  fingerRight.position.set(-0.034, 0.06, 0);
  j6Flange.add(fingerRight);

  const fingerRBar = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.08, 0.022), m.castAluminiumLight);
  fingerRBar.position.y = 0.04;
  fingerRBar.castShadow = true;
  fingerRight.add(fingerRBar);

  const padRight = new THREE.Mesh(new THREE.BoxGeometry(0.014, 0.03, 0.024), m.vermilionAccent);
  padRight.position.set(-0.002, 0.07, 0);
  fingerRight.add(padRight);

  // ── Two Distinct Physics Objects on the Table ──
  // Object 1: Precision Machined Brass Cube
  const object1 = new THREE.Mesh(new THREE.BoxGeometry(0.046, 0.046, 0.046), m.machinedBrass);
  object1.castShadow = true;
  root.add(object1);

  const obj1Cross = new THREE.Mesh(new THREE.RingGeometry(0.008, 0.015, 16), m.vermilionAccent);
  obj1Cross.rotation.x = -Math.PI / 2;
  obj1Cross.position.y = 0.0232;
  object1.add(obj1Cross);

  // Object 2: Precision Anodized Cyan / Cobalt Cylindrical Component
  const object2 = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.046, 24), m.laserCyan);
  object2.castShadow = true;
  root.add(object2);

  const obj2Cap = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.008, 24), m.castAluminiumLight);
  obj2Cap.position.y = 0.012;
  object2.add(obj2Cap);

  // ── Dual Workstations (East & West) with 2 Precision Nesting Bays Each ──
  // Station East (+X, Right Side) positioned with generous clearance so the arm can reach freely
  const stationEast = new THREE.Group();
  stationEast.position.set(0.63, 0, 0);
  root.add(stationEast);

  const eastBase = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.08, 0.46), m.castIronDark);
  eastBase.position.y = 0.04;
  eastBase.castShadow = true;
  stationEast.add(eastBase);

  // Nest E1 (Gold / Vermilion, Z = +0.16)
  const nestE1 = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.06, 0.015, 24), m.vermilionAccent);
  nestE1.position.set(0, 0.085, 0.16);
  stationEast.add(nestE1);

  const ringE1 = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.045, 24), m.machinedBrass);
  ringE1.rotation.x = -Math.PI / 2;
  ringE1.position.set(0, 0.093, 0.16);
  stationEast.add(ringE1);

  // Nest E2 (Cyan / Steel, Z = -0.16)
  const nestE2 = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.06, 0.015, 24), m.castIronDark);
  nestE2.position.set(0, 0.085, -0.16);
  stationEast.add(nestE2);

  const ringE2 = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.045, 24), m.laserCyan);
  ringE2.rotation.x = -Math.PI / 2;
  ringE2.position.set(0, 0.093, -0.16);
  stationEast.add(ringE2);

  // Station West (-X, Left Side) positioned symmetrically with full arm clearance
  const stationWest = new THREE.Group();
  stationWest.position.set(-0.63, 0, 0);
  root.add(stationWest);

  const westBase = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.08, 0.46), m.castIronDark);
  westBase.position.y = 0.04;
  westBase.castShadow = true;
  stationWest.add(westBase);

  // Nest W1 (Gold / Vermilion, Z = +0.16)
  const nestW1 = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.06, 0.015, 24), m.castIronDark);
  nestW1.position.set(0, 0.085, 0.16);
  stationWest.add(nestW1);

  const ringW1 = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.045, 24), m.machinedBrass);
  ringW1.rotation.x = -Math.PI / 2;
  ringW1.position.set(0, 0.093, 0.16);
  stationWest.add(ringW1);

  // Nest W2 (Cyan / Steel, Z = -0.16)
  const nestW2 = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.06, 0.015, 24), m.vermilionAccent);
  nestW2.position.set(0, 0.085, -0.16);
  stationWest.add(nestW2);

  const ringW2 = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.045, 24), m.laserCyan);
  ringW2.rotation.x = -Math.PI / 2;
  ringW2.position.set(0, 0.093, -0.16);
  stationWest.add(ringW2);

  // Toolpath Trajectory Spline Arcs (Opposing Directions, Open Free-Flight Radius R ~ 0.66m)
  // Arc 1: Westbound Transit (East Bay 1 -> West Bay 1, Z > 0)
  const arcPts1: THREE.Vector3[] = [];
  for (let i = 0; i <= 36; i++) {
    const f = i / 36;
    const yaw = -0.202 + f * (-2.852 - (-0.202));
    const r = 0.66;
    arcPts1.push(new THREE.Vector3(
      r * Math.cos(yaw),
      0.42 + Math.sin(f * Math.PI) * 0.06,
      -r * Math.sin(yaw)
    ));
  }
  const arcLine1 = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(arcPts1),
    new THREE.LineDashedMaterial({ color: 0xd4af37, dashSize: 0.04, gapSize: 0.025, transparent: true, opacity: 0.65 })
  );
  arcLine1.computeLineDistances();
  root.add(arcLine1);

  // Arc 2: Eastbound Transit (West Bay 2 -> East Bay 2, Z < 0)
  const arcPts2: THREE.Vector3[] = [];
  for (let i = 0; i <= 36; i++) {
    const f = i / 36;
    const yaw = -3.345 + f * (-5.985 - (-3.345));
    const r = 0.66;
    arcPts2.push(new THREE.Vector3(
      r * Math.cos(yaw),
      0.42 + Math.sin(f * Math.PI) * 0.06,
      -r * Math.sin(yaw)
    ));
  }
  const arcLine2 = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(arcPts2),
    new THREE.LineDashedMaterial({ color: 0x2dd4bf, dashSize: 0.04, gapSize: 0.025, transparent: true, opacity: 0.65 })
  );
  arcLine2.computeLineDistances();
  root.add(arcLine2);

  scene.add(root);
  root.scale.setScalar(1.25);

  // Pre-allocated vectors for exact TCP kinematics
  const tcpWorld = new THREE.Vector3();
  const obj1Pos = new THREE.Vector3();
  const obj2Pos = new THREE.Vector3();

  // 4 Target Nest Coordinates (Wide clearance R = 0.65m)
  const posE1 = new THREE.Vector3(0.63, 0.116, 0.16);
  const posW1 = new THREE.Vector3(-0.63, 0.116, 0.16);
  const posW2 = new THREE.Vector3(-0.63, 0.116, -0.16);
  const posE2 = new THREE.Vector3(0.63, 0.116, -0.16);

  // Cubic Hermite smoothstep utility
  const smooth = (a: number, b: number, v: number) => {
    const x = Math.max(0, Math.min(1, (v - a) / (b - a)));
    return x * x * (3 - 2 * x);
  };

  return {
    root,
    update: (time: number, speedMultiplier: number) => {
      // 4-Phase Full Alternating Pick-and-Place Cycle: 24 seconds total
      const totalCycle = 24.0;
      const t = (time * 1.0 * speedMultiplier) % totalCycle;

      let j1Angle = -0.202;
      let heightParam = 0; // 0 = High Transit (y ~ 0.42m), 1 = Down at Nest Level (y ~ 0.116m)
      let gripWidth = 0.036; // Open: 0.036m, Clamped: 0.024m
      let statusText = 'TRAJECTORY TRACKING';
      let obj1Held = false;
      let obj2Held = false;

      if (t < 6.0) {
        // ── TRANSFER 1: Pick Object 1 at East Bay 1, Transfer to West Bay 1 (Westbound) ──
        const sub = t / 6.0;
        if (sub < 0.16) {
          // Approach East Bay 1
          j1Angle = -0.202;
          heightParam = smooth(0, 0.16, sub);
          gripWidth = 0.036;
          obj1Held = false;
          obj1Pos.copy(posE1);
          statusText = 'APPROACHING OBJECT 1 (EAST BAY 1)';
        } else if (sub < 0.30) {
          // Grip Object 1
          j1Angle = -0.202;
          heightParam = 1.0;
          const p = smooth(0.16, 0.30, sub);
          gripWidth = 0.036 - p * 0.012;
          obj1Held = true;
          statusText = 'GRIPPING OBJECT 1 · CONTACT FORCES ACTIVE (4.8 N)';
        } else if (sub < 0.44) {
          // Lift Object 1 to Transit Height
          j1Angle = -0.202;
          heightParam = 1.0 - smooth(0.30, 0.44, sub);
          gripWidth = 0.024;
          obj1Held = true;
          statusText = 'LIFTING OBJECT 1 (BRASS CUBE)';
        } else if (sub < 0.74) {
          // 180° Westbound Transit Swing across wide arc
          const p = smooth(0.44, 0.74, sub);
          j1Angle = -0.202 + p * (-2.852 - (-0.202));
          heightParam = 0.0;
          gripWidth = 0.024;
          obj1Held = true;
          statusText = 'TRANSFERRING WESTBOUND (180° FREE-FLIGHT)';
        } else if (sub < 0.86) {
          // Descend onto West Bay 1
          const p = smooth(0.74, 0.86, sub);
          j1Angle = -2.852;
          heightParam = p;
          gripWidth = 0.024;
          obj1Held = true;
          statusText = 'DEPOSITING OBJECT 1 ON WEST BAY 1';
        } else if (sub < 0.94) {
          // Release Clamp
          const p = smooth(0.86, 0.94, sub);
          j1Angle = -2.852;
          heightParam = 1.0;
          gripWidth = 0.024 + p * 0.012;
          obj1Held = false;
          obj1Pos.copy(posW1);
          statusText = 'OBJECT 1 PLACED · RELEASING CLAMP';
        } else {
          // Retract Arm
          const p = smooth(0.94, 1.0, sub);
          j1Angle = -2.852;
          heightParam = 1.0 - p;
          gripWidth = 0.036;
          obj1Held = false;
          obj1Pos.copy(posW1);
          statusText = 'OBJECT 1 DEPOSITED · RETRACTING';
        }

        // Object 2 rests at West Bay 2
        obj2Held = false;
        obj2Pos.copy(posW2);

      } else if (t < 12.0) {
        // ── TRANSFER 2: Pick Object 2 at West Bay 2, Transfer to East Bay 2 (Eastbound, Opposite Direction) ──
        const sub = (t - 6.0) / 6.0;
        if (sub < 0.14) {
          // Slew from West Bay 1 to West Bay 2
          const p = smooth(0, 0.14, sub);
          j1Angle = -2.852 + p * (-3.345 - (-2.852));
          heightParam = 0.0;
          gripWidth = 0.036;
          obj2Held = false;
          obj2Pos.copy(posW2);
          statusText = 'SLEWING TO OBJECT 2 (WEST BAY 2)';
        } else if (sub < 0.26) {
          // Descend onto Object 2
          const p = smooth(0.14, 0.26, sub);
          j1Angle = -3.345;
          heightParam = p;
          gripWidth = 0.036;
          obj2Held = false;
          obj2Pos.copy(posW2);
          statusText = 'APPROACHING OBJECT 2 (WEST BAY 2)';
        } else if (sub < 0.38) {
          // Grip Object 2
          const p = smooth(0.26, 0.38, sub);
          j1Angle = -3.345;
          heightParam = 1.0;
          gripWidth = 0.036 - p * 0.012;
          obj2Held = true;
          statusText = 'GRIPPING OBJECT 2 · CONTACT FORCES ACTIVE (4.2 N)';
        } else if (sub < 0.48) {
          // Lift Object 2 to Transit Height
          const p = smooth(0.38, 0.48, sub);
          j1Angle = -3.345;
          heightParam = 1.0 - p;
          gripWidth = 0.024;
          obj2Held = true;
          statusText = 'LIFTING OBJECT 2 (CYAN COLLAR)';
        } else if (sub < 0.74) {
          // 180° Eastbound Transit Swing in Opposite Direction
          const p = smooth(0.48, 0.74, sub);
          j1Angle = -3.345 + p * (-5.985 - (-3.345));
          heightParam = 0.0;
          gripWidth = 0.024;
          obj2Held = true;
          statusText = 'TRANSFERRING EASTBOUND (OPPOSITE DIRECTION)';
        } else if (sub < 0.86) {
          // Descend onto East Bay 2
          const p = smooth(0.74, 0.86, sub);
          j1Angle = -5.985;
          heightParam = p;
          gripWidth = 0.024;
          obj2Held = true;
          statusText = 'DEPOSITING OBJECT 2 ON EAST BAY 2';
        } else if (sub < 0.94) {
          // Release Clamp
          const p = smooth(0.86, 0.94, sub);
          j1Angle = -5.985;
          heightParam = 1.0;
          gripWidth = 0.024 + p * 0.012;
          obj2Held = false;
          obj2Pos.copy(posE2);
          statusText = 'OBJECT 2 PLACED · RELEASING CLAMP';
        } else {
          // Retract Arm
          const p = smooth(0.94, 1.0, sub);
          j1Angle = -5.985;
          heightParam = 1.0 - p;
          gripWidth = 0.036;
          obj2Held = false;
          obj2Pos.copy(posE2);
          statusText = 'OBJECT 2 DEPOSITED · RETRACTING';
        }

        // Object 1 rests at West Bay 1
        obj1Held = false;
        obj1Pos.copy(posW1);

      } else if (t < 18.0) {
        // ── TRANSFER 3: Return Object 1 from West Bay 1 back to East Bay 1 (Eastbound Return) ──
        const sub = (t - 12.0) / 6.0;
        if (sub < 0.14) {
          // Slew from East Bay 2 to West Bay 1
          const p = smooth(0, 0.14, sub);
          j1Angle = 0.298 + p * (-2.852 - 0.298);
          heightParam = 0.0;
          gripWidth = 0.036;
          obj1Held = false;
          obj1Pos.copy(posW1);
          statusText = 'SLEWING TO OBJECT 1 (WEST BAY 1)';
        } else if (sub < 0.26) {
          // Descend onto Object 1 at West Bay 1
          const p = smooth(0.14, 0.26, sub);
          j1Angle = -2.852;
          heightParam = p;
          gripWidth = 0.036;
          obj1Held = false;
          obj1Pos.copy(posW1);
          statusText = 'APPROACHING OBJECT 1 (WEST BAY 1)';
        } else if (sub < 0.38) {
          // Grip Object 1
          const p = smooth(0.26, 0.38, sub);
          j1Angle = -2.852;
          heightParam = 1.0;
          gripWidth = 0.036 - p * 0.012;
          obj1Held = true;
          statusText = 'GRIPPING OBJECT 1 · CONTACT DETECTED (4.6 N)';
        } else if (sub < 0.48) {
          // Lift Object 1
          const p = smooth(0.38, 0.48, sub);
          j1Angle = -2.852;
          heightParam = 1.0 - p;
          gripWidth = 0.024;
          obj1Held = true;
          statusText = 'LIFTING OBJECT 1 (RETURN PASS)';
        } else if (sub < 0.74) {
          // 180° Eastbound Return Swing
          const p = smooth(0.48, 0.74, sub);
          j1Angle = -2.852 + p * (-0.202 - (-2.852));
          heightParam = 0.0;
          gripWidth = 0.024;
          obj1Held = true;
          statusText = 'TRANSFERRING EASTBOUND (RETURN 180°)';
        } else if (sub < 0.86) {
          // Descend onto East Bay 1
          const p = smooth(0.74, 0.86, sub);
          j1Angle = -0.202;
          heightParam = p;
          gripWidth = 0.024;
          obj1Held = true;
          statusText = 'DEPOSITING OBJECT 1 ON EAST BAY 1';
        } else if (sub < 0.94) {
          // Release Clamp
          const p = smooth(0.86, 0.94, sub);
          j1Angle = -0.202;
          heightParam = 1.0;
          gripWidth = 0.024 + p * 0.012;
          obj1Held = false;
          obj1Pos.copy(posE1);
          statusText = 'OBJECT 1 PLACED ON EAST BAY 1';
        } else {
          // Retract Arm
          const p = smooth(0.94, 1.0, sub);
          j1Angle = -0.202;
          heightParam = 1.0 - p;
          gripWidth = 0.036;
          obj1Held = false;
          obj1Pos.copy(posE1);
          statusText = 'OBJECT 1 DEPOSITED · RETRACTING';
        }

        // Object 2 rests at East Bay 2
        obj2Held = false;
        obj2Pos.copy(posE2);

      } else {
        // ── TRANSFER 4: Return Object 2 from East Bay 2 back to West Bay 2 (Westbound Return) ──
        const sub = (t - 18.0) / 6.0;
        if (sub < 0.14) {
          // Slew from East Bay 1 to East Bay 2
          const p = smooth(0, 0.14, sub);
          j1Angle = -0.202 + p * (0.298 - (-0.202));
          heightParam = 0.0;
          gripWidth = 0.036;
          obj2Held = false;
          obj2Pos.copy(posE2);
          statusText = 'SLEWING TO OBJECT 2 (EAST BAY 2)';
        } else if (sub < 0.26) {
          // Descend onto Object 2 at East Bay 2
          const p = smooth(0.14, 0.26, sub);
          j1Angle = 0.298;
          heightParam = p;
          gripWidth = 0.036;
          obj2Held = false;
          obj2Pos.copy(posE2);
          statusText = 'APPROACHING OBJECT 2 (EAST BAY 2)';
        } else if (sub < 0.38) {
          // Grip Object 2
          const p = smooth(0.26, 0.38, sub);
          j1Angle = 0.298;
          heightParam = 1.0;
          gripWidth = 0.036 - p * 0.012;
          obj2Held = true;
          statusText = 'GRIPPING OBJECT 2 · CONTACT DETECTED (4.1 N)';
        } else if (sub < 0.48) {
          // Lift Object 2
          const p = smooth(0.38, 0.48, sub);
          j1Angle = 0.298;
          heightParam = 1.0 - p;
          gripWidth = 0.024;
          obj2Held = true;
          statusText = 'LIFTING OBJECT 2 (RETURN PASS)';
        } else if (sub < 0.74) {
          // 180° Westbound Return Swing in Opposite Direction
          const p = smooth(0.48, 0.74, sub);
          j1Angle = 0.298 + p * (2.938 - 0.298);
          heightParam = 0.0;
          gripWidth = 0.024;
          obj2Held = true;
          statusText = 'TRANSFERRING WESTBOUND (RETURN 180°)';
        } else if (sub < 0.86) {
          // Descend onto West Bay 2
          const p = smooth(0.74, 0.86, sub);
          j1Angle = 2.938;
          heightParam = p;
          gripWidth = 0.024;
          obj2Held = true;
          statusText = 'DEPOSITING OBJECT 2 ON WEST BAY 2';
        } else if (sub < 0.94) {
          // Release Clamp
          const p = smooth(0.86, 0.94, sub);
          j1Angle = 2.938;
          heightParam = 1.0;
          gripWidth = 0.024 + p * 0.012;
          obj2Held = false;
          obj2Pos.copy(posW2);
          statusText = 'OBJECT 2 PLACED ON WEST BAY 2';
        } else {
          // Retract Arm and seamless wrap towards East Bay 1
          const p = smooth(0.94, 1.0, sub);
          heightParam = 1.0 - p;
          gripWidth = 0.036;
          obj2Held = false;
          obj2Pos.copy(posW2);
          j1Angle = 2.938 + p * (6.081 - 2.938);
          statusText = 'CYCLE COMPLETE · REINITIALIZING';
        }

        // Object 1 rests at East Bay 1
        obj1Held = false;
        obj1Pos.copy(posE1);
      }

      // Apply Kinematic Rotations to Robot Links
      j1Turntable.rotation.y = j1Angle;

      // Generous, natural reach kinematics with open elbow and wide clearance:
      // When heightParam = 0 (Transit high): j2 = -1.40, j3 = 0.80, j4 = -2.54 (y ~ 0.42m, R ~ 0.66m)
      // When heightParam = 1 (Nest contact): j2 = -2.02, j3 = 1.34, j4 = -2.46 (y ~ 0.116m, R ~ 0.65m)
      // Tool axis always points vertically downward: j4 = -Math.PI - (j2 + j3)
      const j2Angle = -1.40 - heightParam * 0.62;
      const j3Angle = 0.80 + heightParam * 0.54;
      const j4Angle = -Math.PI - (j2Angle + j3Angle);
      j2Pivot.rotation.z = j2Angle;
      j3Pivot.rotation.z = j3Angle;
      j4Pivot.rotation.z = j4Angle;
      j5Pivot.rotation.y = 0;
      j6Flange.rotation.y = 0;

      // Finger Actuation
      fingerLeft.position.x = gripWidth;
      fingerRight.position.x = -gripWidth;

      // Optical LED tactile indicator
      if (obj1Held || obj2Held) {
        gripperLed.material = m.laserCyan;
      } else {
        gripperLed.material = m.laserRed;
      }

      // CRITICAL: Force matrix update before sampling gripCenter world coordinates
      root.updateMatrixWorld(true);
      gripCenter.getWorldPosition(tcpWorld);
      root.worldToLocal(tcpWorld);

      // Synchronize Carried Objects with Tool Grip Center
      if (obj1Held) {
        object1.position.set(tcpWorld.x, tcpWorld.y - 0.006, tcpWorld.z);
        object1.rotation.y = j1Angle;
      } else {
        object1.position.copy(obj1Pos);
        object1.rotation.y = 0;
      }

      if (obj2Held) {
        object2.position.set(tcpWorld.x, tcpWorld.y - 0.006, tcpWorld.z);
        object2.rotation.y = j1Angle;
      } else {
        object2.position.copy(obj2Pos);
        object2.rotation.y = 0;
      }

      return {
        j1: (j1Turntable.rotation.y * (180 / Math.PI)).toFixed(1),
        j2: (j2Pivot.rotation.z * (180 / Math.PI)).toFixed(1),
        j3: (j3Pivot.rotation.z * (180 / Math.PI)).toFixed(1),
        j4: (j4Pivot.rotation.z * (180 / Math.PI)).toFixed(1),
        tcp: `${tcpWorld.x.toFixed(2)}, ${tcpWorld.y.toFixed(2)}, ${tcpWorld.z.toFixed(2)}`,
        status: statusText,
      };
    },
    dispose: () => {
      scene.remove(root);
    }
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Accurate SCARA Robot (Epson / Yamaha Architecture)
// ─────────────────────────────────────────────────────────────────────────────
function buildAccurateSCARARobot(scene: THREE.Scene, isWireframe = false) {
  const m = createStudioMaterials(isWireframe);
  const root = new THREE.Group();

  // 1. Heavy Rigid Mounting Column with Rear Reinforcing Spine
  const columnBase = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.08, 0.3), m.castIronDark);
  columnBase.position.y = 0.04;
  columnBase.castShadow = true;
  root.add(columnBase);

  // 4 Corner Mounting Anchor Bolts
  for (let b = 0; b < 4; b++) {
    const ang = (b / 4) * Math.PI * 2 + Math.PI / 4;
    const bolt = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.02, 12), m.chromeShaft);
    bolt.position.set(Math.cos(ang) * 0.12, 0.085, Math.sin(ang) * 0.12);
    root.add(bolt);
  }

  const mainColumn = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 0.56, 32), m.castAluminiumLight);
  mainColumn.position.y = 0.32;
  mainColumn.castShadow = true;
  root.add(mainColumn);

  const columnRearRib = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.52, 0.16), m.castIronDark);
  columnRearRib.position.set(-0.09, 0.32, 0);
  root.add(columnRearRib);

  const topColumnCap = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.03, 32), m.vermilionAccent);
  topColumnCap.position.y = 0.61;
  root.add(topColumnCap);

  // 2. Arm 1 (J1: Rotates horizontally around vertical column axis, L1 = 0.38m)
  const arm1Group = new THREE.Group();
  arm1Group.position.y = 0.58;
  root.add(arm1Group);

  const arm1Casting = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.11, 0.14), m.ceramicShell);
  arm1Casting.position.x = 0.19;
  arm1Casting.castShadow = true;
  arm1Group.add(arm1Casting);

  const arm1PivotCap = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.12, 28), m.castIronDark);
  arm1PivotCap.position.x = 0.38;
  arm1Group.add(arm1PivotCap);

  const arm1AccentStripe = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.012, 0.142), m.vermilionAccent);
  arm1AccentStripe.position.set(0.19, 0.05, 0);
  arm1Group.add(arm1AccentStripe);

  // 3. Arm 2 (J2: Rotates horizontally around Arm 1 elbow, L2 = 0.32m)
  const arm2Group = new THREE.Group();
  arm2Group.position.set(0.38, 0, 0);
  arm1Group.add(arm2Group);

  const arm2Casting = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.09, 0.11), m.castAluminiumLight);
  arm2Casting.position.x = 0.16;
  arm2Casting.castShadow = true;
  arm2Group.add(arm2Casting);

  // J3 / J4 Spline Quill Housing
  const quillHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.15, 24), m.castIronDark);
  quillHousing.position.x = 0.32;
  arm2Group.add(quillHousing);

  const quillAccent = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.015, 24), m.vermilionAccent);
  quillAccent.position.set(0.32, 0.06, 0);
  arm2Group.add(quillAccent);

  // 4. J3 Linear Shaft & J4 Rotary Theta Axis
  const splineShaftGroup = new THREE.Group();
  splineShaftGroup.position.set(0.32, 0, 0);
  arm2Group.add(splineShaftGroup);

  // Ball-Screw Chrome Shaft with threaded ridges
  const chromeQuill = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.44, 20), m.chromeShaft);
  chromeQuill.position.y = -0.14;
  splineShaftGroup.add(chromeQuill);

  // Vacuum Tool Head & Silicone Suction Cup
  const vacuumTool = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.04, 16), m.castIronDark);
  vacuumTool.position.y = -0.36;
  splineShaftGroup.add(vacuumTool);

  const suctionCup = new THREE.Mesh(new THREE.ConeGeometry(0.026, 0.03, 16), m.vermilionAccent);
  suctionCup.rotation.x = Math.PI;
  suctionCup.position.y = -0.39;
  splineShaftGroup.add(suctionCup);

  // Active Vacuum Glow Ring at Suction Tip
  const vacuumGlow = new THREE.Mesh(new THREE.TorusGeometry(0.020, 0.004, 12, 24), m.laserCyan);
  vacuumGlow.rotation.x = Math.PI / 2;
  vacuumGlow.position.y = -0.404;
  vacuumGlow.visible = false;
  splineShaftGroup.add(vacuumGlow);

  // ── 5. Workstation Fixtures: Feeder Pallet (Source) & Target PCB (Destination) ──
  // Input Feeder Pallet (CNC Machined Aluminum matrix tray at Z < 0)
  const feederBase = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.04, 0.22), m.castAluminiumLight);
  feederBase.position.set(0.36, 0.02, -0.18);
  feederBase.castShadow = true;
  root.add(feederBase);

  const feederPlate = new THREE.Mesh(new THREE.BoxGeometry(0.248, 0.012, 0.228), m.castIronDark);
  feederPlate.position.set(0.36, 0.006, -0.18);
  root.add(feederPlate);

  const feederHeader = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.005, 0.018), m.vermilionAccent);
  feederHeader.position.set(0.36, 0.042, -0.281);
  root.add(feederHeader);

  // Destination PCB Assembly Workstation (FR-4 substrate with SMD landing pads at Z > 0)
  const pcbSubstrate = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.04, 0.22), m.castIronDark);
  pcbSubstrate.position.set(0.36, 0.02, 0.18);
  pcbSubstrate.castShadow = true;
  root.add(pcbSubstrate);

  const pcbRail1 = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.046, 0.228), m.castAluminiumLight);
  pcbRail1.position.set(0.236, 0.023, 0.18);
  root.add(pcbRail1);

  const pcbRail2 = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.046, 0.228), m.castAluminiumLight);
  pcbRail2.position.set(0.484, 0.023, 0.18);
  root.add(pcbRail2);

  const pcbGrid = new THREE.GridHelper(0.22, 10, 0x2dd4bf, 0x334155);
  pcbGrid.position.set(0.36, 0.041, 0.18);
  root.add(pcbGrid);

  // ── Coordinates for the 6 Feeder Slots and 6 PCB Sockets ──
  const sourcePos: THREE.Vector3[] = [
    new THREE.Vector3(0.31, 0.046, -0.23),
    new THREE.Vector3(0.31, 0.046, -0.18),
    new THREE.Vector3(0.31, 0.046, -0.13),
    new THREE.Vector3(0.41, 0.046, -0.23),
    new THREE.Vector3(0.41, 0.046, -0.18),
    new THREE.Vector3(0.41, 0.046, -0.13),
  ];

  const destPos: THREE.Vector3[] = [
    new THREE.Vector3(0.31, 0.046, 0.13),
    new THREE.Vector3(0.31, 0.046, 0.18),
    new THREE.Vector3(0.31, 0.046, 0.23),
    new THREE.Vector3(0.41, 0.046, 0.13),
    new THREE.Vector3(0.41, 0.046, 0.18),
    new THREE.Vector3(0.41, 0.046, 0.23),
  ];

  const sourceAngles = [0, 0, 0, 0, 0, 0];
  const destAngles = [0, Math.PI / 2, 0, Math.PI / 2, 0, Math.PI / 2];

  // Precision Recesses on Feeder & Sockets on PCB
  const pcbLeds: THREE.Mesh[] = [];
  const ledOffMat = new THREE.MeshStandardMaterial({ color: 0x181a1b, roughness: 0.8 });

  for (let i = 0; i < 6; i++) {
    // Feeder Nest Pocket
    const nestPocket = new THREE.Mesh(new THREE.BoxGeometry(0.044, 0.005, 0.044), m.castIronDark);
    nestPocket.position.set(sourcePos[i].x, 0.039, sourcePos[i].z);
    root.add(nestPocket);

    const nestBevel = new THREE.Mesh(new THREE.RingGeometry(0.016, 0.021, 16), m.machinedBrass);
    nestBevel.rotation.x = -Math.PI / 2;
    nestBevel.position.set(sourcePos[i].x, 0.042, sourcePos[i].z);
    root.add(nestBevel);

    // PCB SMD Socket Landing Frame
    const pcbSocket = new THREE.Mesh(new THREE.BoxGeometry(0.044, 0.004, 0.044), m.castIronDark);
    pcbSocket.position.set(destPos[i].x, 0.041, destPos[i].z);
    root.add(pcbSocket);

    const socketPads = new THREE.Mesh(new THREE.RingGeometry(0.018, 0.022, 16), m.machinedBrass);
    socketPads.rotation.x = -Math.PI / 2;
    socketPads.position.set(destPos[i].x, 0.043, destPos[i].z);
    root.add(socketPads);

    // Live Socket Status Indicator LED
    const led = new THREE.Mesh(new THREE.CylinderGeometry(0.0035, 0.0035, 0.003, 12), ledOffMat);
    led.position.set(destPos[i].x - 0.025, 0.043, destPos[i].z);
    root.add(led);
    pcbLeds.push(led);

    // Trajectory dashed flight path arc connecting Feeder Slot i to PCB Socket i
    const pts: THREE.Vector3[] = [];
    for (let s = 0; s <= 24; s++) {
      const f = s / 24;
      const arcX = (1 - f) * sourcePos[i].x + f * destPos[i].x + 0.035 * Math.sin(f * Math.PI);
      const arcZ = (1 - f) * sourcePos[i].z + f * destPos[i].z;
      const arcY = 0.048 + Math.sin(f * Math.PI) * 0.12;
      pts.push(new THREE.Vector3(arcX, arcY, arcZ));
    }
    const trajLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineDashedMaterial({ color: 0x2dd4bf, dashSize: 0.02, gapSize: 0.015, transparent: true, opacity: 0.35 })
    );
    trajLine.computeLineDistances();
    root.add(trajLine);
  }

  // ── 6. Six Distinct Physical Micro-Electronic Components ──
  const COMPONENT_SPECS = [
    { code: 'IC-01', name: 'MCU BGA-256' },
    { code: 'IC-02', name: 'DUAL INDUCTOR' },
    { code: 'IC-03', name: '6-DOF IMU CAN' },
    { code: 'IC-04', name: 'OPTO-ISOLATOR' },
    { code: 'IC-05', name: 'LPDDR5 MEMORY' },
    { code: 'IC-06', name: 'NEURAL NPU DIE' },
  ];

  const objects: THREE.Group[] = [];

  // Component 0: High-End Microcontroller (MCU BGA-256)
  const obj0 = new THREE.Group();
  const c0Body = new THREE.Mesh(new THREE.BoxGeometry(0.034, 0.009, 0.034), m.castIronDark);
  c0Body.castShadow = true;
  obj0.add(c0Body);
  const c0Cap = new THREE.Mesh(new THREE.BoxGeometry(0.024, 0.002, 0.024), m.chromeShaft);
  c0Cap.position.y = 0.005;
  obj0.add(c0Cap);
  const c0Pin1 = new THREE.Mesh(new THREE.SphereGeometry(0.0025, 8, 8), m.laserCyan);
  c0Pin1.position.set(-0.011, 0.006, -0.011);
  obj0.add(c0Pin1);
  objects.push(obj0);

  // Component 1: Dual Surface-Mount Power Inductor
  const obj1 = new THREE.Group();
  const c1Body = new THREE.Mesh(new THREE.BoxGeometry(0.036, 0.011, 0.026), m.castIronDark);
  c1Body.castShadow = true;
  obj1.add(c1Body);
  const c1CoilL = new THREE.Mesh(new THREE.CylinderGeometry(0.0055, 0.0055, 0.012, 16), m.vermilionAccent);
  c1CoilL.position.x = -0.008;
  obj1.add(c1CoilL);
  const c1CoilR = new THREE.Mesh(new THREE.CylinderGeometry(0.0055, 0.0055, 0.012, 16), m.vermilionAccent);
  c1CoilR.position.x = 0.008;
  obj1.add(c1CoilR);
  const c1TopPlate = new THREE.Mesh(new THREE.BoxGeometry(0.034, 0.002, 0.024), m.castIronDark);
  c1TopPlate.position.y = 0.006;
  obj1.add(c1TopPlate);
  objects.push(obj1);

  // Component 2: Precision 6-DOF MEMS IMU Sensor
  const obj2 = new THREE.Group();
  const c2Body = new THREE.Mesh(new THREE.BoxGeometry(0.028, 0.008, 0.028), m.machinedBrass);
  c2Body.castShadow = true;
  obj2.add(c2Body);
  const c2Can = new THREE.Mesh(new THREE.CylinderGeometry(0.010, 0.010, 0.004, 16), m.castAluminiumLight);
  c2Can.position.y = 0.005;
  obj2.add(c2Can);
  const c2Notch = new THREE.Mesh(new THREE.BoxGeometry(0.003, 0.003, 0.003), m.vermilionAccent);
  c2Notch.position.set(0.012, 0.004, 0);
  obj2.add(c2Notch);
  objects.push(obj2);

  // Component 3: High-Isolation Optocoupler Transceiver
  const obj3 = new THREE.Group();
  const c3Body = new THREE.Mesh(new THREE.BoxGeometry(0.032, 0.010, 0.026), m.ceramicShell);
  c3Body.castShadow = true;
  obj3.add(c3Body);
  const c3Lens = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.003, 16), m.laserCyan);
  c3Lens.position.y = 0.006;
  obj3.add(c3Lens);
  const c3Shield = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.002, 0.024), m.castIronDark);
  c3Shield.position.y = 0.0055;
  obj3.add(c3Shield);
  objects.push(obj3);

  // Component 4: High-Bandwidth LPDDR5 Memory Module
  const obj4 = new THREE.Group();
  const c4Body = new THREE.Mesh(new THREE.BoxGeometry(0.038, 0.008, 0.024), m.castIronDark);
  c4Body.castShadow = true;
  obj4.add(c4Body);
  const c4HeatStripe = new THREE.Mesh(new THREE.BoxGeometry(0.026, 0.002, 0.016), m.machinedBrass);
  c4HeatStripe.position.y = 0.005;
  obj4.add(c4HeatStripe);
  for (let pin = 0; pin < 4; pin++) {
    const px = -0.015 + pin * 0.010;
    const pMesh1 = new THREE.Mesh(new THREE.BoxGeometry(0.003, 0.004, 0.004), m.chromeShaft);
    pMesh1.position.set(px, 0.002, 0.013);
    obj4.add(pMesh1);
    const pMesh2 = new THREE.Mesh(new THREE.BoxGeometry(0.003, 0.004, 0.004), m.chromeShaft);
    pMesh2.position.set(px, 0.002, -0.013);
    obj4.add(pMesh2);
  }
  objects.push(obj4);

  // Component 5: Neural Edge AI Accelerator (NPU)
  const obj5 = new THREE.Group();
  const c5Body = new THREE.Mesh(new THREE.BoxGeometry(0.034, 0.010, 0.034), m.castIronDark);
  c5Body.castShadow = true;
  obj5.add(c5Body);
  const c5Bezel = new THREE.Mesh(new THREE.BoxGeometry(0.030, 0.002, 0.030), m.vermilionAccent);
  c5Bezel.position.y = 0.005;
  obj5.add(c5Bezel);
  const c5Die = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.003, 0.018), m.chromeShaft);
  c5Die.position.y = 0.0065;
  obj5.add(c5Die);
  objects.push(obj5);

  // Add all 6 objects to root scene
  for (let i = 0; i < 6; i++) {
    objects[i].position.copy(sourcePos[i]);
    objects[i].rotation.y = sourceAngles[i];
    root.add(objects[i]);
  }

  scene.add(root);
  root.scale.setScalar(1.3);

  // ── 7. Inverse Kinematics & Smoothstep Helpers ──
  const L1 = 0.38;
  const L2 = 0.32;

  function solveSCARA_IK(x: number, z: number) {
    const dSq = x * x + z * z;
    let cosJ2 = (dSq - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    cosJ2 = Math.max(-0.9999, Math.min(0.9999, cosJ2));
    const j2 = Math.acos(cosJ2);
    const phi = Math.atan2(-z, x);
    const beta = Math.atan2(L2 * Math.sin(j2), L1 + L2 * Math.cos(j2));
    const j1 = phi - beta;
    return { j1, j2 };
  }

  const smooth = (a: number, b: number, v: number) => {
    const x = Math.max(0, Math.min(1, (v - a) / (b - a)));
    return x * x * (3 - 2 * x);
  };

  return {
    root,
    update: (time: number, speedMultiplier: number) => {
      // Complete continuous 6-object bidirectional cycle: 33.6s total (2.8s per transfer)
      const STEP_TIME = 2.8;
      const TOTAL_CYCLE = STEP_TIME * 12; // 16.8s forward (Feeder -> PCB) + 16.8s reverse (PCB -> Feeder)
      const t = (time * 1.0 * speedMultiplier) % TOTAL_CYCLE;
      const isForward = t < TOTAL_CYCLE / 2;

      let activeStep = 0;
      let stepProg = 0;

      if (isForward) {
        activeStep = Math.min(5, Math.floor(t / STEP_TIME));
        stepProg = (t % STEP_TIME) / STEP_TIME;
      } else {
        const revT = t - TOTAL_CYCLE / 2;
        const revStep = Math.min(5, Math.floor(revT / STEP_TIME));
        activeStep = 5 - revStep;
        stepProg = (revT % STEP_TIME) / STEP_TIME;
      }

      const activeSpec = COMPONENT_SPECS[activeStep];
      const src = isForward ? sourcePos[activeStep] : destPos[activeStep];
      const dst = isForward ? destPos[activeStep] : sourcePos[activeStep];
      const srcAngle = isForward ? sourceAngles[activeStep] : destAngles[activeStep];
      const dstAngle = isForward ? destAngles[activeStep] : sourceAngles[activeStep];

      // Starting horizontal point & angle from previous cycle position
      let fromPos: THREE.Vector3;
      let fromAngle: number;
      if (isForward) {
        if (activeStep === 0) {
          fromPos = sourcePos[0];
          fromAngle = sourceAngles[0];
        } else {
          fromPos = destPos[activeStep - 1];
          fromAngle = destAngles[activeStep - 1];
        }
      } else {
        const revStep = 5 - activeStep;
        if (revStep === 0) {
          fromPos = destPos[5];
          fromAngle = destAngles[5];
        } else {
          fromPos = sourcePos[activeStep + 1];
          fromAngle = sourceAngles[activeStep + 1];
        }
      }

      let tcpX = src.x;
      let tcpZ = src.z;
      let quillY = 0;
      let j4Angle = srcAngle;
      let vacuumActive = false;
      let statusText = '';

      if (stepProg < 0.14) {
        // Phase 1: Transit across from previous fixture to source
        const p = smooth(0, 0.14, stepProg);
        tcpX = (1 - p) * fromPos.x + p * src.x;
        tcpZ = (1 - p) * fromPos.z + p * src.z;
        quillY = 0;
        j4Angle = (1 - p) * fromAngle + p * srcAngle;
        vacuumActive = false;
        statusText = isForward
          ? `APPROACHING FEEDER [${activeStep + 1}/6] · ${activeSpec.name}`
          : `RETRIEVING FROM PCB [${activeStep + 1}/6] · ${activeSpec.name}`;
      } else if (stepProg < 0.28) {
        // Phase 2: High-speed vertical Z plunge onto component
        const p = smooth(0.14, 0.28, stepProg);
        tcpX = src.x;
        tcpZ = src.z;
        quillY = -0.121 * p;
        j4Angle = srcAngle;
        vacuumActive = stepProg > 0.21;
        statusText = isForward
          ? `ENGAGING VACUUM · COMPONENT [${activeStep + 1}/6]`
          : `ENGAGING VACUUM · PCB SOCKET [${activeStep + 1}/6]`;
      } else if (stepProg < 0.42) {
        // Phase 3: Vertical lift of gripped component to transit clearance
        const p = smooth(0.28, 0.42, stepProg);
        tcpX = src.x;
        tcpZ = src.z;
        quillY = -0.121 * (1 - p);
        j4Angle = srcAngle;
        vacuumActive = true;
        statusText = isForward
          ? `LIFTING COMPONENT [${activeStep + 1}/6] (-85 kPa)`
          : `LIFTING COMPONENT [${activeStep + 1}/6] FROM PCB`;
      } else if (stepProg < 0.72) {
        // Phase 4: High-speed horizontal SCARA transfer arc across to destination
        const p = smooth(0.42, 0.72, stepProg);
        tcpX = (1 - p) * src.x + p * dst.x + 0.035 * Math.sin(p * Math.PI);
        tcpZ = (1 - p) * src.z + p * dst.z;
        quillY = 0;
        j4Angle = (1 - p) * srcAngle + p * dstAngle;
        vacuumActive = true;
        statusText = isForward
          ? `TRANSFERRING [${activeStep + 1}/6] TO PCB SOCKET`
          : `RETURNING [${activeStep + 1}/6] TO FEEDER PALLET`;
      } else if (stepProg < 0.86) {
        // Phase 5: High-speed vertical Z plunge to seat component in socket
        const p = smooth(0.72, 0.86, stepProg);
        tcpX = dst.x;
        tcpZ = dst.z;
        quillY = -0.121 * p;
        j4Angle = dstAngle;
        vacuumActive = stepProg < 0.81;
        statusText = isForward
          ? `SEATING COMPONENT [${activeStep + 1}/6] IN SOCKET`
          : `SEATING COMPONENT [${activeStep + 1}/6] IN FEEDER TRAY`;
      } else {
        // Phase 6: Retract quill, leaving component seated
        const p = smooth(0.86, 1.00, stepProg);
        tcpX = dst.x;
        tcpZ = dst.z;
        quillY = -0.121 * (1 - p);
        j4Angle = dstAngle;
        vacuumActive = false;
        statusText = isForward
          ? `COMPONENT [${activeStep + 1}/6] SEATED · VACUUM OFF`
          : `COMPONENT [${activeStep + 1}/6] RESTORED TO FEEDER`;
      }

      // ── Apply Inverse Kinematics to SCARA Robot Arms & Quill ──
      const { j1, j2 } = solveSCARA_IK(tcpX, tcpZ);
      arm1Group.rotation.y = j1;
      arm2Group.rotation.y = j2;
      splineShaftGroup.position.y = quillY;
      chromeQuill.rotation.y = j4Angle;
      vacuumGlow.visible = vacuumActive;

      // ── Synchronize Positions & States for all 6 Objects ──
      const toolTipWorldY = 0.58 + quillY - 0.408;

      for (let i = 0; i < 6; i++) {
        const obj = objects[i];
        if (isForward) {
          if (i < activeStep) {
            // Already placed on PCB
            obj.position.copy(destPos[i]);
            obj.rotation.y = destAngles[i];
            pcbLeds[i].material = m.laserCyan;
          } else if (i > activeStep) {
            // Waiting in Feeder
            obj.position.copy(sourcePos[i]);
            obj.rotation.y = sourceAngles[i];
            pcbLeds[i].material = ledOffMat;
          } else {
            // Active component i
            if (stepProg < 0.28) {
              obj.position.copy(sourcePos[i]);
              obj.rotation.y = sourceAngles[i];
              pcbLeds[i].material = ledOffMat;
            } else if (stepProg < 0.86) {
              // Attached to vacuum cup
              obj.position.set(tcpX, toolTipWorldY - 0.005, tcpZ);
              obj.rotation.y = j4Angle;
              pcbLeds[i].material = ledOffMat;
            } else {
              obj.position.copy(destPos[i]);
              obj.rotation.y = destAngles[i];
              pcbLeds[i].material = m.laserCyan;
            }
          }
        } else {
          // Reverse pass
          if (i > activeStep) {
            // Returned to Feeder
            obj.position.copy(sourcePos[i]);
            obj.rotation.y = sourceAngles[i];
            pcbLeds[i].material = ledOffMat;
          } else if (i < activeStep) {
            // Waiting on PCB
            obj.position.copy(destPos[i]);
            obj.rotation.y = destAngles[i];
            pcbLeds[i].material = m.laserCyan;
          } else {
            // Active component i
            if (stepProg < 0.28) {
              obj.position.copy(destPos[i]);
              obj.rotation.y = destAngles[i];
              pcbLeds[i].material = m.laserCyan;
            } else if (stepProg < 0.86) {
              // Attached to vacuum cup
              obj.position.set(tcpX, toolTipWorldY - 0.005, tcpZ);
              obj.rotation.y = j4Angle;
              pcbLeds[i].material = ledOffMat;
            } else {
              obj.position.copy(sourcePos[i]);
              obj.rotation.y = sourceAngles[i];
              pcbLeds[i].material = ledOffMat;
            }
          }
        }
      }

      return {
        j1: (j1 * (180 / Math.PI)).toFixed(1),
        j2: (j2 * (180 / Math.PI)).toFixed(1),
        j3: `${(Math.abs(quillY) * 1000).toFixed(0)} mm`,
        j4: `${(((j4Angle * (180 / Math.PI)) % 360 + 360) % 360).toFixed(0)}°`,
        tcp: `[${activeStep + 1}/6] ${activeSpec.code} ${activeSpec.name}`,
        status: statusText,
      };
    },
    dispose: () => {
      scene.remove(root);
    }
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Accurate Delta Spider Robot with Dynamic Ball-Joint Parallel Linkages
// ─────────────────────────────────────────────────────────────────────────────
function buildAccurateDeltaRobot(scene: THREE.Scene, isWireframe = false) {
  const m = createStudioMaterials(isWireframe);
  const root = new THREE.Group();

  // ── 1. Heavy Industrial Overhead Gantry Cage Frame ("Proper Robot Cell") ──
  const gantryGroup = new THREE.Group();
  root.add(gantryGroup);

  // 4 Rigid Vertical Corner Pillars with Anchor Footplates
  const pillarOffsets = [
    { x: -0.42, z: -0.42 },
    { x: 0.42, z: -0.42 },
    { x: -0.42, z: 0.42 },
    { x: 0.42, z: 0.42 },
  ];

  pillarOffsets.forEach(pos => {
    // Vertical tubular column
    const col = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1.08, 16), m.castIronDark);
    col.position.set(pos.x, 0.54, pos.z);
    col.castShadow = true;
    gantryGroup.add(col);

    // Floor mounting foot flange with anchor bolts
    const foot = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.014, 0.065), m.castAluminiumLight);
    foot.position.set(pos.x, 0.007, pos.z);
    gantryGroup.add(foot);

    // Top structural corner bracket
    const cap = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.04, 0.05), m.vermilionAccent);
    cap.position.set(pos.x, 1.06, pos.z);
    gantryGroup.add(cap);
  });

  // 4 Overhead Horizontal Framing Girders
  const girderNorth = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.03, 0.03), m.castAluminiumLight);
  girderNorth.position.set(0, 1.06, -0.42);
  gantryGroup.add(girderNorth);

  const girderSouth = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.03, 0.03), m.castAluminiumLight);
  girderSouth.position.set(0, 1.06, 0.42);
  gantryGroup.add(girderSouth);

  const girderWest = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 0.84), m.castAluminiumLight);
  girderWest.position.set(-0.42, 1.06, 0);
  gantryGroup.add(girderWest);

  const girderEast = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 0.84), m.castAluminiumLight);
  girderEast.position.set(0.42, 1.06, 0);
  gantryGroup.add(girderEast);

  // Top Corner Industrial Signal Stack Beacon Tower (Red/Amber/Green status LED)
  const beaconMast = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.12, 12), m.castAluminiumLight);
  beaconMast.position.set(-0.42, 1.14, 0.42);
  gantryGroup.add(beaconMast);

  const beaconGreen = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.022, 12), m.laserCyan);
  beaconGreen.position.set(-0.42, 1.21, 0.42);
  gantryGroup.add(beaconGreen);

  const beaconAmber = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.022, 12), m.machinedBrass);
  beaconAmber.position.set(-0.42, 1.234, 0.42);
  gantryGroup.add(beaconAmber);

  // ── 2. Overhead Delta Robot Actuation Base & 3 Servo Gearmotors ──
  const Y_BASE = 1.05;
  const R_BASE = 0.22;
  const R_EE = 0.07;
  const L_BICEP = 0.28;
  const L_ROD = 0.54;

  const topBase = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.42, 0.05, 3, 1), m.castIronDark);
  topBase.position.y = Y_BASE;
  topBase.castShadow = true;
  root.add(topBase);

  const topAccent = new THREE.Mesh(new THREE.CylinderGeometry(0.39, 0.39, 0.014, 3), m.vermilionAccent);
  topAccent.position.y = Y_BASE + 0.028;
  root.add(topAccent);

  // 3 Upper Actuated Direct-Drive Biceps (Analytical Closed-Form Delta IK)
  const bicepGroups: THREE.Group[] = [];
  const elbowHolders: THREE.Object3D[] = [];

  for (let i = 0; i < 3; i++) {
    const angle = (i / 3) * Math.PI * 2;
    const motorGroup = new THREE.Group();
    motorGroup.rotation.y = angle;
    motorGroup.position.y = Y_BASE;
    root.add(motorGroup);

    // High-Torque Servo Gearbox Housing with Heat Dissipation Fins
    const motorHousing = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.11, 0.13), m.castIronDark);
    motorHousing.position.set(R_BASE - 0.03, 0.02, 0);
    motorGroup.add(motorHousing);

    const motorCap = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.038, 0.135, 16), m.castAluminiumLight);
    motorCap.rotation.x = Math.PI / 2;
    motorCap.position.set(R_BASE - 0.03, 0.02, 0);
    motorGroup.add(motorCap);

    // Bicep Pivot Group (Rotates around Z in motor coordinates)
    const bicepPivot = new THREE.Group();
    bicepPivot.position.set(R_BASE, 0, 0);
    motorGroup.add(bicepPivot);
    bicepGroups.push(bicepPivot);

    // Cast Aluminum Bicep Arm (Length = L_BICEP)
    const bicepArm = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.018, L_BICEP, 16), m.castAluminiumLight);
    bicepArm.position.set(0, -L_BICEP / 2, 0);
    bicepArm.castShadow = true;
    bicepPivot.add(bicepArm);

    // Vermilion Trim Stripe along Upper Arm
    const bicepStripe = new THREE.Mesh(new THREE.BoxGeometry(0.008, L_BICEP * 0.7, 0.024), m.vermilionAccent);
    bicepStripe.position.set(0, -L_BICEP / 2, 0);
    bicepPivot.add(bicepStripe);

    // Elbow Dual-Ball Joint Crossbar
    const elbowBar = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.07, 12), m.chromeShaft);
    elbowBar.rotation.x = Math.PI / 2;
    elbowBar.position.set(0, -L_BICEP, 0);
    bicepPivot.add(elbowBar);

    // Dual Chrome Ball Studs at Elbow Ends
    const ballL = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 12), m.chromeShaft);
    ballL.position.set(0, -L_BICEP, 0.03);
    bicepPivot.add(ballL);

    const ballR = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 12), m.chromeShaft);
    ballR.position.set(0, -L_BICEP, -0.03);
    bicepPivot.add(ballR);

    // Reference marker for the elbow center
    const elbowPt = new THREE.Object3D();
    elbowPt.position.set(0, -L_BICEP, 0);
    bicepPivot.add(elbowPt);
    elbowHolders.push(elbowPt);
  }

  // ── 3. Lower Moving Tool Platform (End-Effector with Vacuum Gripper) ──
  const toolPlatform = new THREE.Group();
  toolPlatform.position.set(-0.22, 0.36, 0);
  root.add(toolPlatform);

  const eePlate = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.025, 3), m.castAluminiumLight);
  toolPlatform.add(eePlate);

  const eeCollar = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.03, 20), m.castIronDark);
  eeCollar.position.y = -0.02;
  toolPlatform.add(eeCollar);

  const eeShaft = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.10, 16), m.chromeShaft);
  eeShaft.position.y = -0.07;
  toolPlatform.add(eeShaft);

  // Silicone Bellows Vacuum Suction Cup
  const eeCup = new THREE.Mesh(new THREE.ConeGeometry(0.024, 0.028, 16), m.vermilionAccent);
  eeCup.rotation.x = Math.PI;
  eeCup.position.y = -0.126;
  toolPlatform.add(eeCup);

  // Active Vacuum Glow Ring at Suction Tip
  const vacuumGlow = new THREE.Mesh(new THREE.TorusGeometry(0.018, 0.0035, 12, 24), m.laserCyan);
  vacuumGlow.rotation.x = Math.PI / 2;
  vacuumGlow.position.y = -0.140;
  vacuumGlow.visible = false;
  toolPlatform.add(vacuumGlow);

  // 6 Rigid Carbon-Fiber Parallel Linkage Rods (Constant Length L_ROD = 0.54m)
  const parallelRods: THREE.Mesh[] = [];
  for (let r = 0; r < 6; r++) {
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.0065, 0.0065, 1.0, 12), m.castIronDark);
    rod.castShadow = true;
    root.add(rod);
    parallelRods.push(rod);
  }

  // ── 4. Continuous Infeed Belt Conveyor System (X = -0.22m, running along +Z) ──
  const conveyorGroup = new THREE.Group();
  conveyorGroup.position.set(-0.22, 0, 0);
  root.add(conveyorGroup);

  // Conveyor Bed Extrusion Frame
  const convFrame = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.045, 1.16), m.castAluminiumLight);
  convFrame.position.y = 0.142;
  convFrame.castShadow = true;
  conveyorGroup.add(convFrame);

  // 4 Support Legs to Floor
  const legZOffsets = [-0.48, 0.48];
  legZOffsets.forEach(lz => {
    const legL = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.12, 12), m.castIronDark);
    legL.position.set(-0.065, 0.06, lz);
    conveyorGroup.add(legL);

    const legR = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.12, 12), m.castIronDark);
    legR.position.set(0.065, 0.06, lz);
    conveyorGroup.add(legR);
  });

  // Dark High-Friction Moving Rubber Conveyor Belt
  const convBelt = new THREE.Mesh(new THREE.BoxGeometry(0.125, 0.006, 1.16), m.rubberTread);
  convBelt.position.y = 0.166;
  conveyorGroup.add(convBelt);

  // Chrome Drive Rollers at End Terminals
  const rollerFront = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.13, 20), m.chromeShaft);
  rollerFront.rotation.z = Math.PI / 2;
  rollerFront.position.set(0, 0.145, 0.58);
  conveyorGroup.add(rollerFront);

  const rollerRear = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.024, 0.13, 20), m.chromeShaft);
  rollerRear.rotation.z = Math.PI / 2;
  rollerRear.position.set(0, 0.145, -0.58);
  conveyorGroup.add(rollerRear);

  // Side Guide Railings
  const railL = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.028, 1.16), m.vermilionAccent);
  railL.position.set(-0.068, 0.18, 0);
  conveyorGroup.add(railL);

  const railR = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.028, 1.16), m.vermilionAccent);
  railR.position.set(0.068, 0.18, 0);
  conveyorGroup.add(railR);

  // Enclosed Infeed Tunnel Shroud at Upstream End (Where products naturally enter!)
  const infeedShroud = new THREE.Mesh(new THREE.BoxGeometry(0.146, 0.09, 0.14), m.castAluminiumLight);
  infeedShroud.position.set(0, 0.205, -0.51);
  conveyorGroup.add(infeedShroud);

  const infeedShroudRim = new THREE.Mesh(new THREE.BoxGeometry(0.148, 0.092, 0.012), m.vermilionAccent);
  infeedShroudRim.position.set(0, 0.205, -0.44);
  conveyorGroup.add(infeedShroudRim);

  const infeedTunnelMouth = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.06, 0.02), m.castIronDark);
  infeedTunnelMouth.position.set(0, 0.195, -0.44);
  conveyorGroup.add(infeedTunnelMouth);

  // Enclosed Outfeed Catch Shroud at Downstream End
  const outfeedCatch = new THREE.Mesh(new THREE.BoxGeometry(0.146, 0.07, 0.10), m.castAluminiumLight);
  outfeedCatch.position.set(0, 0.195, 0.53);
  conveyorGroup.add(outfeedCatch);

  // Moving Conveyor Belt Cleats (Moving continuously at constant velocity)
  const beltCleats: THREE.Mesh[] = [];
  for (let c = 0; c < 8; c++) {
    const cleat = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.002, 0.008), m.machinedBrass);
    cleat.position.y = 0.170;
    conveyorGroup.add(cleat);
    beltCleats.push(cleat);
  }

  // Optical Photo-Eye Laser Part Detection Gate at Z = -0.16
  const sensorPostL = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.06, 12), m.castIronDark);
  sensorPostL.position.set(-0.075, 0.19, -0.16);
  conveyorGroup.add(sensorPostL);

  const sensorPostR = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.06, 12), m.castIronDark);
  sensorPostR.position.set(0.075, 0.19, -0.16);
  conveyorGroup.add(sensorPostR);

  const laserBeam = new THREE.Mesh(new THREE.CylinderGeometry(0.0015, 0.0015, 0.15, 8), m.laserCyan);
  laserBeam.rotation.z = Math.PI / 2;
  laserBeam.position.set(0, 0.20, -0.16);
  conveyorGroup.add(laserBeam);

  // ── 5. Outfeed Packaging Station with Dual Alternating Trays (Destination at X = +0.22) ──
  const packagingGroup = new THREE.Group();
  packagingGroup.position.set(0.22, 0, 0);
  root.add(packagingGroup);

  // Aluminum Packaging Table & Gravity Roller Bed (Runs from Z = -0.40 to +0.55)
  const packTable = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.045, 0.95), m.castAluminiumLight);
  packTable.position.set(0, 0.142, 0.07);
  packTable.castShadow = true;
  packagingGroup.add(packTable);

  const packRailL = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.024, 0.95), m.castIronDark);
  packRailL.position.set(-0.086, 0.17, 0.07);
  packagingGroup.add(packRailL);

  const packRailR = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.024, 0.95), m.castIronDark);
  packRailR.position.set(0.086, 0.17, 0.07);
  packagingGroup.add(packRailR);

  // Downstream Outfeed Tunnel Hood for Packed Trays
  const trayTunnel = new THREE.Mesh(new THREE.BoxGeometry(0.186, 0.10, 0.14), m.castAluminiumLight);
  trayTunnel.position.set(0, 0.21, 0.48);
  packagingGroup.add(trayTunnel);

  const trayTunnelMouth = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.07, 0.02), m.castIronDark);
  trayTunnelMouth.position.set(0, 0.20, 0.41);
  packagingGroup.add(trayTunnelMouth);

  // 4 Pocket Offsets inside each 2x2 Blister Tray
  const pocketLocalOffsets = [
    new THREE.Vector3(-0.042, 0.022, -0.055),
    new THREE.Vector3(0.042, 0.022, -0.055),
    new THREE.Vector3(-0.042, 0.022, 0.055),
    new THREE.Vector3(0.042, 0.022, 0.055),
  ];

  const ledDarkMat = new THREE.MeshStandardMaterial({ color: 0x181a1b, roughness: 0.8 });

  // Helper to build a 4-pocket blister tray carton
  function createBlisterTray() {
    const carton = new THREE.Group();
    carton.position.set(0, 0.165, 0);

    const base = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.016, 0.22), m.castIronDark);
    base.position.y = 0.008;
    carton.add(base);

    const rim = new THREE.Mesh(new THREE.BoxGeometry(0.154, 0.004, 0.224), m.vermilionAccent);
    rim.position.y = 0.018;
    carton.add(rim);

    const leds: THREE.Mesh[] = [];
    for (let p = 0; p < 4; p++) {
      const pos = pocketLocalOffsets[p];

      // Recessed seating chamber
      const chamber = new THREE.Mesh(new THREE.CylinderGeometry(0.024, 0.022, 0.012, 20), m.castIronDark);
      chamber.position.set(pos.x, pos.y, pos.z);
      carton.add(chamber);

      // Gold beveled rim
      const bRim = new THREE.Mesh(new THREE.RingGeometry(0.020, 0.025, 20), m.machinedBrass);
      bRim.rotation.x = -Math.PI / 2;
      bRim.position.set(pos.x, pos.y + 0.007, pos.z);
      carton.add(bRim);

      // Status indicator LED
      const led = new THREE.Mesh(new THREE.CylinderGeometry(0.0035, 0.0035, 0.004, 12), ledDarkMat);
      led.position.set(pos.x + 0.026, pos.y + 0.007, pos.z);
      carton.add(led);
      leds.push(led);
    }

    packagingGroup.add(carton);
    return { carton, leds };
  }

  // Dual Alternating Trays (Tray A for Batch 1, Tray B for Batch 2)
  const trayA = createBlisterTray();
  const trayB = createBlisterTray();

  // ── 6. Eight Precision Canister Products in Continuous Flow ──
  const products: THREE.Group[] = [];
  const prodCapColors = [
    m.machinedBrass, m.vermilionAccent, m.laserCyan, m.ceramicShell,
    m.machinedBrass, m.vermilionAccent, m.laserCyan, m.ceramicShell
  ];

  for (let i = 0; i < 8; i++) {
    const prod = new THREE.Group();

    // Cylindrical Canister Body
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.017, 0.017, 0.020, 20), m.castIronDark);
    body.position.y = 0.010;
    body.castShadow = true;
    prod.add(body);

    // Colored Metallic Cap
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.004, 20), prodCapColors[i]);
    cap.position.y = 0.022;
    prod.add(cap);

    // Chrome Seal Ring
    const band = new THREE.Mesh(new THREE.CylinderGeometry(0.0175, 0.0175, 0.004, 20), m.chromeShaft);
    band.position.y = 0.012;
    prod.add(band);

    // Optical Inspection Center Dot
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.0025, 8, 8), m.laserCyan);
    dot.position.y = 0.0245;
    prod.add(dot);

    root.add(prod);
    products.push(prod);
  }

  // Trajectory Dashed Guidance Arc (High-G Parabolic Flight Path)
  const trajPts: THREE.Vector3[] = [];
  for (let s = 0; s <= 28; s++) {
    const f = s / 28;
    const px = (1 - f) * (-0.22) + f * 0.22;
    const py = 0.19 + Math.sin(f * Math.PI) * 0.22;
    const pz = (1 - f) * (-0.04) + f * 0.0;
    trajPts.push(new THREE.Vector3(px, py, pz));
  }
  const trajLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(trajPts),
    new THREE.LineDashedMaterial({ color: 0x2dd4bf, dashSize: 0.025, gapSize: 0.015, transparent: true, opacity: 0.45 })
  );
  trajLine.computeLineDistances();
  root.add(trajLine);

  scene.add(root);
  root.scale.setScalar(1.2);

  // ── 7. Analytical Closed-Form Delta Inverse Kinematics ──
  function solveDeltaArmIK(armIdx: number, px: number, py: number, pz: number): number {
    const psi = (armIdx / 3) * Math.PI * 2;
    const cosP = Math.cos(psi);
    const sinP = Math.sin(psi);

    // Rotate platform anchor into motor arm plane
    const xPrime = px * cosP + pz * sinP - R_EE;
    const zPrime = -px * sinP + pz * cosP;
    const yPrime = py;

    const X0 = xPrime - R_BASE;
    const Y0 = yPrime - Y_BASE;

    const dSq = L_ROD * L_ROD - zPrime * zPrime;
    if (dSq <= 0) return 0.85;

    const C = (X0 * X0 + Y0 * Y0 + L_BICEP * L_BICEP - dSq) / (2 * L_BICEP);
    const Rhyp = Math.sqrt(X0 * X0 + Y0 * Y0);
    if (Rhyp < 0.001) return 0.85;

    const ratio = Math.max(-0.9999, Math.min(0.9999, C / Rhyp));
    const delta = Math.acos(ratio);
    const phi = Math.atan2(-Y0, X0);

    // Elbow-down configuration
    return phi - delta;
  }

  const smooth = (a: number, b: number, v: number) => {
    const x = Math.max(0, Math.min(1, (v - a) / (b - a)));
    return x * x * (3 - 2 * x);
  };

  const worldElbowPos = new THREE.Vector3();
  const rodVec = new THREE.Vector3();
  const upAxis = new THREE.Vector3(0, 1, 0);

  // Belt constant parameters: continuous stream with zero jumps
  const V_BELT = 0.20; // Constant belt speed: 0.20 m/s
  const STEP_TIME = 1.05; // Time per pick cycle
  const TOTAL_CYCLE = STEP_TIME * 8; // 8.40s full cycle for 8 items (2 full 4-pack cartons)
  const Z_PICK_ZONE = -0.04; // Standard pick point on the moving belt

  return {
    root,
    update: (time: number, speedMultiplier: number) => {
      // 100% Continuous Flow Time Loop: 8.40s cycle
      const t = (time * 1.0 * speedMultiplier) % TOTAL_CYCLE;

      // Animate Moving Conveyor Belt Cleats continuously at constant velocity
      for (let c = 0; c < 8; c++) {
        const offset = c * (1.10 / 8);
        const cz = -0.55 + ((offset + time * V_BELT * speedMultiplier) % 1.10);
        beltCleats[c].position.z = cz;
      }

      const activeStep = Math.min(7, Math.floor(t / STEP_TIME));
      const stepProg = (t % STEP_TIME) / STEP_TIME;

      // Which tray carton is currently active?
      // Steps 0-3 fill Tray A (Batch 1), Steps 4-7 fill Tray B (Batch 2)
      const isBatchA = activeStep < 4;
      const activeTray = isBatchA ? trayA : trayB;
      const otherTray = isBatchA ? trayB : trayA;
      const activePocketIdx = activeStep % 4;

      // Calculate smooth tray advance & dispatch transitions
      // While Tray A is packing (t < 4.20s), Tray B is in queue / advancing
      if (t < 4.20) {
        // Tray A is stationed in packing bay
        trayA.carton.position.z = 0;
        // Tray B advances from upstream into position
        const pPrep = smooth(2.8, 4.2, t);
        trayB.carton.position.z = -0.38 * (1 - pPrep);
      } else {
        // Tray A dispatches downstream into outfeed tunnel
        const pDisp = smooth(4.2, 5.6, t);
        trayA.carton.position.z = pDisp * 0.46;
        // Tray B is stationed in packing bay
        trayB.carton.position.z = 0;
      }

      // Calculate Target Position for the Active Item in Destination Tray
      const activePocketOffset = pocketLocalOffsets[activePocketIdx];
      const activeTrayWorldZ = activeTray.carton.position.z;
      const targetPlaceX = 0.22 + activePocketOffset.x;
      const targetPlaceZ = activeTrayWorldZ + activePocketOffset.z;

      // Delta Robot Kinematic Trajectory
      let targetX = -0.22;
      let targetY = 0.36;
      let targetZ = Z_PICK_ZONE;
      let vacuumEngaged = false;
      let statusText = '';

      if (stepProg < 0.20) {
        // Phase 1: Rapid In-Flight Synchronization & Plunge onto Moving Belt Canister
        const p = smooth(0, 0.20, stepProg);
        targetX = -0.22;
        targetZ = Z_PICK_ZONE + (stepProg - 0.20) * V_BELT;
        targetY = 0.44 - p * 0.11; // Plunges down to 0.33m (contact with canister top)
        vacuumEngaged = stepProg > 0.12;
        statusText = `CONVEYOR IN-FLIGHT PICK [${activeStep + 1}/8] (-85 kPa)`;
      } else if (stepProg < 0.68) {
        // Phase 2: High-G Parabolic Transfer Arc across cell (15.2G peak accel)
        const p = smooth(0.20, 0.68, stepProg);
        targetX = (1 - p) * (-0.22) + p * targetPlaceX;
        targetZ = (1 - p) * Z_PICK_ZONE + p * targetPlaceZ;
        targetY = 0.33 + Math.sin(p * Math.PI) * 0.15; // High clearance arc up to 0.48m
        vacuumEngaged = true;
        statusText = `HIGH-G PARABOLIC TRANSIT (15.2G ACCEL)`;
      } else if (stepProg < 0.82) {
        // Phase 3: Plunge into Blister Pocket & Blow-Off Vacuum Release
        targetX = targetPlaceX;
        targetZ = targetPlaceZ;
        targetY = 0.33;
        vacuumEngaged = stepProg < 0.76;
        statusText = `SEATING IN BLISTER POCKET #${activePocketIdx + 1}`;
      } else {
        // Phase 4: High-Speed Snap Retract back to Conveyor Infeed
        const p = smooth(0.82, 1.00, stepProg);
        targetX = (1 - p) * targetPlaceX + p * (-0.22);
        targetZ = (1 - p) * targetPlaceZ + p * Z_PICK_ZONE;
        targetY = 0.33 + Math.sin(p * Math.PI) * 0.12;
        vacuumEngaged = false;
        statusText = `SNAP RETRACT · INTERCEPTING NEXT UNIT`;
      }

      // Update Tool Platform Position & Vacuum Glow Indicator
      toolPlatform.position.set(targetX, targetY, targetZ);
      vacuumGlow.visible = vacuumEngaged;

      // ── Solve Exact Closed-Form Delta Inverse Kinematics for All 3 Arms ──
      const bicepAngles: number[] = [];
      for (let i = 0; i < 3; i++) {
        const theta = solveDeltaArmIK(i, targetX, targetY, targetZ);
        bicepAngles.push(theta);
        bicepGroups[i].rotation.z = Math.PI / 2 - theta;

        // Obtain exact elbow position in root space
        elbowHolders[i].getWorldPosition(worldElbowPos);
        root.worldToLocal(worldElbowPos);

        // Platform dual ball-joint anchors (spaced 0.06m)
        const platformAngle = (i / 3) * Math.PI * 2;
        const eeAnchor1 = new THREE.Vector3(
          targetX + Math.cos(platformAngle + 0.16) * R_EE,
          targetY,
          targetZ + Math.sin(platformAngle + 0.16) * R_EE
        );
        const eeAnchor2 = new THREE.Vector3(
          targetX + Math.cos(platformAngle - 0.16) * R_EE,
          targetY,
          targetZ + Math.sin(platformAngle - 0.16) * R_EE
        );

        // Rod 1 of parallel pair: rigidly connects elbow ball to platform anchor
        const rod1 = parallelRods[i * 2];
        rodVec.subVectors(eeAnchor1, worldElbowPos);
        rod1.position.copy(worldElbowPos).addScaledVector(rodVec, 0.5);
        rod1.quaternion.setFromUnitVectors(upAxis, rodVec.clone().normalize());
        rod1.scale.set(1, L_ROD, 1);

        // Rod 2 of parallel pair
        const rod2 = parallelRods[i * 2 + 1];
        rodVec.subVectors(eeAnchor2, worldElbowPos);
        rod2.position.copy(worldElbowPos).addScaledVector(rodVec, 0.5);
        rod2.quaternion.setFromUnitVectors(upAxis, rodVec.clone().normalize());
        rod2.scale.set(1, L_ROD, 1);
      }

      // ── Continuous, Non-Spawning Conveyor Product Stream ──
      // For each product i, it travels at constant velocity V_BELT on the conveyor until picked!
      for (let i = 0; i < 8; i++) {
        const prod = products[i];
        const isItemBatchA = i < 4;
        const itemTray = isItemBatchA ? trayA : trayB;
        const itemPocketIdx = i % 4;
        const itemOffset = pocketLocalOffsets[itemPocketIdx];
        const itemGripTime = i * STEP_TIME + 0.20;

        // Calculate time delta until this item is gripped
        let timeUntilGrip = itemGripTime - t;
        if (timeUntilGrip < -0.05) {
          timeUntilGrip += TOTAL_CYCLE;
        }

        if (timeUntilGrip <= 0.01 && timeUntilGrip > -(STEP_TIME * 0.60)) {
          // Case 1: Currently Held in Flight by Vacuum Cup!
          prod.visible = true;
          prod.position.set(targetX, targetY - 0.14 - 0.012, targetZ);
        } else if (timeUntilGrip <= -(STEP_TIME * 0.60) || (i < activeStep && (activeStep - i) < 4)) {
          // Case 2: Successfully Placed into its Blister Tray Pocket!
          const trayZ = itemTray.carton.position.z;
          prod.visible = (trayZ + itemOffset.z) < 0.44; // Hides inside outfeed tunnel
          prod.position.set(0.22 + itemOffset.x, 0.187, trayZ + itemOffset.z);
          itemTray.leds[itemPocketIdx].material = m.laserCyan;
        } else {
          // Case 3: Continuously Traveling on the Conveyor Belt at Constant Velocity V_BELT!
          // Distance behind the pick point is exactly V_BELT * timeUntilGrip
          const conveyorZ = Z_PICK_ZONE - V_BELT * timeUntilGrip;

          // Products emerge from the infeed shroud (-0.50m) and are visible along the belt
          if (conveyorZ >= -0.54 && conveyorZ <= 0.55) {
            prod.visible = true;
            prod.position.set(-0.22, 0.177, conveyorZ);
          } else {
            // Inside upstream infeed hopper
            prod.visible = false;
            prod.position.set(-0.22, 0.177, -0.65);
          }

          // Unlit LED while empty
          if (i >= activeStep) {
            itemTray.leds[itemPocketIdx].material = ledDarkMat;
          }
        }
      }

      return {
        j1: (bicepAngles[0] * (180 / Math.PI)).toFixed(1),
        j2: (bicepAngles[1] * (180 / Math.PI)).toFixed(1),
        j3: `${(bicepAngles[2] * (180 / Math.PI)).toFixed(1)}° · 15.2G`,
        j4: `160 PICKS/MIN`,
        tcp: `BELT 0.20m/s · [${activeStep + 1}/8] UNIT`,
        status: statusText,
      };
    },
    dispose: () => {
      scene.remove(root);
    }
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Accurate Autonomous Mobile Robot (Differential AGV with 360° LiDAR)
// ─────────────────────────────────────────────────────────────────────────────
function buildAccurateMobileRover(scene: THREE.Scene, isWireframe = false) {
  const m = createStudioMaterials(isWireframe);
  const root = new THREE.Group();

  // 1. Lower Industrial Welded Chassis Base
  const chassisBase = new THREE.Mesh(new THREE.BoxGeometry(0.54, 0.12, 0.38), m.castIronDark);
  chassisBase.position.y = 0.12;
  chassisBase.castShadow = true;
  root.add(chassisBase);

  // Upper Molded Ceramic Cover Deck
  const coverDeck = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.08, 0.32), m.ceramicShell);
  coverDeck.position.y = 0.22;
  coverDeck.castShadow = true;
  root.add(coverDeck);

  // 360° Safety Perimeter LED Strip (Pulses amber/vermilion)
  const safetyBand = new THREE.Mesh(new THREE.BoxGeometry(0.548, 0.016, 0.388), m.vermilionAccent);
  safetyBand.position.y = 0.17;
  root.add(safetyBand);

  // 2. Drive Wheels (Two mid-mounted traction drive wheels)
  // Wheel meshes are created with cylinder oriented along X, so rotating around X rolls along Z heading!
  const wheelGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.05, 24);
  const wheelHubGeo = new THREE.CylinderGeometry(0.055, 0.055, 0.054, 24);

  // Left Drive Wheel
  const leftWheelGroup = new THREE.Group();
  leftWheelGroup.position.set(0, 0.09, 0.21);
  root.add(leftWheelGroup);

  const leftTire = new THREE.Mesh(wheelGeo, m.rubberTread);
  leftTire.rotation.x = Math.PI / 2;
  leftTire.castShadow = true;
  leftWheelGroup.add(leftTire);

  const leftHub = new THREE.Mesh(wheelHubGeo, m.vermilionAccent);
  leftHub.rotation.x = Math.PI / 2;
  leftWheelGroup.add(leftHub);

  // Right Drive Wheel
  const rightWheelGroup = new THREE.Group();
  rightWheelGroup.position.set(0, 0.09, -0.21);
  root.add(rightWheelGroup);

  const rightTire = new THREE.Mesh(wheelGeo, m.rubberTread);
  rightTire.rotation.x = Math.PI / 2;
  rightTire.castShadow = true;
  rightWheelGroup.add(rightTire);

  const rightHub = new THREE.Mesh(wheelHubGeo, m.vermilionAccent);
  rightHub.rotation.x = Math.PI / 2;
  rightWheelGroup.add(rightHub);

  // 4 Corner Omnidirectional Caster Housings
  const casterPositions = [
    [0.2, 0.045, 0.13],
    [0.2, 0.045, -0.13],
    [-0.2, 0.045, 0.13],
    [-0.2, 0.045, -0.13],
  ];
  casterPositions.forEach(([cx, cy, cz]) => {
    const casterBracket = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.04), m.castIronDark);
    casterBracket.position.set(cx, cy + 0.02, cz);
    root.add(casterBracket);

    const casterBall = new THREE.Mesh(new THREE.SphereGeometry(0.025, 14, 14), m.chromeShaft);
    casterBall.position.set(cx, cy, cz);
    root.add(casterBall);
  });

  // 3. Sensor Tower & 360° LiDAR Scanner
  const sensorPylon = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, 0.12, 20), m.castIronDark);
  sensorPylon.position.y = 0.32;
  root.add(sensorPylon);

  // Spinning LiDAR Head
  const lidarRotor = new THREE.Group();
  lidarRotor.position.y = 0.39;
  root.add(lidarRotor);

  const lidarPuck = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.05, 24), m.castAluminiumLight);
  lidarRotor.add(lidarPuck);

  const lidarOpticsRing = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.058, 0.015, 24), m.laserCyan);
  lidarOpticsRing.position.y = 0.01;
  lidarRotor.add(lidarOpticsRing);

  // 24 Active Planar LiDAR Laser Beams emitting 360°
  for (let r = 0; r < 24; r++) {
    const beamAngle = (r / 24) * Math.PI * 2;
    const beamDistance = 0.85 + (r % 4) * 0.3;
    const beamPts = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(Math.cos(beamAngle) * beamDistance, -0.12, Math.sin(beamAngle) * beamDistance)
    ];
    const beamGeo = new THREE.BufferGeometry().setFromPoints(beamPts);
    const beamLine = new THREE.Line(beamGeo, new THREE.LineBasicMaterial({
      color: 0x2dd4bf, transparent: true, opacity: 0.55
    }));
    lidarRotor.add(beamLine);
  }

  // Front Stereo RGB-D Cameras (Intel RealSense form-factor)
  const cameraBar = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.025, 0.14), m.castIronDark);
  cameraBar.position.set(0.25, 0.23, 0);
  root.add(cameraBar);

  const lensLeft = new THREE.Mesh(new THREE.SphereGeometry(0.008, 12, 12), m.chromeShaft);
  lensLeft.position.set(0.272, 0.23, 0.045);
  root.add(lensLeft);

  const lensRight = new THREE.Mesh(new THREE.SphereGeometry(0.008, 12, 12), m.chromeShaft);
  lensRight.position.set(0.272, 0.23, -0.045);
  root.add(lensRight);

  // Dynamic Obstacle Pylons on Ground
  const obsA = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.35, 24), m.castIronDark);
  obsA.position.set(0.9, 0.175, 0.45);
  root.add(obsA);

  const obsB = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.24, 0.24), m.vermilionAccent);
  obsB.position.set(-0.85, 0.12, -0.5);
  root.add(obsB);

  scene.add(root);
  root.scale.setScalar(1.3);

  return {
    root,
    update: (time: number, speedMultiplier: number) => {
      const t = time * 0.9 * speedMultiplier;

      // Realistic smooth path trajectory
      root.position.x = Math.sin(t * 0.8) * 0.4;
      root.position.z = Math.cos(t * 0.6) * 0.35;
      root.rotation.y = Math.sin(t * 0.8) * 0.6;

      // Physically rolling drive wheels
      const wheelRoll = t * 4.0;
      leftTire.rotation.x = wheelRoll;
      leftHub.rotation.x = wheelRoll;
      rightTire.rotation.x = wheelRoll;
      rightHub.rotation.x = wheelRoll;

      // Continuous 360° LiDAR rotational sweep
      lidarRotor.rotation.y += 0.14 * speedMultiplier;

      return {
        j1: `X: ${root.position.x.toFixed(2)} m`,
        j2: `Z: ${root.position.z.toFixed(2)} m`,
        j3: `HEADING: ${(root.rotation.y * (180 / Math.PI)).toFixed(1)}°`,
        j4: `VEL: 1.84 m/s`,
        tcp: `24 BEAMS · OBSTACLES: CLEAR`,
        status: 'ODOMETRY STREAMING (60Hz)',
      };
    },
    dispose: () => {
      scene.remove(root);
    }
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive Robot 3D Canvas with OrbitControls
// ─────────────────────────────────────────────────────────────────────────────
function InteractiveRobotStage({
  activeRobotId,
  isWireframe,
  speedMultiplier,
  isPaused,
  onTelemetryUpdate
}: {
  activeRobotId: string;
  isWireframe: boolean;
  speedMultiplier: number;
  isPaused: boolean;
  onTelemetryUpdate: (data: any) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<any>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const isPausedRef = useRef(isPaused);
  isPausedRef.current = isPaused;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 740;
    const height = container.clientHeight || 580;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.05, 100);
    camera.position.set(1.25, 1.15, 2.55);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Smooth OrbitControls with Damping
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0.42, 0);
    controls.minDistance = 1.0;
    controls.maxDistance = 4.5;
    controls.maxPolarAngle = Math.PI / 2 + 0.05;
    controlsRef.current = controls;

    // High-End Studio Three-Point Lighting
    const ambient = new THREE.AmbientLight(0xfff7ee, 0.65);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xfffaee, 3.4);
    keyLight.position.set(3, 5, 4);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.setScalar(2048);
    keyLight.shadow.bias = -0.0001;
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0xdbe6ff, 1.8);
    rimLight.position.set(-4, 3, -3);
    scene.add(rimLight);

    const floorBounce = new THREE.PointLight(0xc4472d, 1.4, 4);
    floorBounce.position.set(0, 0.05, 0.5);
    scene.add(floorBounce);

    // Ground Plane with Grid
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(8, 8),
      new THREE.MeshStandardMaterial({ color: 0xede9e0, roughness: 0.75, metalness: 0.08 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    const grid = new THREE.GridHelper(6, 24, 0xb8b2a3, 0xd0c9bb);
    grid.position.y = 0.001;
    scene.add(grid);

    // Animation Loop
    let animId = 0;
    let clock = new THREE.Clock();

    const renderLoop = () => {
      animId = requestAnimationFrame(renderLoop);
      controls.update();

      if (!isPausedRef.current && controllerRef.current) {
        const elapsed = clock.getElapsedTime();
        const telemetry = controllerRef.current.update(elapsed, speedMultiplier);
        if (onTelemetryUpdate) onTelemetryUpdate(telemetry);
      }

      renderer.render(scene, camera);
    };
    renderLoop();

    const handleResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Rebuild the active robot when type or wireframe toggles
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (controllerRef.current) {
      controllerRef.current.dispose();
      controllerRef.current = null;
    }

    if (activeRobotId === 'kuka') {
      controllerRef.current = buildAccurateKUKAArm(scene, isWireframe);
    } else if (activeRobotId === 'scara') {
      controllerRef.current = buildAccurateSCARARobot(scene, isWireframe);
    } else if (activeRobotId === 'delta') {
      controllerRef.current = buildAccurateDeltaRobot(scene, isWireframe);
    } else if (activeRobotId === 'mobile') {
      controllerRef.current = buildAccurateMobileRover(scene, isWireframe);
    }
  }, [activeRobotId, isWireframe]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        cursor: 'grab',
        touchAction: 'none'
      }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Bubble Pill Nav Bar (ReactBits / Gooey Pill Animation)
// ─────────────────────────────────────────────────────────────────────────────
function BubbleNavBar({
  onOpenDocs,
  onNavigateSection
}: {
  onOpenDocs: () => void;
  onNavigateSection: (sectionId: string) => void;
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [bubbleStyle, setBubbleStyle] = useState({ left: 0, width: 0, opacity: 0 });
  const itemRefs = useRef<(HTMLDivElement | null)[]>([]);

  const navItems = [
    { label: 'Platform', action: () => onNavigateSection('platform') },
    { label: '3D Exploded', action: () => onNavigateSection('exploded') },
    { label: 'Robots', action: () => onNavigateSection('robots') },
    { label: 'Research', action: () => onNavigateSection('research') },
    { label: 'Documentation', action: onOpenDocs },
  ];

  const handleMouseEnter = (index: number) => {
    setHoveredIdx(index);
    const el = itemRefs.current[index];
    if (el) {
      setBubbleStyle({
        left: el.offsetLeft,
        width: el.offsetWidth,
        opacity: 1,
      });
    }
  };

  const handleMouseLeave = () => {
    setHoveredIdx(null);
    setBubbleStyle(prev => ({ ...prev, opacity: 0 }));
  };

  return (
    <header style={{
      position: 'fixed',
      top: '18px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 200,
      width: 'min(94%, 1100px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '8px 14px 8px 22px',
      background: 'rgba(251, 249, 245, 0.88)',
      backdropFilter: 'blur(20px) saturate(180%)',
      WebkitBackdropFilter: 'blur(20px) saturate(180%)',
      border: '1px solid rgba(255, 255, 255, 0.85)',
      boxShadow: '0 12px 36px -10px rgba(32, 34, 34, 0.12), 0 2px 8px rgba(0,0,0,0.04)',
      borderRadius: '100px',
    }}>
      {/* Brand logo */}
      <Link to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div style={{
          width: '9px',
          height: '9px',
          borderRadius: '50%',
          background: C.red,
          boxShadow: `0 0 12px ${C.redGlow}`,
          animation: 'pulseBead 2.4s cubic-bezier(0.4, 0, 0.6, 1) infinite'
        }} />
        <span style={{
          fontFamily: 'IBM Plex Mono, monospace',
          fontSize: '12px',
          letterSpacing: '0.22em',
          fontWeight: 700,
          color: C.black,
        }}>
          ROBOFORGE
        </span>
      </Link>

      {/* Center Liquid Bubble Navigation Container */}
      <nav
        onMouseLeave={handleMouseLeave}
        style={{
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          background: 'rgba(32, 34, 34, 0.04)',
          padding: '4px',
          borderRadius: '100px',
        }}
      >
        {/* Sliding Fluid Pill Background (ReactBits / CSS Bubble pill) */}
        <div style={{
          position: 'absolute',
          top: '4px',
          bottom: '4px',
          left: `${bubbleStyle.left}px`,
          width: `${bubbleStyle.width}px`,
          opacity: bubbleStyle.opacity,
          background: 'white',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0,0,0,0.05)',
          borderRadius: '100px',
          pointerEvents: 'none',
          transition: 'all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1)',
        }} />

        {navItems.map((item, idx) => (
          <div
            key={item.label}
            ref={el => { itemRefs.current[idx] = el; }}
            onMouseEnter={() => handleMouseEnter(idx)}
            onClick={item.action}
            style={{
              position: 'relative',
              zIndex: 2,
              padding: '7px 18px',
              fontFamily: 'Manrope, sans-serif',
              fontSize: '13px',
              fontWeight: 600,
              color: hoveredIdx === idx ? C.red : C.black,
              opacity: hoveredIdx === idx ? 1 : 0.75,
              cursor: 'pointer',
              borderRadius: '100px',
              transition: 'color 0.2s, transform 0.2s',
              transform: hoveredIdx === idx ? 'scale(1.04)' : 'scale(1)',
              userSelect: 'none',
            }}
          >
            {item.label}
          </div>
        ))}
      </nav>

      {/* Action Group */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <Link to="/login" style={{ textDecoration: 'none' }}>
          <div style={{
            fontFamily: 'Manrope, sans-serif',
            fontSize: '13px',
            fontWeight: 600,
            color: C.black,
            opacity: 0.75,
            padding: '8px 14px',
            borderRadius: '100px',
          }}>
            Log in
          </div>
        </Link>

        <Link to="/studio" style={{ textDecoration: 'none' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: C.red,
              color: 'white',
              fontFamily: 'Manrope, sans-serif',
              fontSize: '13px',
              fontWeight: 700,
              letterSpacing: '0.02em',
              padding: '9px 20px',
              borderRadius: '100px',
              boxShadow: `0 6px 20px -2px ${C.redGlow}`,
              transition: 'all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1)',
              cursor: 'pointer',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.transform = 'scale(1.05) translateY(-1px)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.transform = 'scale(1) translateY(0)';
            }}
          >
            <span>Open Studio</span>
            <ArrowRight size={14} />
          </div>
        </Link>
      </div>
    </header>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Documentation Reference Modal
// ─────────────────────────────────────────────────────────────────────────────
function DocumentationModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<'quickstart' | 'urdf' | 'telemetry' | 'ros2'>('quickstart');
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const copyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 500,
      background: 'rgba(23, 25, 26, 0.75)',
      backdropFilter: 'blur(12px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '24px',
    }}>
      <div style={{
        background: '#1A1C1E',
        color: C.ivory,
        width: 'min(92%, 840px)',
        maxHeight: '85vh',
        borderRadius: '18px',
        border: '1px solid rgba(255, 255, 255, 0.12)',
        boxShadow: '0 32px 80px rgba(0,0,0,0.6)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          padding: '20px 28px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '32px', height: '32px', borderRadius: '8px',
              background: 'rgba(196,71,45,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.red,
            }}>
              <BookOpen size={18} />
            </div>
            <div>
              <div style={{ fontFamily: 'Manrope, sans-serif', fontSize: '16px', fontWeight: 700 }}>
                RoboForge Engineering Documentation
              </div>
              <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px', color: 'rgba(255,255,255,0.4)' }}>
                MuJoCo 3.17 · FastAPI · ROS 2 · URDF / MJCF Spec
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'rgba(255,255,255,0.06)', border: 'none', borderRadius: '50%',
              width: '32px', height: '32px', display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: 'rgba(255,255,255,0.6)', cursor: 'pointer',
            }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Tab switcher */}
        <div style={{
          display: 'flex', padding: '12px 28px 0', gap: '8px', borderBottom: '1px solid rgba(255,255,255,0.06)',
        }}>
          {[
            { id: 'quickstart', label: '1. Quickstart & CLI' },
            { id: 'urdf', label: '2. URDF / MJCF Compiler' },
            { id: 'telemetry', label: '3. 60Hz Telemetry Socket' },
            { id: 'ros2', label: '4. ROS 2 Bridge Topics' },
          ].map(tab => (
            <div
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              style={{
                padding: '8px 16px',
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '12px',
                fontWeight: 600,
                color: activeTab === tab.id ? C.red : 'rgba(255,255,255,0.5)',
                borderBottom: `2px solid ${activeTab === tab.id ? C.red : 'transparent'}`,
                cursor: 'pointer',
                marginBottom: '-1px',
              }}
            >
              {tab.label}
            </div>
          ))}
        </div>

        {/* Body content */}
        <div style={{ padding: '24px 28px', overflowY: 'auto', flex: 1 }}>
          {activeTab === 'quickstart' && (
            <div>
              <p style={{ fontSize: '14px', lineHeight: 1.6, opacity: 0.8, marginBottom: '16px' }}>
                RoboForge runs an authoritative MuJoCo physics simulation in Python 3.11 with 60–200 Hz WebSocket state broadcast.
              </p>
              <div style={{
                background: '#111213', padding: '16px', borderRadius: '8px',
                fontFamily: 'IBM Plex Mono, monospace', fontSize: '12px', lineHeight: 1.7, position: 'relative',
              }}>
                <button
                  onClick={() => copyCode('.venv\\Scripts\\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000')}
                  style={{
                    position: 'absolute', right: '12px', top: '12px',
                    background: 'rgba(255,255,255,0.08)', border: 'none',
                    borderRadius: '4px', padding: '4px 8px', color: 'white',
                    display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '11px',
                  }}
                >
                  {copied ? <CheckCircle2 size={12} color={C.cyan} /> : <Copy size={12} />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
                <span style={{ color: 'rgba(255,255,255,0.3)' }}># Start FastAPI Physics Server</span><br />
                <span style={{ color: C.cyan }}>.venv\Scripts\python</span> -m uvicorn backend.app:app --host 127.0.0.1 --port 8000<br /><br />
                <span style={{ color: 'rgba(255,255,255,0.3)' }}># Start Vite Interactive Studio</span><br />
                <span style={{ color: C.cyan }}>cd</span> frontend && <span style={{ color: C.cyan }}>npm</span> run dev
              </div>
            </div>
          )}

          {activeTab === 'urdf' && (
            <div>
              <p style={{ fontSize: '14px', lineHeight: 1.6, opacity: 0.8, marginBottom: '16px' }}>
                Import standard ROS URDF XML definitions or native MuJoCo MJCF models with automatic inertial tensor compilation:
              </p>
              <div style={{
                background: '#111213', padding: '16px', borderRadius: '8px',
                fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px', color: '#c8d8f0', lineHeight: 1.6,
              }}>
                {`POST /api/urdf/import
Content-Type: application/json
{
  "urdf_content": "<robot name='arm_6dof'>\\n  <link name='base'/>\\n  <joint type='revolute'/>\\n</robot>"
}

Response 200 OK:
{
  "success": true,
  "links_count": 7,
  "joints_count": 6,
  "mjcf_compiled": true
}`}
              </div>
            </div>
          )}

          {activeTab === 'telemetry' && (
            <div>
              <p style={{ fontSize: '14px', lineHeight: 1.6, opacity: 0.8, marginBottom: '16px' }}>
                Connect to the live 60 Hz bidirectional WebSocket hub to stream joint states, contact forces (`mj_contactForce`), and telemetry:
              </p>
              <div style={{
                background: '#111213', padding: '16px', borderRadius: '8px',
                fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px', color: C.cyan, lineHeight: 1.6,
              }}>
                {`ws://127.0.0.1:8000/ws

Stream Packet:
{
  "sim_time": 4.250,
  "real_time_factor": 1.002,
  "joints": { "j1": 0.42, "j2": -0.18, "j3": 0.71 },
  "contacts": [{ "geom1": "gripper_tip", "geom2": "target_cube", "force": 3.84 }],
  "imu": { "accel": [0.0, 0.0, 9.81], "gyro": [0.01, -0.02, 0.0] }
}`}
              </div>
            </div>
          )}

          {activeTab === 'ros2' && (
            <div>
              <p style={{ fontSize: '14px', lineHeight: 1.6, opacity: 0.8, marginBottom: '16px' }}>
                Native ROS 2 bridge topics serializing standard `geometry_msgs`, `sensor_msgs`, and `nav_msgs`:
              </p>
              <ul style={{ paddingLeft: '20px', fontSize: '13px', lineHeight: 1.8, opacity: 0.8 }}>
                <li><code>/cmd_vel</code> — Geometry Twist command teleoperation</li>
                <li><code>/odom</code> — High-frequency wheel odometry and orientation quaternion</li>
                <li><code>/scan</code> — 360° 32-beam LaserScan point range distances</li>
                <li><code>/imu/data</code> — Linear accelerations and angular velocities</li>
                <li><code>/joint_states</code> — Joint positions, velocities, and applied torques</li>
              </ul>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '16px 28px', borderTop: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#141618',
        }}>
          <span style={{ fontSize: '12px', opacity: 0.5, fontFamily: 'IBM Plex Mono, monospace' }}>
            FastAPI Swagger: http://127.0.0.1:8000/docs
          </span>
          <Link to="/studio" style={{ textDecoration: 'none' }}>
            <div style={{
              background: C.red, color: 'white', padding: '8px 18px',
              borderRadius: '6px', fontFamily: 'Manrope, sans-serif', fontWeight: 700, fontSize: '12px',
            }}>
              Launch Studio Workspace ↗
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Page Layout Component
// ─────────────────────────────────────────────────────────────────────────────
export function LandingPage() {
  const [activeRobotId, setActiveRobotId] = useState('kuka');
  const [isWireframe, setIsWireframe] = useState(false);
  const [speedMultiplier, setSpeedMultiplier] = useState(1.0);
  const [isPaused, setIsPaused] = useState(false);
  const [telemetry, setTelemetry] = useState<any>({
    j1: '0.0', j2: '0.0', j3: '0.0', j4: '0.0', tcp: '0.42, 0.35, 0.12', status: 'SYNCHRONIZED'
  });
  const [isDocsOpen, setIsDocsOpen] = useState(false);

  const activeRobot = ROBOT_CATALOG.find(r => r.id === activeRobotId) || ROBOT_CATALOG[0];

  const handleNavigateSection = (sectionId: string) => {
    const el = document.getElementById(sectionId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div style={{
      fontFamily: 'Manrope, sans-serif',
      color: C.black,
      background: C.ivory,
      minHeight: '100vh',
      position: 'relative',
    }}>
      {/* ── Floating Bubble Navigation ── */}
      <BubbleNavBar
        onOpenDocs={() => setIsDocsOpen(true)}
        onNavigateSection={handleNavigateSection}
      />

      {/* ── Documentation Modal ── */}
      <DocumentationModal
        isOpen={isDocsOpen}
        onClose={() => setIsDocsOpen(false)}
      />

      {/* ── HERO SECTION: Industrial Product Launch ── */}
      <section style={{
        minHeight: '100vh',
        paddingTop: '96px',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        background: `radial-gradient(ellipse at 70% 40%, rgba(255,255,255,0.92) 0%, ${C.ivory} 70%)`,
        overflow: 'hidden',
      }}>
        {/* Subtle background industrial grid */}
        <div style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `linear-gradient(to right, rgba(32,34,34,0.04) 1px, transparent 1px), linear-gradient(to bottom, rgba(32,34,34,0.04) 1px, transparent 1px)`,
          backgroundSize: '48px 48px',
          pointerEvents: 'none',
        }} />

        <div style={{
          width: 'min(94%, 1320px)',
          margin: '0 auto',
          display: 'grid',
          gridTemplateColumns: '1.05fr 1.35fr',
          gap: '40px',
          alignItems: 'center',
          position: 'relative',
          zIndex: 10,
          padding: '24px 0 60px',
        }}>
          {/* Left Column: Product Narrative & Typography */}
          <div>
            {/* Pill Tag */}
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 14px',
              borderRadius: '100px',
              background: 'rgba(196, 71, 45, 0.08)',
              border: '1px solid rgba(196, 71, 45, 0.22)',
              marginBottom: '24px',
            }}>
              <span style={{
                width: '6px', height: '6px', borderRadius: '50%', background: C.red,
                boxShadow: `0 0 8px ${C.red}`,
              }} />
              <span style={{
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '11px',
                fontWeight: 600,
                color: C.red,
                letterSpacing: '0.12em',
              }}>
                MUJOCO 3.17 PHYSICS CORE
              </span>
            </div>

            <h1 style={{
              fontSize: 'clamp(42px, 5.2vw, 76px)',
              fontWeight: 800,
              lineHeight: 1.05,
              letterSpacing: '-0.035em',
              color: C.black,
              margin: '0 0 20px',
            }}>
              From mechanism<br />
              <span style={{
                background: `linear-gradient(135deg, ${C.red} 0%, #E25A38 100%)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>
                to motion.
              </span>
            </h1>

            <p style={{
              fontSize: '17px',
              lineHeight: 1.65,
              color: 'rgba(32, 34, 34, 0.72)',
              margin: '0 0 32px',
              maxWidth: '480px',
            }}>
              A browser-based robotics engineering studio. Model articulated kinematic chains, compile native URDF & MJCF, run deterministic collision dynamics, and train autonomous policies.
            </p>

            {/* CTAs */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap', marginBottom: '44px' }}>
              <Link to="/studio" style={{ textDecoration: 'none' }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  background: C.black,
                  color: C.ivory,
                  padding: '16px 32px',
                  borderRadius: '10px',
                  fontFamily: 'Manrope, sans-serif',
                  fontWeight: 700,
                  fontSize: '14px',
                  boxShadow: '0 10px 24px -4px rgba(32,34,34,0.3)',
                  transition: 'all 0.25s',
                  cursor: 'pointer',
                }}>
                  <span>Open Engineering Studio</span>
                  <ArrowRight size={16} />
                </div>
              </Link>

              <button
                onClick={() => setIsDocsOpen(true)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  background: 'white',
                  color: C.black,
                  border: '1px solid rgba(32, 34, 34, 0.15)',
                  padding: '15px 26px',
                  borderRadius: '10px',
                  fontFamily: 'Manrope, sans-serif',
                  fontWeight: 600,
                  fontSize: '14px',
                  cursor: 'pointer',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.04)',
                }}
              >
                <Code2 size={16} />
                <span>API & URDF Docs</span>
              </button>
            </div>

            {/* Interactive Robot Picker Pills */}
            <div>
              <div style={{
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '10px',
                letterSpacing: '0.18em',
                color: C.muted,
                marginBottom: '10px',
                textTransform: 'uppercase',
              }}>
                Select Active Mechanism:
              </div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {ROBOT_CATALOG.map(robot => (
                  <div
                    key={robot.id}
                    onClick={() => setActiveRobotId(robot.id)}
                    style={{
                      padding: '8px 16px',
                      borderRadius: '8px',
                      border: `1.5px solid ${activeRobotId === robot.id ? C.red : 'rgba(32,34,34,0.12)'}`,
                      background: activeRobotId === robot.id ? 'white' : 'rgba(255,255,255,0.4)',
                      boxShadow: activeRobotId === robot.id ? '0 4px 14px rgba(196,71,45,0.15)' : 'none',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                  >
                    <div style={{
                      fontFamily: 'Manrope, sans-serif',
                      fontSize: '12px',
                      fontWeight: 700,
                      color: activeRobotId === robot.id ? C.red : C.black,
                    }}>
                      {robot.name}
                    </div>
                    <div style={{
                      fontFamily: 'IBM Plex Mono, monospace',
                      fontSize: '10px',
                      color: 'rgba(32,34,34,0.5)',
                    }}>
                      {robot.dof}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: High-Fidelity 3D Robot Stage */}
          <div style={{
            position: 'relative',
            height: '620px',
            background: 'linear-gradient(145deg, #FCFBF9 0%, #EFEBE1 100%)',
            borderRadius: '24px',
            border: '1px solid rgba(255,255,255,0.9)',
            boxShadow: '0 24px 60px -15px rgba(32, 34, 34, 0.12), inset 0 1px 0 rgba(255,255,255,0.8)',
            overflow: 'hidden',
          }}>
            {/* 3D Canvas with OrbitControls */}
            <InteractiveRobotStage
              activeRobotId={activeRobotId}
              isWireframe={isWireframe}
              speedMultiplier={speedMultiplier}
              isPaused={isPaused}
              onTelemetryUpdate={setTelemetry}
            />

            {/* Top-Right Model Spec Overlay */}
            <div style={{
              position: 'absolute',
              top: '20px',
              right: '20px',
              background: 'rgba(23, 25, 26, 0.88)',
              color: 'white',
              backdropFilter: 'blur(16px)',
              borderRadius: '12px',
              padding: '14px 18px',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              minWidth: '220px',
              boxShadow: '0 12px 28px rgba(0,0,0,0.25)',
              pointerEvents: 'none',
            }}>
              <div style={{
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '10px',
                color: C.red,
                letterSpacing: '0.16em',
                marginBottom: '4px',
              }}>
                MODEL SPECIFICATION
              </div>
              <div style={{ fontSize: '15px', fontWeight: 800, marginBottom: '2px' }}>
                {activeRobot.name}
              </div>
              <div style={{ fontSize: '11px', opacity: 0.65, marginBottom: '10px' }}>
                {activeRobot.category}
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '8px',
                paddingTop: '8px',
                borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '10px',
              }}>
                <div>
                  <span style={{ opacity: 0.45 }}>PAYLOAD</span><br />
                  <span style={{ fontWeight: 600 }}>{activeRobot.payload}</span>
                </div>
                <div>
                  <span style={{ opacity: 0.45 }}>CYCLE</span><br />
                  <span style={{ fontWeight: 600 }}>{activeRobot.cycleTime}</span>
                </div>
              </div>
            </div>

            {/* Drag to Orbit Hint Badge (Top-Left) */}
            <div style={{
              position: 'absolute',
              top: '20px',
              left: '20px',
              background: 'rgba(255, 255, 255, 0.88)',
              backdropFilter: 'blur(12px)',
              borderRadius: '100px',
              padding: '6px 14px',
              border: '1px solid rgba(32,34,34,0.08)',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '11px',
              fontWeight: 600,
              color: 'rgba(32,34,34,0.65)',
              pointerEvents: 'none',
            }}>
              <Orbit size={13} color={C.red} />
              <span>Drag to orbit / Scroll to zoom</span>
            </div>

            {/* Floating Live Telemetry HUD Bar (Bottom-Left) */}
            <div style={{
              position: 'absolute',
              bottom: '20px',
              left: '20px',
              background: 'rgba(255, 255, 255, 0.92)',
              backdropFilter: 'blur(16px)',
              borderRadius: '12px',
              padding: '12px 18px',
              border: '1px solid rgba(32, 34, 34, 0.08)',
              boxShadow: '0 12px 28px rgba(32, 34, 34, 0.08)',
              display: 'flex',
              alignItems: 'center',
              gap: '18px',
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '11px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{
                  width: '7px', height: '7px', borderRadius: '50%', background: C.red,
                  animation: 'pulseBead 1.5s infinite'
                }} />
                <span style={{ fontWeight: 700, color: C.black }}>{telemetry.status}</span>
              </div>
              <div style={{ height: '18px', width: '1px', background: 'rgba(32,34,34,0.1)' }} />
              <div>
                <span style={{ opacity: 0.45 }}>J1:</span> <b>{telemetry.j1}°</b>
              </div>
              <div>
                <span style={{ opacity: 0.45 }}>J2:</span> <b>{telemetry.j2}°</b>
              </div>
              <div>
                <span style={{ opacity: 0.45 }}>J3:</span> <b>{telemetry.j3}</b>
              </div>
              <div style={{ display: 'block' }}>
                <span style={{ opacity: 0.45 }}>TCP:</span> <b>{telemetry.tcp}</b>
              </div>
            </div>

            {/* Viewport Control Buttons (Bottom-Right) */}
            <div style={{
              position: 'absolute',
              bottom: '20px',
              right: '20px',
              display: 'flex',
              gap: '8px',
            }}>
              <button
                onClick={() => setIsPaused(!isPaused)}
                style={{
                  background: isPaused ? C.red : 'rgba(255, 255, 255, 0.92)',
                  color: isPaused ? 'white' : C.black,
                  border: '1px solid rgba(32,34,34,0.12)',
                  borderRadius: '8px',
                  padding: '8px 12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
                }}
              >
                {isPaused ? <Play size={13} /> : <Pause size={13} />}
                <span>{isPaused ? 'Resume' : 'Pause'}</span>
              </button>

              <button
                onClick={() => setIsWireframe(!isWireframe)}
                style={{
                  background: isWireframe ? C.black : 'rgba(255, 255, 255, 0.92)',
                  color: isWireframe ? 'white' : C.black,
                  border: '1px solid rgba(32,34,34,0.12)',
                  borderRadius: '8px',
                  padding: '8px 12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
                }}
              >
                <Eye size={13} />
                <span>{isWireframe ? 'Solid' : 'Wireframe'}</span>
              </button>

              <button
                onClick={() => setSpeedMultiplier(prev => (prev === 1.0 ? 2.0 : prev === 2.0 ? 0.5 : 1.0))}
                style={{
                  background: 'rgba(255, 255, 255, 0.92)',
                  color: C.black,
                  border: '1px solid rgba(32,34,34,0.12)',
                  borderRadius: '8px',
                  padding: '8px 12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
                }}
              >
                <Zap size={13} color={C.red} />
                <span>{speedMultiplier}x Speed</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ── PLATFORM CAPABILITIES SECTION ── */}
      <section id="platform" style={{
        padding: '120px 0',
        background: 'white',
        borderTop: '1px solid rgba(32, 34, 34, 0.08)',
        borderBottom: '1px solid rgba(32, 34, 34, 0.08)',
      }}>
        <div style={{ width: 'min(94%, 1280px)', margin: '0 auto' }}>
          <div style={{ maxWidth: '640px', marginBottom: '64px' }}>
            <div style={{
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '11px',
              fontWeight: 600,
              color: C.red,
              letterSpacing: '0.18em',
              marginBottom: '12px',
              textTransform: 'uppercase',
            }}>
              Engine Architecture
            </div>
            <h2 style={{
              fontSize: 'clamp(32px, 3.8vw, 52px)',
              fontWeight: 800,
              lineHeight: 1.1,
              letterSpacing: '-0.03em',
              margin: '0 0 16px',
            }}>
              Every layer built for physical reality.
            </h2>
            <p style={{ fontSize: '16px', color: 'rgba(32,34,34,0.7)', lineHeight: 1.6 }}>
              RoboForge unifies forward & inverse kinematics, dynamic contact formulations, and ROS 2 communication into a high-performance workspace.
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
            gap: '24px',
          }}>
            {/* Card 1 */}
            <div style={{
              padding: '36px',
              borderRadius: '16px',
              background: C.ivory,
              border: '1px solid rgba(32, 34, 34, 0.08)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              transition: 'transform 0.25s, box-shadow 0.25s',
            }}>
              <div>
                <div style={{
                  width: '44px',
                  height: '44px',
                  borderRadius: '10px',
                  background: 'white',
                  border: '1px solid rgba(32,34,34,0.08)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: C.red,
                  marginBottom: '24px',
                }}>
                  <Cpu size={22} />
                </div>
                <h3 style={{ fontSize: '22px', fontWeight: 700, margin: '0 0 12px' }}>
                  Kinematics & Solver Core
                </h3>
                <p style={{ fontSize: '14px', lineHeight: 1.6, opacity: 0.75, margin: '0 0 24px' }}>
                  Damped Least-Squares Jacobian inverse kinematics with automatic joint limit avoidance, manipulability ellipses, and singularity avoidance.
                </p>
              </div>
              <div style={{
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '11px',
                padding: '12px',
                borderRadius: '8px',
                background: 'rgba(32,34,34,0.04)',
                border: '1px solid rgba(32,34,34,0.08)',
              }}>
                <code>J_dls = J^T (J J^T + λ^2 I)^(-1)</code>
              </div>
            </div>

            {/* Card 2 */}
            <div style={{
              padding: '36px',
              borderRadius: '16px',
              background: C.ivory,
              border: '1px solid rgba(32, 34, 34, 0.08)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
            }}>
              <div>
                <div style={{
                  width: '44px',
                  height: '44px',
                  borderRadius: '10px',
                  background: 'white',
                  border: '1px solid rgba(32,34,34,0.08)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: C.red,
                  marginBottom: '24px',
                }}>
                  <Activity size={22} />
                </div>
                <h3 style={{ fontSize: '22px', fontWeight: 700, margin: '0 0 12px' }}>
                  MuJoCo 3.17 Physics
                </h3>
                <p style={{ fontSize: '14px', lineHeight: 1.6, opacity: 0.75, margin: '0 0 24px' }}>
                  Authoritative continuous stepping with implicit-fast integration, 3D frictional contact ellipsoids, restitution, and non-penetration constraints.
                </p>
              </div>
              <div style={{
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '11px',
                padding: '12px',
                borderRadius: '8px',
                background: 'rgba(32,34,34,0.04)',
                border: '1px solid rgba(32,34,34,0.08)',
              }}>
                <code>F_contact = mj_contactForce(m, d, id)</code>
              </div>
            </div>

            {/* Card 3 */}
            <div style={{
              padding: '36px',
              borderRadius: '16px',
              background: C.ivory,
              border: '1px solid rgba(32, 34, 34, 0.08)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
            }}>
              <div>
                <div style={{
                  width: '44px',
                  height: '44px',
                  borderRadius: '10px',
                  background: 'white',
                  border: '1px solid rgba(32,34,34,0.08)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: C.red,
                  marginBottom: '24px',
                }}>
                  <Layers size={22} />
                </div>
                <h3 style={{ fontSize: '22px', fontWeight: 700, margin: '0 0 12px' }}>
                  ROS 2 Bag & Bridge
                </h3>
                <p style={{ fontSize: '14px', lineHeight: 1.6, opacity: 0.75, margin: '0 0 24px' }}>
                  Standardized JSON/binary serialization of sensor topics, direct bag recording, and swarm multi-robot orchestration for industrial fleets.
                </p>
              </div>
              <div style={{
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '11px',
                padding: '12px',
                borderRadius: '8px',
                background: 'rgba(32,34,34,0.04)',
                border: '1px solid rgba(32,34,34,0.08)',
              }}>
                <code>topics: [/cmd_vel, /odom, /scan, /imu]</code>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── 3D EXPLODED DECONSTRUCTION SECTION (AURIX ARCHITECTURE) ── */}
      <section
        id="exploded"
        style={{
          position: 'relative',
          width: '100%',
          height: '100vh',
          background: '#0b0f14',
          color: '#f8fafc',
          overflow: 'hidden',
        }}
      >
        {/* Fullscreen Interactive 50-Frame Scrubber with Wheel Lock */}
        <AurixScrollShowcase />
      </section>

      {/* ── ROBOTS CATALOG SECTION ── */}
      <section id="robots" style={{ padding: '120px 0', background: C.ivory }}>
        <div style={{ width: 'min(94%, 1280px)', margin: '0 auto' }}>
          <div style={{ textAlign: 'center', maxWidth: '640px', margin: '0 auto 64px' }}>
            <div style={{
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '11px',
              fontWeight: 600,
              color: C.red,
              letterSpacing: '0.18em',
              marginBottom: '12px',
              textTransform: 'uppercase',
            }}>
              Mechanism Catalog
            </div>
            <h2 style={{
              fontSize: 'clamp(32px, 3.8vw, 48px)',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              margin: '0 0 16px',
            }}>
              Articulated, SCARA, Delta & Mobile.
            </h2>
            <p style={{ fontSize: '16px', color: 'rgba(32,34,34,0.7)' }}>
              Switch mechanisms instantly or import your own custom URDF XML robot definitions.
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '24px',
          }}>
            {ROBOT_CATALOG.map(robot => (
              <div
                key={robot.id}
                onClick={() => {
                  setActiveRobotId(robot.id);
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                style={{
                  background: 'white',
                  borderRadius: '16px',
                  padding: '30px',
                  border: `1.5px solid ${activeRobotId === robot.id ? C.red : 'rgba(32, 34, 34, 0.08)'}`,
                  boxShadow: activeRobotId === robot.id ? '0 16px 40px rgba(196,71,45,0.12)' : '0 4px 16px rgba(0,0,0,0.04)',
                  cursor: 'pointer',
                  transition: 'all 0.25s',
                }}
              >
                <div style={{
                  fontFamily: 'IBM Plex Mono, monospace',
                  fontSize: '11px',
                  color: C.red,
                  fontWeight: 600,
                  marginBottom: '8px',
                }}>
                  {robot.dof}
                </div>
                <h3 style={{ fontSize: '20px', fontWeight: 800, margin: '0 0 8px' }}>
                  {robot.name}
                </h3>
                <p style={{ fontSize: '13px', lineHeight: 1.6, opacity: 0.7, margin: '0 0 20px' }}>
                  {robot.description}
                </p>
                <div style={{ borderTop: '1px solid rgba(32,34,34,0.08)', paddingTop: '16px' }}>
                  {robot.features.map(f => (
                    <div key={f} style={{
                      display: 'flex', alignItems: 'center', gap: '8px',
                      fontSize: '12px', color: 'rgba(32,34,34,0.8)', marginBottom: '6px'
                    }}>
                      <ChevronRight size={12} color={C.red} />
                      <span>{f}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── AUTONOMOUS RESEARCH MODULE (Dark Graphite Section) ── */}
      <section id="research" style={{
        padding: '120px 0',
        background: C.graphite,
        color: C.ivory,
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute',
          right: '5%',
          top: '20%',
          width: '500px',
          height: '500px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(196,71,45,0.12) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />

        <div style={{ width: 'min(94%, 1280px)', margin: '0 auto', position: 'relative', zIndex: 10 }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1.1fr 1fr',
            gap: '60px',
            alignItems: 'center',
          }}>
            <div>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '6px 14px',
                borderRadius: '100px',
                background: 'rgba(196, 71, 45, 0.16)',
                border: '1px solid rgba(196, 71, 45, 0.3)',
                marginBottom: '20px',
              }}>
                <BrainCircuit size={14} color={C.red} />
                <span style={{
                  fontFamily: 'IBM Plex Mono, monospace',
                  fontSize: '11px',
                  fontWeight: 600,
                  color: C.red,
                  letterSpacing: '0.14em',
                }}>
                  AUTONOMOUS DISCOVERY BENCHMARK
                </span>
              </div>

              <h2 style={{
                fontSize: 'clamp(32px, 3.8vw, 52px)',
                fontWeight: 800,
                lineHeight: 1.1,
                letterSpacing: '-0.03em',
                margin: '0 0 20px',
              }}>
                AI that formulates hypotheses & verifies in MuJoCo.
              </h2>

              <p style={{ fontSize: '16px', lineHeight: 1.7, opacity: 0.75, margin: '0 0 32px' }}>
                RoboForge features an automated hypothesis engine that synthesizes physical conjectures, compiles test scenarios, executes multi-seed rollouts, and performs evidence-based candidate ranking.
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '36px' }}>
                {[
                  'Autonomous parameter sweep & zero-collision policy discovery',
                  'COCO JSON synthetic perception export with 3D pinhole bounding boxes',
                  'Automated evidence analyzer verifying candidate ranking against baselines',
                ].map(item => (
                  <div key={item} style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '14px' }}>
                    <div style={{
                      width: '18px', height: '18px', borderRadius: '50%',
                      background: 'rgba(45,212,191,0.2)', color: C.cyan,
                      display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}>
                      <CheckCircle2 size={12} />
                    </div>
                    <span style={{ opacity: 0.9 }}>{item}</span>
                  </div>
                ))}
              </div>

              <Link to="/studio" style={{ textDecoration: 'none' }}>
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                  background: C.red,
                  color: 'white',
                  padding: '14px 28px',
                  borderRadius: '10px',
                  fontFamily: 'Manrope, sans-serif',
                  fontWeight: 700,
                  fontSize: '13px',
                }}>
                  <span>Run Discovery Experiments</span>
                  <ArrowRight size={14} />
                </div>
              </Link>
            </div>

            {/* Research Sandbox Card Preview */}
            <div style={{
              background: '#1D2022',
              borderRadius: '20px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              padding: '28px',
              boxShadow: '0 24px 60px rgba(0,0,0,0.5)',
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid rgba(255,255,255,0.08)',
                paddingBottom: '16px',
                marginBottom: '20px',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: C.cyan }} />
                  <span style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '12px', fontWeight: 600 }}>
                    HYPOTHESIS_RUN_048
                  </span>
                </div>
                <span style={{
                  fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px',
                  background: 'rgba(45,212,191,0.15)', color: C.cyan,
                  padding: '3px 8px', borderRadius: '4px'
                }}>
                  VERIFIED (p &lt; 0.01)
                </span>
              </div>

              <div style={{
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '12px',
                background: '#121415',
                padding: '16px',
                borderRadius: '8px',
                marginBottom: '20px',
                lineHeight: 1.6,
                color: 'rgba(255,255,255,0.8)',
              }}>
                <span style={{ color: C.red }}># Objective:</span> Minimize EE vibration during 15G transition<br />
                <span style={{ color: C.cyan }}>→ Hypothesis:</span> Soft-damping torque limiter prevents overshoot<br />
                <span style={{ color: '#EAB308' }}>→ Result:</span> 38.4% reduction in peak contact shear force
              </div>

              {/* Mock SVG Convergence Curve */}
              <div style={{ height: '140px', width: '100%', position: 'relative' }}>
                <svg width="100%" height="100%" viewBox="0 0 340 120" fill="none">
                  <path d="M 10 100 Q 80 85 140 45 T 260 20 T 330 15" stroke={C.cyan} strokeWidth="2.5" />
                  <path d="M 10 105 Q 80 95 160 80 T 330 65" stroke="rgba(255,255,255,0.2)" strokeWidth="1.5" strokeDasharray="4 4" />
                  <circle cx="330" cy="15" r="4" fill={C.cyan} />
                </svg>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontFamily: 'IBM Plex Mono, monospace',
                  fontSize: '10px',
                  opacity: 0.45,
                }}>
                  <span>Epoch 0</span>
                  <span>Mean Convergence (100 Rollouts)</span>
                  <span>Epoch 500</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── CLOSING CALL TO ACTION ── */}
      <section style={{
        padding: '140px 0',
        background: `linear-gradient(180deg, ${C.ivory} 0%, #EAE4D5 100%)`,
        textAlign: 'center',
      }}>
        <div style={{ width: 'min(92%, 800px)', margin: '0 auto' }}>
          <div style={{
            fontFamily: 'IBM Plex Mono, monospace',
            fontSize: '11px',
            fontWeight: 700,
            color: C.red,
            letterSpacing: '0.2em',
            marginBottom: '16px',
            textTransform: 'uppercase',
          }}>
            READY TO LAUNCH
          </div>

          <h2 style={{
            fontSize: 'clamp(36px, 4.5vw, 64px)',
            fontWeight: 800,
            lineHeight: 1.1,
            letterSpacing: '-0.03em',
            margin: '0 0 20px',
          }}>
            Your next experiment starts here.
          </h2>

          <p style={{ fontSize: '17px', color: 'rgba(32,34,34,0.7)', lineHeight: 1.6, margin: '0 0 40px' }}>
            Open the browser workspace, inspect kinematics, and simulate true multi-body dynamics without local installation friction.
          </p>

          <div style={{ display: 'flex', justifyContent: 'center', gap: '14px', flexWrap: 'wrap' }}>
            <Link to="/studio" style={{ textDecoration: 'none' }}>
              <div style={{
                background: C.black,
                color: C.ivory,
                padding: '16px 36px',
                borderRadius: '10px',
                fontFamily: 'Manrope, sans-serif',
                fontWeight: 700,
                fontSize: '14px',
                boxShadow: '0 12px 28px rgba(32,34,34,0.25)',
              }}>
                Open Studio Workspace ↗
              </div>
            </Link>

            <Link to="/signup" style={{ textDecoration: 'none' }}>
              <div style={{
                background: 'white',
                color: C.black,
                border: '1px solid rgba(32,34,34,0.15)',
                padding: '15px 30px',
                borderRadius: '10px',
                fontFamily: 'Manrope, sans-serif',
                fontWeight: 600,
                fontSize: '14px',
              }}>
                Create Workspace Account
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer style={{
        background: C.black,
        color: 'white',
        padding: '60px 0 40px',
        borderTop: '1px solid rgba(255,255,255,0.08)',
      }}>
        <div style={{
          width: 'min(94%, 1280px)',
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '24px',
        }}>
          <div>
            <div style={{
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '13px',
              fontWeight: 700,
              letterSpacing: '0.2em',
              marginBottom: '6px',
            }}>
              ROBOFORGE STUDIO
            </div>
            <div style={{ fontSize: '12px', opacity: 0.45 }}>
              MuJoCo Physics · Three.js 3D Viewport · React 19 · MIT License
            </div>
          </div>

          <div style={{ display: 'flex', gap: '24px', fontSize: '13px', opacity: 0.7 }}>
            <span onClick={() => handleNavigateSection('platform')} style={{ cursor: 'pointer' }}>Platform</span>
            <span onClick={() => handleNavigateSection('robots')} style={{ cursor: 'pointer' }}>Mechanisms</span>
            <span onClick={() => handleNavigateSection('research')} style={{ cursor: 'pointer' }}>Research</span>
            <span onClick={() => setIsDocsOpen(true)} style={{ cursor: 'pointer' }}>Documentation</span>
            <Link to="/login" style={{ color: 'white', textDecoration: 'none' }}>Log in</Link>
          </div>
        </div>
      </footer>

      {/* Global CSS Keyframes */}
      <style>{`
        @keyframes pulseBead {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.35); opacity: 0.7; }
        }
      `}</style>
    </div>
  );
}
