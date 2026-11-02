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
    description: 'Precision SCARA with rigid cast column, dual horizontal arms, high-speed vertical ball-screw spline quill, and vacuum tool head.',
    features: ['High-speed Z-Quill Ball Screw Plunge', 'Silicone Vacuum Suction Head', 'Rigid Dual-Bearing Column Support'],
  },
  {
    id: 'delta',
    name: 'D3-1200 FlexPicker Delta',
    category: '3-Axis Parallel Delta Kinematics',
    dof: '3 Parallel Arm Pairs',
    payload: '3.0 kg',
    cycleTime: '0.15 s',
    description: 'Parallel Delta robot featuring 3 direct-drive upper arms and 6 carbon-fiber parallel link rods dynamically constrained to the tool plate.',
    features: ['Closed-Loop Parallel Rod Constraints', 'Dual Spherical Ball-Joint Linkages', 'High-G Cartesian Electronic Assembly'],
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

  // Heavy Rigid Mounting Column with Rear Reinforcing Spine
  const columnBase = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.08, 0.3), m.castIronDark);
  columnBase.position.y = 0.04;
  columnBase.castShadow = true;
  root.add(columnBase);

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

  // Arm 1 (J1: Rotates horizontally around vertical column axis)
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

  // Arm 2 (J2: Rotates horizontally around Arm 1 elbow)
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

  // J3 Linear Shaft & J4 Rotary Theta Axis
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

  const suctionCup = new THREE.Mesh(new THREE.ConeGeometry(0.028, 0.03, 16), m.vermilionAccent);
  suctionCup.rotation.x = Math.PI;
  suctionCup.position.y = -0.39;
  splineShaftGroup.add(suctionCup);

  // Picked Microchip component
  const electronicChip = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.006, 0.04), m.castIronDark);
  electronicChip.position.y = -0.408;
  splineShaftGroup.add(electronicChip);

  // PCB Workstation fixture right beneath operating envelope
  const pcbTable = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.04, 0.45), m.castIronDark);
  pcbTable.position.set(0.36, 0.02, 0.1);
  root.add(pcbTable);

  const pcbGrid = new THREE.GridHelper(0.4, 8, 0x2dd4bf, 0x555555);
  pcbGrid.position.set(0.36, 0.042, 0.1);
  root.add(pcbGrid);

  scene.add(root);
  root.scale.setScalar(1.3);

  return {
    root,
    update: (time: number, speedMultiplier: number) => {
      const t = time * 2.2 * speedMultiplier;
      arm1Group.rotation.y = Math.sin(t * 0.9) * 0.85;
      arm2Group.rotation.y = Math.cos(t * 1.2) * 1.1;

      // Realistic high-speed vertical Z plunge
      const plungePhase = Math.sin(t * 2.4);
      splineShaftGroup.position.y = Math.max(-0.16, plungePhase * 0.15 - 0.04);
      chromeQuill.rotation.y = t * 4.0;

      return {
        j1: (arm1Group.rotation.y * (180 / Math.PI)).toFixed(1),
        j2: (arm2Group.rotation.y * (180 / Math.PI)).toFixed(1),
        j3: `${(Math.abs(splineShaftGroup.position.y) * 1000).toFixed(0)} mm`,
        j4: `${((chromeQuill.rotation.y % (Math.PI * 2)) * (180 / Math.PI)).toFixed(0)}°`,
        tcp: `REPEATABILITY ±0.012mm`,
        status: 'HIGH-SPEED SMT PLACEMENT',
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

  // Top Triangular Fixed Baseplate
  const topBase = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.46, 0.06, 3, 1), m.castIronDark);
  topBase.position.y = 1.05;
  topBase.castShadow = true;
  root.add(topBase);

  const topAccent = new THREE.Mesh(new THREE.CylinderGeometry(0.43, 0.43, 0.015, 3), m.vermilionAccent);
  topAccent.position.y = 1.02;
  root.add(topAccent);

  // 3 Upper Actuated Direct-Drive Biceps
  const bicepJoints: { motorGroup: THREE.Group; arm: THREE.Mesh; elbowPoint: THREE.Object3D }[] = [];
  for (let i = 0; i < 3; i++) {
    const angle = (i / 3) * Math.PI * 2;
    const motorGroup = new THREE.Group();
    motorGroup.rotation.y = angle;
    motorGroup.position.y = 1.02;

    const motorHousing = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.12), m.castIronDark);
    motorHousing.position.set(0.24, 0, 0);
    motorGroup.add(motorHousing);

    const bicepArm = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.018, 0.34, 16), m.castAluminiumLight);
    bicepArm.position.set(0.36, -0.14, 0);
    bicepArm.rotation.z = 0.55;
    bicepArm.castShadow = true;
    motorGroup.add(bicepArm);

    // Elbow Pivot Point Marker
    const elbowPoint = new THREE.Object3D();
    elbowPoint.position.set(0.45, -0.28, 0);
    motorGroup.add(elbowPoint);

    root.add(motorGroup);
    bicepJoints.push({ motorGroup, arm: bicepArm, elbowPoint });
  }

  // Moving Lower Tool Platform
  const toolPlatform = new THREE.Group();
  toolPlatform.position.y = 0.42;
  root.add(toolPlatform);

  const eePlate = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 0.03, 3), m.castAluminiumLight);
  toolPlatform.add(eePlate);

  const eeSuctionShaft = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.01, 0.14, 16), m.chromeShaft);
  eeSuctionShaft.position.y = -0.07;
  toolPlatform.add(eeSuctionShaft);

  const eeCup = new THREE.Mesh(new THREE.ConeGeometry(0.025, 0.025, 16), m.vermilionAccent);
  eeCup.rotation.x = Math.PI;
  eeCup.position.y = -0.14;
  toolPlatform.add(eeCup);

  // 6 Dynamic Parallel Carbon Fiber Rods (2 per bicep arm)
  // These are oriented dynamically each frame using vector lookAt to guarantee true physical connection!
  const parallelRods: THREE.Mesh[] = [];
  for (let r = 0; r < 6; r++) {
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 1.0, 12), m.castIronDark);
    rod.castShadow = true;
    root.add(rod);
    parallelRods.push(rod);
  }

  // Rotary Indexing Table below Delta
  const indexingDisc = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.44, 0.04, 32), m.castIronDark);
  indexingDisc.position.y = 0.02;
  root.add(indexingDisc);

  const discGrid = new THREE.GridHelper(0.38, 12, 0xc4472d, 0x444444);
  discGrid.position.y = 0.041;
  root.add(discGrid);

  scene.add(root);
  root.scale.setScalar(1.2);

  // Reusable vectors for vector constraint math
  const worldElbowPos = new THREE.Vector3();
  const worldEEPos = new THREE.Vector3();
  const rodVec = new THREE.Vector3();
  const upAxis = new THREE.Vector3(0, 1, 0);

  return {
    root,
    update: (time: number, speedMultiplier: number) => {
      const t = time * 3.6 * speedMultiplier;

      // Realistic high-G Cartesian FlexPicker picking pattern
      toolPlatform.position.x = Math.sin(t) * 0.16;
      toolPlatform.position.z = Math.cos(t * 1.3) * 0.15;
      toolPlatform.position.y = 0.44 + Math.abs(Math.sin(t * 2.6)) * 0.14;

      // Update 3 biceps and mathematically constrain all 6 parallel rods
      bicepJoints.forEach((item, i) => {
        const bicepPitch = 0.55 + Math.sin(t + (i * Math.PI * 2) / 3) * 0.22;
        item.arm.rotation.z = bicepPitch;

        // Position of elbow in root coordinate space
        item.elbowPoint.getWorldPosition(worldElbowPos);
        root.worldToLocal(worldElbowPos);

        // Platform attachment anchor
        const platformAngle = (i / 3) * Math.PI * 2;
        const eeAnchor1 = new THREE.Vector3(
          toolPlatform.position.x + Math.cos(platformAngle + 0.15) * 0.09,
          toolPlatform.position.y,
          toolPlatform.position.z + Math.sin(platformAngle + 0.15) * 0.09
        );
        const eeAnchor2 = new THREE.Vector3(
          toolPlatform.position.x + Math.cos(platformAngle - 0.15) * 0.09,
          toolPlatform.position.y,
          toolPlatform.position.z + Math.sin(platformAngle - 0.15) * 0.09
        );

        // Rod 1 of pair: dynamically connect elbow to platform anchor
        const rod1 = parallelRods[i * 2];
        rodVec.subVectors(eeAnchor1, worldElbowPos);
        const len1 = rodVec.length();
        rod1.position.copy(worldElbowPos).addScaledVector(rodVec, 0.5);
        rod1.quaternion.setFromUnitVectors(upAxis, rodVec.clone().normalize());
        rod1.scale.set(1, len1, 1);

        // Rod 2 of pair
        const rod2 = parallelRods[i * 2 + 1];
        rodVec.subVectors(eeAnchor2, worldElbowPos);
        const len2 = rodVec.length();
        rod2.position.copy(worldElbowPos).addScaledVector(rodVec, 0.5);
        rod2.quaternion.setFromUnitVectors(upAxis, rodVec.clone().normalize());
        rod2.scale.set(1, len2, 1);
      });

      return {
        j1: `X: ${(toolPlatform.position.x * 100).toFixed(1)} cm`,
        j2: `Y: ${(toolPlatform.position.y * 100).toFixed(1)} cm`,
        j3: `Z: ${(toolPlatform.position.z * 100).toFixed(1)} cm`,
        j4: `ACCEL: 18G`,
        tcp: `220 PICKS / MIN`,
        status: 'CLOSED-LOOP PARALLEL SOLVER',
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
