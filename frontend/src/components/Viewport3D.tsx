import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls, TransformControls } from 'three-stdlib';
import { Project, SimulationState, SceneObject, JointModel } from '../types';
import {
  Camera, Compass, Crosshair, ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
  Move, RotateCw, Maximize2, X, Eye, EyeOff, Radio, Magnet, Video, Zap,
  GitFork, Layers, Route, FileUp, Trash2, Cpu, Sparkles
} from 'lucide-react';
import { importURDF, importMJCF } from '../services/api';

interface Viewport3DProps {
  project: Project;
  simState: SimulationState;
  selectedObjectId: string | null;
  onSelectObject: (id: string | null) => void;
  activeKeys: { [key: string]: boolean };
  onManualTeleop?: (linear: number, angular: number) => void;
  targetPosition?: [number, number, number] | null;
  trajectoryTrail?: Array<[number, number, number]> | null;
  onUpdateObject?: (obj: SceneObject) => void;
  onUpdateTarget?: (target: [number, number, number]) => void;
  autoDriveEnabled?: boolean;
  onToggleAutoDrive?: () => void;
}

export const Viewport3D: React.FC<Viewport3DProps> = ({
  project,
  simState,
  selectedObjectId,
  onSelectObject,
  activeKeys,
  onManualTeleop,
  targetPosition,
  trajectoryTrail,
  onUpdateObject,
  onUpdateTarget,
  autoDriveEnabled = false,
  onToggleAutoDrive,
}) => {
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const [sceneInstance, setSceneInstance] = useState<THREE.Scene | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const robotEyeCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const transformControlsRef = useRef<TransformControls | null>(null);

  // 3D Transform Gizmo Mode ('translate', 'rotate', or 'scale')
  const [transformMode, setTransformMode] = useState<'translate' | 'rotate' | 'scale'>('translate');

  // Snapping settings
  const [snapTranslate, setSnapTranslate] = useState<number>(0); // 0 (off), 0.1, 0.5, 1.0
  const [snapRotate, setSnapRotate] = useState<number>(0); // 0 (off), 15, 45, 90 deg
  const [surfaceSnap, setSurfaceSnap] = useState<boolean>(true); // Ground and stacking surface snapping
  const [attachmentSnap, setAttachmentSnap] = useState<boolean>(true); // Robot mount & EE flange snapping
  const [snapNotification, setSnapNotification] = useState<string | null>(null);

  // Onboard Camera HUD Picture-in-Picture
  const [showCameraHUD, setShowCameraHUD] = useState<boolean>(true);
  const showCameraHUDRef = useRef<boolean>(true);
  useEffect(() => {
    showCameraHUDRef.current = showCameraHUD;
  }, [showCameraHUD]);

  // 3D Trajectory Trail Overlay & Prediction Horizon State
  const [showTrajectoryTrail, setShowTrajectoryTrail] = useState<boolean>(true);
  const showTrajectoryTrailRef = useRef<boolean>(true);
  useEffect(() => {
    showTrajectoryTrailRef.current = showTrajectoryTrail;
  }, [showTrajectoryTrail]);

  const trailHistoryRef = useRef<Array<{ x: number; y: number; z: number; speed: number }>>([]);
  const trailMeshRef = useRef<THREE.LineSegments | null>(null);
  const horizonMeshRef = useRef<THREE.Line | null>(null);

  // Drag-and-Drop URDF / MJCF import state
  const [isDraggingFile, setIsDraggingFile] = useState<boolean>(false);
  const [importNotification, setImportNotification] = useState<string | null>(null);

  // Fast refs for transform event handlers
  const selectedObjectIdRef = useRef(selectedObjectId);
  const onUpdateObjectRef = useRef(onUpdateObject);
  const onUpdateTargetRef = useRef(onUpdateTarget);
  const projectRef = useRef(project);
  const surfaceSnapRef = useRef(surfaceSnap);
  const attachmentSnapRef = useRef(attachmentSnap);
  const liveEePoseRef = useRef<any>(null);
  const simStateRef = useRef<SimulationState>(simState);

  useEffect(() => {
    selectedObjectIdRef.current = selectedObjectId;
    onUpdateObjectRef.current = onUpdateObject;
    onUpdateTargetRef.current = onUpdateTarget;
    projectRef.current = project;
    surfaceSnapRef.current = surfaceSnap;
    attachmentSnapRef.current = attachmentSnap;
    simStateRef.current = simState;
  }, [selectedObjectId, onUpdateObject, onUpdateTarget, project, surfaceSnap, attachmentSnap, simState]);

  // Mesh registries for fast per-frame update
  const robotGroupRef = useRef<THREE.Group | null>(null);
  const obstaclesGroupRef = useRef<THREE.Group | null>(null);
  const genericRobotGroupRef = useRef<THREE.Group | null>(null);
  const jointMotionNodesRef = useRef<Map<string, { group: THREE.Group; joint: JointModel; axis: THREE.Vector3; type: string }>>(new Map());
  const linkNodesRef = useRef<Map<string, THREE.Group>>(new Map());

  // Kinematic Visualization Overlays Toggle State
  const [showKinematicFrames, setShowKinematicFrames] = useState<boolean>(true);
  const [showJointAxes, setShowJointAxes] = useState<boolean>(true);
  const [showJointLimits, setShowJointLimits] = useState<boolean>(true);
  const [showLinkHierarchy, setShowLinkHierarchy] = useState<boolean>(true);
  const [showEndEffectorMarker, setShowEndEffectorMarker] = useState<boolean>(true);
  const endEffectorMarkerRef = useRef<THREE.Group | null>(null);
  const [liveEePose, setLiveEePose] = useState<{
    linkName: string;
    x: number;
    y: number;
    z: number;
    roll: number;
    pitch: number;
    yaw: number;
  } | null>(null);

  const leftWheelRef = useRef<THREE.Mesh | null>(null);
  const rightWheelRef = useRef<THREE.Mesh | null>(null);
  const objectMeshesRef = useRef<Map<string, THREE.Mesh | THREE.Group>>(new Map());
  const droneRotorsRef = useRef<THREE.Mesh[]>([]);

  // Sensors and visualization refs
  const lidarTurretRef = useRef<THREE.Mesh | null>(null);
  const cameraMountGroupRef = useRef<THREE.Group | null>(null);
  const lidarLinesRef = useRef<THREE.LineSegments | null>(null);
  const contactMarkersGroupRef = useRef<THREE.Group | null>(null);

  // Smooth pose interpolation refs to eliminate jitter
  const isDraggingTransformRef = useRef<boolean>(false);
  const targetRobotPosRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0.07));
  const targetRobotQuatRef = useRef<THREE.Quaternion>(new THREE.Quaternion(0, 0, 0, 1));
  const leftWheelVelocityRef = useRef<number>(0);
  const rightWheelVelocityRef = useRef<number>(0);

  // Follow Robot Camera Mode
  const [followRobot, setFollowRobot] = useState<boolean>(true);
  const followRobotRef = useRef<boolean>(true);
  useEffect(() => {
    followRobotRef.current = followRobot;
  }, [followRobot]);

  // Synchronize Kinematic Visualization Overlays visibility
  useEffect(() => {
    if (!genericRobotGroupRef.current) return;
    genericRobotGroupRef.current.traverse((child: any) => {
      if (child.userData?.isKinematicFrame) child.visible = showKinematicFrames;
      if (child.userData?.isJointAxis) child.visible = showJointAxes;
      if (child.userData?.isJointLimit) child.visible = showJointLimits;
      if (child.userData?.isHierarchyLine) child.visible = showLinkHierarchy;
      if (child.userData?.isEndEffector) child.visible = showEndEffectorMarker;
    });
  }, [showKinematicFrames, showJointAxes, showJointLimits, showLinkHierarchy, showEndEffectorMarker]);

  // Click vs Drag Detection
  const pointerDownPos = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());

  // Helper to build/rebuild robot and obstacle meshes into Three.js scene
  const buildSceneObjects = useCallback((scene: THREE.Scene, proj: Project) => {
    if (!scene || !proj) return;

    // 1. Remove old robot group if exists
    if (robotGroupRef.current) {
      scene.remove(robotGroupRef.current);
      robotGroupRef.current.traverse((child: any) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach((m: any) => m.dispose());
          else child.material.dispose();
        }
      });
      robotGroupRef.current = null;
    }

    // 2. Ensure obstacles group exists in scene without wiping it
    let obsGrp = obstaclesGroupRef.current;
    if (!obsGrp || obsGrp.parent !== scene) {
      obsGrp = new THREE.Group();
      obsGrp.name = 'obstacles_group';
      scene.add(obsGrp);
      obstaclesGroupRef.current = obsGrp;
    }

    // 3. Create Robot Visual Model
    const rConfig = proj.robot;
    const robotGroup = new THREE.Group();
    robotGroup.name = 'robot';
    (robotGroup as any).userData = { id: 'robot', isRobot: true };

    const [hx, hy, hz] = rConfig.chassis_size;
    const wr = rConfig.wheel_radius;
    const ww = rConfig.wheel_width;
    const wb = rConfig.wheel_base;
    const cr = rConfig.caster_radius;

    // Main Chassis Box
    const chassisGeo = new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2);
    const chassisMat = new THREE.MeshStandardMaterial({
      color: 0x2563eb,
      metalness: 0.3,
      roughness: 0.4,
    });
    const chassisMesh = new THREE.Mesh(chassisGeo, chassisMat);
    chassisMesh.castShadow = true;
    chassisMesh.receiveShadow = true;
    robotGroup.add(chassisMesh);

    // Forward Direction Arrow / Nose
    const noseGeo = new THREE.BoxGeometry(hx * 0.5, hy * 0.5, hz * 0.8);
    const noseMat = new THREE.MeshStandardMaterial({
      color: 0xf97316,
      roughness: 0.3,
    });
    const noseMesh = new THREE.Mesh(noseGeo, noseMat);
    noseMesh.position.set(hx * 0.85, 0, hz * 0.3);
    noseMesh.castShadow = true;
    robotGroup.add(noseMesh);

    // Local Robot Base Axes Helper (RGB Coordinate Frame)
    const robotAxes = new THREE.AxesHelper(0.35);
    robotAxes.position.set(0, 0, hz + 0.02);
    robotGroup.add(robotAxes);

    // Left Wheel (Cylinder)
    const wheelGeo = new THREE.CylinderGeometry(wr, wr, ww, 24);
    const wheelMat = new THREE.MeshStandardMaterial({
      color: 0x22242a,
      roughness: 0.8,
      metalness: 0.2,
    });
    const leftWheel = new THREE.Mesh(wheelGeo, wheelMat);
    leftWheel.rotation.x = Math.PI / 2;
    leftWheel.position.set(0, wb * 0.5, -0.01);
    leftWheel.castShadow = true;
    robotGroup.add(leftWheel);
    leftWheelRef.current = leftWheel;

    // Right Wheel (Cylinder)
    const rightWheel = new THREE.Mesh(wheelGeo, wheelMat);
    rightWheel.rotation.x = Math.PI / 2;
    rightWheel.position.set(0, -wb * 0.5, -0.01);
    rightWheel.castShadow = true;
    robotGroup.add(rightWheel);
    rightWheelRef.current = rightWheel;

    // 4 Corner Stabilizer Casters
    const casterGeo = new THREE.SphereGeometry(cr, 16, 16);
    const casterMat = new THREE.MeshStandardMaterial({ color: 0x4a4d56, roughness: 0.5 });
    const casterX = hx * 0.75;
    const casterY = wb * 0.35;
    const casterZ = -0.01 - wr + cr;

    const casterFL = new THREE.Mesh(casterGeo, casterMat);
    casterFL.position.set(casterX, casterY, casterZ);
    robotGroup.add(casterFL);

    const casterFR = new THREE.Mesh(casterGeo, casterMat);
    casterFR.position.set(casterX, -casterY, casterZ);
    robotGroup.add(casterFR);

    const casterRL = new THREE.Mesh(casterGeo, casterMat);
    casterRL.position.set(-casterX, casterY, casterZ);
    robotGroup.add(casterRL);

    const casterRR = new THREE.Mesh(casterGeo, casterMat);
    casterRR.position.set(-casterX, -casterY, casterZ);
    robotGroup.add(casterRR);

    // Visible Simulated LiDAR Sensor Placement (Toggleable)
    const lidarCfg = proj.sensors?.lidar;
    const lidarMount = lidarCfg?.mount_pos || [0.10, 0.0, 0.09];
    const lidarTurretGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.03, 16);
    const lidarTurretMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.7, roughness: 0.3 });
    const lidarMesh = new THREE.Mesh(lidarTurretGeo, lidarTurretMat);
    lidarMesh.rotation.x = Math.PI / 2;
    lidarMesh.position.set(lidarMount[0], lidarMount[1], lidarMount[2]);
    const lidarLens = new THREE.Mesh(
      new THREE.CylinderGeometry(0.025, 0.025, 0.01, 16),
      new THREE.MeshBasicMaterial({ color: 0x0284c7 })
    );
    lidarLens.position.set(0, 0.016, 0);
    lidarMesh.add(lidarLens);
    const lidarAxes = new THREE.AxesHelper(0.12);
    lidarMesh.add(lidarAxes);
    if (lidarCfg && lidarCfg.enabled === false) {
      lidarMesh.visible = false;
    }
    robotGroup.add(lidarMesh);
    lidarTurretRef.current = lidarMesh;

    // Visible Simulated IMU Sensor Placement (Toggleable)
    const imuCfg = proj.sensors?.imu;
    const imuMount = imuCfg?.mount_pos || [0.0, 0.0, 0.05];
    const imuChipGeo = new THREE.BoxGeometry(0.025, 0.025, 0.01);
    const imuChipMat = new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.5 });
    const imuMesh = new THREE.Mesh(imuChipGeo, imuChipMat);
    imuMesh.position.set(imuMount[0], imuMount[1], imuMount[2]);
    const imuAxes = new THREE.AxesHelper(0.08);
    imuMesh.add(imuAxes);
    if (imuCfg && imuCfg.enabled === false) {
      imuMesh.visible = false;
    }
    robotGroup.add(imuMesh);

    // Visible Camera Sensor Placement & Onboard Camera (Toggleable)
    const camCfg = proj.sensors?.camera;
    const camMount = camCfg?.mount_pos || [0.18, 0.0, 0.08];
    const camGroup = new THREE.Group();
    camGroup.name = 'camera_sensor';
    camGroup.position.set(camMount[0], camMount[1], camMount[2]);

    const camHousingGeo = new THREE.CylinderGeometry(0.018, 0.022, 0.035, 16);
    const camHousingMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.8, roughness: 0.2 });
    const camHousingMesh = new THREE.Mesh(camHousingGeo, camHousingMat);
    camHousingMesh.rotation.z = Math.PI / 2;
    camGroup.add(camHousingMesh);

    const camLensGeo = new THREE.CylinderGeometry(0.014, 0.014, 0.005, 16);
    const camLensMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.9, roughness: 0.1 });
    const camLensMesh = new THREE.Mesh(camLensGeo, camLensMat);
    camLensMesh.position.set(0.019, 0, 0);
    camLensMesh.rotation.z = Math.PI / 2;
    camGroup.add(camLensMesh);

    if (camCfg && camCfg.enabled === false) {
      camGroup.visible = false;
    }
    robotGroup.add(camGroup);
    cameraMountGroupRef.current = camGroup;

    // Attach Three.js perspective camera to robot body for vision navigation
    const robotEye = new THREE.PerspectiveCamera(camCfg?.fov_deg || 65, 4 / 3, 0.05, 40);
    robotEye.up.set(0, 0, 1);
    robotEye.position.set(camMount[0] + 0.03, camMount[1], camMount[2]);
    robotEye.rotation.set(-Math.PI / 2, 0, -Math.PI / 2);
    robotGroup.add(robotEye);
    robotEyeCameraRef.current = robotEye;

    // Initial position from robot config
    const [initX, initY, initZ] = rConfig.initial_position;
    robotGroup.position.set(initX, initY, initZ);
    scene.add(robotGroup);
    robotGroupRef.current = robotGroup;

    // 4. Generic Articulated Robot Kinematic Tree
    const oldGeneric = scene.getObjectByName('generic_robot_group');
    if (oldGeneric) {
      scene.remove(oldGeneric);
      oldGeneric.traverse((child: any) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach((m: any) => m.dispose());
          else child.material.dispose();
        }
      });
      genericRobotGroupRef.current = null;
    }
    jointMotionNodesRef.current.clear();
    linkNodesRef.current.clear();

    const gr = proj.generic_robot;
    if (gr && gr.links && gr.links.length > 0) {
      const grGroup = new THREE.Group();
      grGroup.name = 'generic_robot_group';
      (grGroup as any).userData = { id: 'generic_robot', isRobot: true };

      const bPos = gr.base_position || [2.5, 2.5, 0.05];
      const bRpy = gr.base_rpy || [0, 0, 0];
      grGroup.position.set(bPos[0], bPos[1], bPos[2]);
      grGroup.rotation.set(bRpy[0], bRpy[1], bRpy[2]);

      // Base coordinate frame
      const baseFrame = new THREE.AxesHelper(0.3);
      (baseFrame as any).userData = { isKinematicFrame: true };
      baseFrame.visible = showKinematicFrames;
      grGroup.add(baseFrame);

      // Build link-joint tree
      const childJointsByParent = new Map<string, JointModel[]>();
      const parentJointByChild = new Map<string, JointModel>();
      (gr.joints || []).forEach((j) => {
        if (!childJointsByParent.has(j.parent_link)) {
          childJointsByParent.set(j.parent_link, []);
        }
        childJointsByParent.get(j.parent_link)!.push(j);
        parentJointByChild.set(j.child_link, j);
      });

      // Find root link
      const rootLinkName =
        gr.root_link ||
        gr.links.find((l) => !parentJointByChild.has(l.name))?.name ||
        gr.links[0].name;

      // Identify end effector links (links without outgoing joints)
      const leafLinkNames = new Set(
        gr.links
          .filter((l) => !childJointsByParent.has(l.name) || childJointsByParent.get(l.name)!.length === 0)
          .map((l) => l.name)
      );

      const buildLinkNode = (linkName: string): THREE.Group => {
        const linkGroup = new THREE.Group();
        linkGroup.name = `link_${linkName}`;
        (linkGroup as any).userData = { id: 'generic_robot', linkName, isRobot: true };
        linkNodesRef.current.set(linkName, linkGroup);

        const linkData = gr.links.find((l) => l.name === linkName);
        if (linkData) {
          const geom = linkData.visual || linkData.collision;
          if (geom) {
            let gGeo: THREE.BufferGeometry;
            if (geom.type === 'box') {
              gGeo = new THREE.BoxGeometry(
                Math.max(0.01, (geom.size[0] || 0.1) * 2),
                Math.max(0.01, (geom.size[1] || 0.1) * 2),
                Math.max(0.01, (geom.size[2] || 0.1) * 2)
              );
            } else if (geom.type === 'sphere') {
              gGeo = new THREE.SphereGeometry(Math.max(0.01, geom.size[0] || 0.1), 16, 16);
            } else {
              gGeo = new THREE.CylinderGeometry(
                Math.max(0.005, geom.size[0] || 0.05),
                Math.max(0.005, geom.size[0] || 0.05),
                Math.max(0.01, (geom.size[1] || 0.15) * 2),
                20
              );
            }
            const mat = new THREE.MeshStandardMaterial({
              color: new THREE.Color(geom.color || '#3b82f6'),
              metalness: 0.35,
              roughness: 0.45,
            });
            const mesh = new THREE.Mesh(gGeo, mat);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            mesh.position.set(geom.origin_pos[0] || 0, geom.origin_pos[1] || 0, geom.origin_pos[2] || 0);
            if (geom.origin_rpy) {
              mesh.rotation.set(geom.origin_rpy[0] || 0, geom.origin_rpy[1] || 0, geom.origin_rpy[2] || 0);
            }
            (mesh as any).userData = { id: 'generic_robot', linkName, isRobot: true };
            linkGroup.add(mesh);
          }
        }

        // If this is a leaf link (End-Effector), place the End-Effector visual marker
        if (leafLinkNames.has(linkName)) {
          const eeGroup = new THREE.Group();
          eeGroup.name = `ee_marker_${linkName}`;
          (eeGroup as any).userData = { isEndEffector: true, linkName };
          eeGroup.visible = showEndEffectorMarker;
          endEffectorMarkerRef.current = eeGroup;

          // Glowing tip marker
          const eeSphere = new THREE.Mesh(
            new THREE.SphereGeometry(0.028, 16, 16),
            new THREE.MeshStandardMaterial({
              color: 0x10b981,
              emissive: 0x10b981,
              emissiveIntensity: 0.8,
              roughness: 0.2,
            })
          );
          eeGroup.add(eeSphere);

          // EE Frame
          const eeAxes = new THREE.AxesHelper(0.15);
          (eeAxes as any).userData = { isKinematicFrame: true };
          eeAxes.visible = showKinematicFrames;
          eeGroup.add(eeAxes);

          linkGroup.add(eeGroup);
        }

        // Attach outgoing child joints
        const childJoints = childJointsByParent.get(linkName) || [];
        childJoints.forEach((j) => {
          const jointOriginGroup = new THREE.Group();
          jointOriginGroup.name = `joint_origin_${j.name}`;
          jointOriginGroup.position.set(j.origin_pos[0] || 0, j.origin_pos[1] || 0, j.origin_pos[2] || 0);
          if (j.origin_rpy) {
            jointOriginGroup.rotation.set(j.origin_rpy[0] || 0, j.origin_rpy[1] || 0, j.origin_rpy[2] || 0);
          }

          // 1. Joint Coordinate Frame
          const jFrame = new THREE.AxesHelper(0.16);
          (jFrame as any).userData = { isKinematicFrame: true };
          jFrame.visible = showKinematicFrames;
          jointOriginGroup.add(jFrame);

          // 2. Joint Axis Vector
          const axisDir = new THREE.Vector3(j.axis[0] || 0, j.axis[1] || 0, j.axis[2] || 1).normalize();
          const arrowHelper = new THREE.ArrowHelper(axisDir, new THREE.Vector3(0, 0, 0), 0.22, 0xfacc15, 0.05, 0.035);
          (arrowHelper as any).userData = { isJointAxis: true };
          arrowHelper.visible = showJointAxes;
          jointOriginGroup.add(arrowHelper);

          // 3. Joint Limits Overlay
          if (j.type === 'revolute' || j.type === 'continuous') {
            const lower = j.limit?.lower ?? -Math.PI;
            const upper = j.limit?.upper ?? Math.PI;
            const arcRadius = 0.12;
            const arcSegments = 24;
            const arcPoints: THREE.Vector3[] = [];
            // Create arc in plane perpendicular to axis
            const up = Math.abs(axisDir.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
            const u = new THREE.Vector3().crossVectors(axisDir, up).normalize();
            const v = new THREE.Vector3().crossVectors(axisDir, u).normalize();

            for (let s = 0; s <= arcSegments; s++) {
              const theta = lower + (s / arcSegments) * (upper - lower);
              const pt = new THREE.Vector3()
                .addScaledVector(u, Math.cos(theta) * arcRadius)
                .addScaledVector(v, Math.sin(theta) * arcRadius);
              arcPoints.push(pt);
            }
            const arcGeo = new THREE.BufferGeometry().setFromPoints(arcPoints);
            const arcMat = new THREE.LineBasicMaterial({ color: 0x06b6d4, linewidth: 2, transparent: true, opacity: 0.8 });
            const arcLine = new THREE.Line(arcGeo, arcMat);
            (arcLine as any).userData = { isJointLimit: true };
            arcLine.visible = showJointLimits;
            jointOriginGroup.add(arcLine);
          } else if (j.type === 'prismatic') {
            const lower = j.limit?.lower ?? -0.2;
            const upper = j.limit?.upper ?? 0.2;
            const pPoints = [
              new THREE.Vector3().copy(axisDir).multiplyScalar(lower),
              new THREE.Vector3().copy(axisDir).multiplyScalar(upper),
            ];
            const pGeo = new THREE.BufferGeometry().setFromPoints(pPoints);
            const pMat = new THREE.LineBasicMaterial({ color: 0xa855f7, linewidth: 2, transparent: true, opacity: 0.8 });
            const pLine = new THREE.Line(pGeo, pMat);
            (pLine as any).userData = { isJointLimit: true };
            pLine.visible = showJointLimits;
            jointOriginGroup.add(pLine);
          }

          // 4. Link Hierarchy Skeleton Line (parent link origin -> joint origin)
          const skPoints = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(j.origin_pos[0], j.origin_pos[1], j.origin_pos[2])];
          const skGeo = new THREE.BufferGeometry().setFromPoints(skPoints);
          const skMat = new THREE.LineDashedMaterial({ color: 0x94a3b8, dashSize: 0.03, gapSize: 0.02 });
          const skLine = new THREE.Line(skGeo, skMat);
          skLine.computeLineDistances();
          (skLine as any).userData = { isHierarchyLine: true };
          skLine.visible = showLinkHierarchy;
          linkGroup.add(skLine);

          // 5. Joint Motion Group
          const jointMotionGroup = new THREE.Group();
          jointMotionGroup.name = `joint_motion_${j.name}`;
          jointOriginGroup.add(jointMotionGroup);

          jointMotionNodesRef.current.set(j.name, {
            group: jointMotionGroup,
            joint: j,
            axis: axisDir,
            type: j.type,
          });

          // Recursively build downstream child link
          const childLinkNode = buildLinkNode(j.child_link);
          jointMotionGroup.add(childLinkNode);

          linkGroup.add(jointOriginGroup);
        });

        return linkGroup;
      };

      const rootNode = buildLinkNode(rootLinkName);
      grGroup.add(rootNode);

      scene.add(grGroup);
      genericRobotGroupRef.current = grGroup;
    }

    // 5. World Containment Boundary Wireframe Box
    const oldBounds = scene.getObjectByName('boundary_wireframe');
    if (oldBounds) scene.remove(oldBounds);
    if (proj.settings?.enable_boundaries) {
      const b = proj.settings.world_boundaries || [-15, 15, -15, 15, 0, 8];
      const bx = (b[1] - b[0]);
      const by = (b[3] - b[2]);
      const bz = (b[5] - b[4]);
      const midX = (b[0] + b[1]) * 0.5;
      const midY = (b[2] + b[3]) * 0.5;
      const midZ = (b[4] + b[5]) * 0.5;

      const boundGeo = new THREE.BoxGeometry(bx, by, bz);
      const wireMat = new THREE.LineBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.35 });
      const wireMesh = new THREE.LineSegments(new THREE.EdgesGeometry(boundGeo), wireMat);
      wireMesh.name = 'boundary_wireframe';
      wireMesh.position.set(midX, midY, midZ);
      scene.add(wireMesh);
    }

    // 6. Incremental Synchronization of Obstacle Meshes
    // This preserves existing meshes and their references, preventing TransformControls detachment
    const currentObjectIds = new Set(proj.objects.map((o) => o.id));

    // Remove deleted objects
    objectMeshesRef.current.forEach((mesh, id) => {
      if (!currentObjectIds.has(id)) {
        obsGrp!.remove(mesh);
        mesh.traverse((child: any) => {
          if (child.geometry) child.geometry.dispose();
          if (child.material) {
            if (Array.isArray(child.material)) child.material.forEach((m: any) => m.dispose());
            else child.material.dispose();
          }
        });
        objectMeshesRef.current.delete(id);
      }
    });

    // Clear previous drone rotors
    droneRotorsRef.current = [];

    // Add or update existing objects
    proj.objects.forEach((obj: SceneObject) => {
      let mesh = objectMeshesRef.current.get(obj.id);
      const isDrone = obj.name.toLowerCase().includes('drone') || obj.id.startsWith('drone');
      const isSecondaryRobot = (obj.name.toLowerCase().includes('robot') || obj.id.startsWith('robot') || obj.name.toLowerCase().includes('forgebot')) && !isDrone;

      const createDifferentialRobotGroup = (): THREE.Group => {
        const rob = new THREE.Group();
        rob.name = `obj_${obj.id}`;
        (rob as any).userData = { id: obj.id, isSecondaryRobot: true, isObstacle: false };

        const chassisColor = new THREE.Color(obj.color || '#06b6d4');

        // 1. Sleek Main Chassis
        const hx = 0.18, hy = 0.12, hz = 0.08;
        const chassisGeo = new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2);
        const chassisMat = new THREE.MeshStandardMaterial({
          color: chassisColor,
          metalness: 0.4,
          roughness: 0.35,
        });
        const chassisMesh = new THREE.Mesh(chassisGeo, chassisMat);
        chassisMesh.castShadow = true;
        chassisMesh.receiveShadow = true;
        rob.add(chassisMesh);

        // 2. High-visibility Forward Direction Nose
        const noseGeo = new THREE.BoxGeometry(hx * 0.45, hy * 0.45, hz * 0.7);
        const noseMat = new THREE.MeshStandardMaterial({
          color: 0xf97316,
          roughness: 0.3,
          metalness: 0.2,
        });
        const noseMesh = new THREE.Mesh(noseGeo, noseMat);
        noseMesh.position.set(hx * 0.85, 0, hz * 0.25);
        noseMesh.castShadow = true;
        rob.add(noseMesh);

        // 3. Dual Drive Wheels
        const wr = 0.045, ww = 0.024, wb = 0.28;
        const wheelGeo = new THREE.CylinderGeometry(wr, wr, ww, 24);
        const wheelMat = new THREE.MeshStandardMaterial({
          color: 0x1e293b,
          roughness: 0.8,
          metalness: 0.2,
        });
        const leftWheel = new THREE.Mesh(wheelGeo, wheelMat);
        leftWheel.rotation.x = Math.PI / 2;
        leftWheel.position.set(0, wb * 0.5, -0.01);
        leftWheel.castShadow = true;
        rob.add(leftWheel);

        const rightWheel = new THREE.Mesh(wheelGeo, wheelMat);
        rightWheel.rotation.x = Math.PI / 2;
        rightWheel.position.set(0, -wb * 0.5, -0.01);
        rightWheel.castShadow = true;
        rob.add(rightWheel);

        // 4. Stabilizing Front & Rear Passive Casters
        const casterGeo = new THREE.SphereGeometry(0.02, 16, 16);
        const casterMat = new THREE.MeshStandardMaterial({
          color: 0x94a3b8,
          metalness: 0.9,
          roughness: 0.1,
        });
        const frontCaster = new THREE.Mesh(casterGeo, casterMat);
        frontCaster.position.set(hx * 0.75, 0, -hz - 0.005);
        rob.add(frontCaster);

        const rearCaster = new THREE.Mesh(casterGeo, casterMat);
        rearCaster.position.set(-hx * 0.75, 0, -hz - 0.005);
        rob.add(rearCaster);

        // 5. 2D LiDAR Turret
        const lidarGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.03, 16);
        const lidarMat = new THREE.MeshStandardMaterial({
          color: 0x0f172a,
          metalness: 0.8,
          roughness: 0.2,
        });
        const lidarMesh = new THREE.Mesh(lidarGeo, lidarMat);
        lidarMesh.position.set(0, 0, hz + 0.015);
        lidarMesh.rotation.x = Math.PI / 2;
        rob.add(lidarMesh);

        // LiDAR Laser Emitter Ring
        const ringGeo = new THREE.TorusGeometry(0.036, 0.004, 8, 24);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.position.set(0, 0, hz + 0.015);
        rob.add(ringMesh);

        return rob;
      };

      const createDroneGroup = (): THREE.Group => {
        const drone = new THREE.Group();
        drone.name = `obj_${obj.id}`;
        (drone as any).userData = { id: obj.id, isDrone: true, isObstacle: false };

        // 1. Central Avionics Fuselage
        const bodyGeo = new THREE.BoxGeometry(0.18, 0.18, 0.07);
        const bodyMat = new THREE.MeshStandardMaterial({
          color: 0x0f172a,
          metalness: 0.85,
          roughness: 0.25,
        });
        const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
        bodyMesh.castShadow = true;
        drone.add(bodyMesh);

        // Glowing Drone Status Core
        const coreGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.02, 16);
        const coreMat = new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          emissive: 0x38bdf8,
          emissiveIntensity: 0.8,
        });
        const coreMesh = new THREE.Mesh(coreGeo, coreMat);
        coreMesh.position.set(0, 0, 0.045);
        drone.add(coreMesh);

        // 2. Diagonal Carbon Fiber Arms & 4 Propeller Rotors
        const armLength = 0.24;
        const armGeo = new THREE.BoxGeometry(armLength * 2, 0.02, 0.015);
        const armMat = new THREE.MeshStandardMaterial({
          color: 0x334155,
          metalness: 0.7,
          roughness: 0.4,
        });

        const arm1 = new THREE.Mesh(armGeo, armMat);
        arm1.rotation.z = Math.PI / 4;
        drone.add(arm1);

        const arm2 = new THREE.Mesh(armGeo, armMat);
        arm2.rotation.z = -Math.PI / 4;
        drone.add(arm2);

        // Motor Hubs & Rotors at 4 corners
        const dOffset = armLength * Math.cos(Math.PI / 4);
        const motorPositions = [
          [dOffset, dOffset],
          [-dOffset, dOffset],
          [-dOffset, -dOffset],
          [dOffset, -dOffset],
        ];

        motorPositions.forEach(([mx, my]) => {
          // Motor can
          const motorGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.03, 12);
          const motorMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.9, roughness: 0.2 });
          const motor = new THREE.Mesh(motorGeo, motorMat);
          motor.position.set(mx, my, 0.02);
          motor.rotation.x = Math.PI / 2;
          drone.add(motor);

          // Translucent Spinning Rotor Disc
          const rotorGeo = new THREE.CircleGeometry(0.12, 16);
          const rotorMat = new THREE.MeshStandardMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.45,
            side: THREE.DoubleSide,
          });
          const rotorMesh = new THREE.Mesh(rotorGeo, rotorMat);
          rotorMesh.position.set(mx, my, 0.04);
          drone.add(rotorMesh);
          droneRotorsRef.current.push(rotorMesh);
        });

        // 3. Ground Altitude Line & Landing Shadow
        const altPoints = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -obj.position[2])];
        const altGeo = new THREE.BufferGeometry().setFromPoints(altPoints);
        const altMat = new THREE.LineDashedMaterial({ color: 0x38bdf8, dashSize: 0.05, gapSize: 0.04, transparent: true, opacity: 0.6 });
        const altLine = new THREE.Line(altGeo, altMat);
        altLine.computeLineDistances();
        drone.add(altLine);

        const shadowGeo = new THREE.RingGeometry(0.15, 0.22, 24);
        const shadowMat = new THREE.MeshBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.4, side: THREE.DoubleSide });
        const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
        shadowMesh.position.set(0, 0, -obj.position[2] + 0.005);
        drone.add(shadowMesh);

        return drone;
      };

      const createGeometry = () => {
        if (obj.type === 'box') {
          const sx = (obj.size[0] || 0.2) * 2;
          const sy = (obj.size[1] || 0.2) * 2;
          const sz = (obj.size[2] || 0.2) * 2;
          return new THREE.BoxGeometry(sx, sy, sz);
        } else if (obj.type === 'sphere') {
          const radius = obj.size[0] || 0.2;
          return new THREE.SphereGeometry(radius, 24, 24);
        } else if (obj.type === 'cylinder') {
          const radius = obj.size[0] || 0.2;
          const height = (obj.size[1] || 0.3) * 2;
          return new THREE.CylinderGeometry(radius, radius, height, 24);
        }
        return new THREE.BoxGeometry(0.4, 0.4, 0.4);
      };

      if (!mesh) {
        if (isDrone) {
          const droneGrp = createDroneGroup();
          droneGrp.position.set(obj.position[0], obj.position[1], obj.position[2]);
          droneGrp.visible = obj.visible !== false;
          obsGrp!.add(droneGrp);
          objectMeshesRef.current.set(obj.id, droneGrp as any);
        } else if (isSecondaryRobot) {
          const robGrp = createDifferentialRobotGroup();
          robGrp.position.set(obj.position[0], obj.position[1], obj.position[2]);
          robGrp.rotation.set(
            ((obj.rotation[0] || 0) * Math.PI) / 180,
            ((obj.rotation[1] || 0) * Math.PI) / 180,
            ((obj.rotation[2] || 0) * Math.PI) / 180
          );
          robGrp.visible = obj.visible !== false;
          obsGrp!.add(robGrp);
          objectMeshesRef.current.set(obj.id, robGrp as any);
        } else {
          // Standard obstacle mesh
          const geo = createGeometry();
          const mat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(obj.color || '#8892b0'),
            roughness: 0.4,
            metalness: 0.2,
          });

          mesh = new THREE.Mesh(geo, mat);
          mesh.name = `obj_${obj.id}`;
          (mesh as any).userData = { id: obj.id, isObstacle: true };
          mesh.castShadow = true;
          mesh.receiveShadow = true;

          mesh.position.set(obj.position[0], obj.position[1], obj.position[2]);
          mesh.rotation.set(
            (obj.rotation[0] * Math.PI) / 180,
            (obj.rotation[1] * Math.PI) / 180,
            (obj.rotation[2] * Math.PI) / 180
          );
          mesh.visible = obj.visible !== false;

          obsGrp!.add(mesh);
          objectMeshesRef.current.set(obj.id, mesh);
        }
      } else {
        // Update color and visibility
        if ((mesh as any).material) {
          const mat = (mesh as any).material as THREE.MeshStandardMaterial;
          mat.color?.set(obj.color || '#8892b0');
        }
        mesh.visible = obj.visible !== false;

        // If not actively being dragged, update position & rotation to match project
        if (!(isDraggingTransformRef.current && selectedObjectIdRef.current === obj.id)) {
          mesh.position.set(obj.position[0], obj.position[1], obj.position[2]);
          mesh.rotation.set(
            (obj.rotation[0] * Math.PI) / 180,
            (obj.rotation[1] * Math.PI) / 180,
            (obj.rotation[2] * Math.PI) / 180
          );
        }
      }
    });
  }, []);

  // Camera Presets
  const setCameraPreset = (preset: 'top' | 'front' | 'side' | 'iso') => {
    if (!cameraRef.current || !controlsRef.current) return;
    const c = cameraRef.current;
    const ctrl = controlsRef.current;
    const rp = robotGroupRef.current ? robotGroupRef.current.position : new THREE.Vector3(0, 0, 0);

    setFollowRobot(false);
    if (preset === 'top') {
      c.position.set(rp.x, rp.y, rp.z + 8);
      ctrl.target.set(rp.x, rp.y, rp.z);
    } else if (preset === 'front') {
      c.position.set(rp.x + 4.5, rp.y, rp.z + 1.2);
      ctrl.target.set(rp.x, rp.y, rp.z + 0.1);
    } else if (preset === 'side') {
      c.position.set(rp.x, rp.y + 4.5, rp.z + 1.2);
      ctrl.target.set(rp.x, rp.y, rp.z + 0.1);
    } else {
      c.position.set(rp.x + 3.5, rp.y - 3.5, rp.z + 3.0);
      ctrl.target.set(rp.x, rp.y, rp.z);
    }
    ctrl.update();
  };

  const focusOnRobot = () => {
    if (!cameraRef.current || !controlsRef.current || !robotGroupRef.current) return;
    const rp = robotGroupRef.current.position;
    controlsRef.current.target.set(rp.x, rp.y, rp.z);
    cameraRef.current.position.set(rp.x + 2.5, rp.y - 2.5, rp.z + 2.0);
    controlsRef.current.update();
  };

  // Initialize Three.js Scene, Camera, Controls
  useEffect(() => {
    const container = canvasContainerRef.current;
    if (!container) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e0f13);
    sceneRef.current = scene;
    setSceneInstance(scene);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.05, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(3.5, -3.5, 3.0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2 - 0.01;
    controls.target.set(0, 0, 0.2);
    controlsRef.current = controls;

    const transformControls = new TransformControls(camera, renderer.domElement);
    (transformControls as any).size = 0.75;
    scene.add(transformControls);
    transformControlsRef.current = transformControls;

    // Commit transforms
    const commitTransform = () => {
      const curSelected = selectedObjectIdRef.current;
      if (!curSelected || curSelected === 'robot') return;

      if (curSelected === 'target') {
        const beacon = scene.getObjectByName('target_beacon');
        if (beacon && onUpdateTargetRef.current) {
          onUpdateTargetRef.current([
            parseFloat(beacon.position.x.toFixed(3)),
            parseFloat(beacon.position.y.toFixed(3)),
            0.25,
          ]);
        }
        return;
      }

      const mesh = objectMeshesRef.current.get(curSelected);
      if (mesh && onUpdateObjectRef.current) {
        const obj = projectRef.current?.objects.find((o) => o.id === curSelected);
        if (obj) {
          const degX = parseFloat(((mesh.rotation.x * 180) / Math.PI).toFixed(1));
          const degY = parseFloat(((mesh.rotation.y * 180) / Math.PI).toFixed(1));
          const degZ = parseFloat(((mesh.rotation.z * 180) / Math.PI).toFixed(1));

          let newSize = [...obj.size];
          if (mesh.scale.x !== 1 || mesh.scale.y !== 1 || mesh.scale.z !== 1) {
            if (obj.type === 'box') {
              newSize = [
                parseFloat((obj.size[0] * mesh.scale.x).toFixed(3)),
                parseFloat((obj.size[1] * mesh.scale.y).toFixed(3)),
                parseFloat((obj.size[2] * mesh.scale.z).toFixed(3)),
              ];
            } else if (obj.type === 'sphere') {
              const maxScale = Math.max(mesh.scale.x, mesh.scale.y, mesh.scale.z);
              newSize = [parseFloat((obj.size[0] * maxScale).toFixed(3))];
            } else if (obj.type === 'cylinder') {
              newSize = [
                parseFloat((obj.size[0] * Math.max(mesh.scale.x, mesh.scale.y)).toFixed(3)),
                parseFloat((obj.size[1] * mesh.scale.z).toFixed(3)),
              ];
            }
            mesh.scale.set(1, 1, 1);
          }

          let finalX = mesh.position.x;
          let finalY = mesh.position.y;
          let finalZ = mesh.position.z;
          let finalDegX = degX;
          let finalDegY = degY;
          let finalDegZ = degZ;

          // Compute geometric vertical half-height and horizontal radii
          let halfHeight = 0.1;
          let halfX = 0.1;
          let halfY = 0.1;
          if (obj.type === 'box') {
            halfX = newSize[0] || 0.1;
            halfY = newSize[1] || 0.1;
            halfHeight = newSize[2] || 0.1;
          } else if (obj.type === 'cylinder') {
            halfX = newSize[0] || 0.1;
            halfY = newSize[0] || 0.1;
            halfHeight = newSize[1] || 0.1;
          } else if (obj.type === 'sphere') {
            halfX = newSize[0] || 0.1;
            halfY = newSize[0] || 0.1;
            halfHeight = newSize[0] || 0.1;
          }

          let snapMsg: string | null = null;

          // 1. Robot Attachment Snapping (if not static)
          if (attachmentSnapRef.current && !obj.is_static) {
            // A. Check mobile robot chassis mount point
            const robotState = simStateRef.current?.robot;
            const robotCfg = projectRef.current?.robot;
            if (robotCfg) {
              const rx = robotState?.position ? robotState.position[0] : robotCfg.initial_position[0];
              const ry = robotState?.position ? robotState.position[1] : robotCfg.initial_position[1];
              const rz = robotState?.position ? robotState.position[2] : robotCfg.initial_position[2];
              const mountZ = rz + (robotCfg.chassis_size?.[2] || 0.1) + halfHeight;
              const distXY = Math.hypot(finalX - rx, finalY - ry);
              if (distXY < 0.35 && Math.abs(finalZ - mountZ) < 0.35) {
                finalX = rx;
                finalY = ry;
                finalZ = parseFloat(mountZ.toFixed(3));
                snapMsg = `Attached to Robot Mount (x=${finalX.toFixed(2)}, y=${finalY.toFixed(2)})`;
              }
            }

            // B. Check Articulated Robot End-Effector Flange
            if (!snapMsg && liveEePoseRef.current) {
              const ee = liveEePoseRef.current;
              const distEE = Math.hypot(finalX - ee.x, finalY - ee.y, finalZ - ee.z);
              if (distEE < 0.35) {
                finalX = parseFloat(ee.x.toFixed(3));
                finalY = parseFloat(ee.y.toFixed(3));
                finalZ = parseFloat(ee.z.toFixed(3));
                finalDegX = parseFloat(ee.roll.toFixed(1));
                finalDegY = parseFloat(ee.pitch.toFixed(1));
                finalDegZ = parseFloat(ee.yaw.toFixed(1));
                snapMsg = `Attached to Robot Flange (${ee.linkName})`;
              }
            }
          }

          // 2. Surface Snapping (Ground plane & Stacking on obstacles)
          if (surfaceSnapRef.current && !snapMsg) {
            const bottomZ = finalZ - halfHeight;
            // Ground plane snap (z = 0)
            if (Math.abs(bottomZ) < 0.15) {
              finalZ = parseFloat(halfHeight.toFixed(3));
              snapMsg = `Snapped to Ground (z=${finalZ.toFixed(3)}m)`;
            } else {
              // Stacking surface snap on other visible objects
              const otherObjects = projectRef.current?.objects.filter((o) => o.id !== curSelected && o.visible !== false) || [];
              for (const other of otherObjects) {
                let otherHalfH = 0.1;
                let otherHalfX = 0.1;
                let otherHalfY = 0.1;
                if (other.type === 'box') {
                  otherHalfX = other.size[0];
                  otherHalfY = other.size[1];
                  otherHalfH = other.size[2];
                } else if (other.type === 'cylinder') {
                  otherHalfX = other.size[0];
                  otherHalfY = other.size[0];
                  otherHalfH = other.size[1];
                } else if (other.type === 'sphere') {
                  otherHalfX = other.size[0];
                  otherHalfY = other.size[0];
                  otherHalfH = other.size[0];
                }

                const topZ = other.position[2] + otherHalfH;
                const distZ = Math.abs(bottomZ - topZ);
                const overlapX = Math.abs(finalX - other.position[0]) <= (halfX + otherHalfX + 0.08);
                const overlapY = Math.abs(finalY - other.position[1]) <= (halfY + otherHalfY + 0.08);

                if (distZ < 0.20 && overlapX && overlapY) {
                  finalZ = parseFloat((topZ + halfHeight).toFixed(3));
                  if (Math.hypot(finalX - other.position[0], finalY - other.position[1]) < 0.14) {
                    finalX = other.position[0];
                    finalY = other.position[1];
                  }
                  snapMsg = `Stacked on ${other.name} (z=${finalZ.toFixed(3)}m)`;
                  break;
                }
              }
            }
          }

          if (snapMsg) {
            setSnapNotification(snapMsg);
            mesh.position.set(finalX, finalY, finalZ);
            mesh.rotation.set((finalDegX * Math.PI) / 180, (finalDegY * Math.PI) / 180, (finalDegZ * Math.PI) / 180);
            setTimeout(() => setSnapNotification(null), 3000);
          }

          onUpdateObjectRef.current({
            ...obj,
            position: [
              parseFloat(finalX.toFixed(3)),
              parseFloat(finalY.toFixed(3)),
              parseFloat(finalZ.toFixed(3)),
            ],
            rotation: [finalDegX, finalDegY, finalDegZ],
            size: newSize,
          });
        }
      }
    };

    (transformControls as any).addEventListener('dragging-changed', (event: any) => {
      const dragging = Boolean(event.value);
      isDraggingTransformRef.current = dragging;
      if (controlsRef.current) {
        controlsRef.current.enabled = !dragging;
      }
      if (!dragging) {
        commitTransform();
      }
    });

    (transformControls as any).addEventListener('objectChange', () => {
      if (selectedObjectIdRef.current === 'target') {
        const beacon = scene.getObjectByName('target_beacon');
        if (beacon) beacon.position.z = 0;
      }
    });

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
    dirLight.position.set(8, -10, 12);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    scene.add(dirLight);

    // Ground Plane
    const groundGeo = new THREE.PlaneGeometry(100, 100);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x16171b,
      roughness: 0.85,
      metalness: 0.1,
    });
    const groundMesh = new THREE.Mesh(groundGeo, groundMat);
    groundMesh.receiveShadow = true;
    scene.add(groundMesh);

    // Grid Helper rotated into XY plane
    const gridHelper = new THREE.GridHelper(50, 50, 0x3d4452, 0x242831);
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.z = 0.001;
    scene.add(gridHelper);

    // Coordinate Axes at World Origin
    const worldAxes = new THREE.AxesHelper(0.8);
    worldAxes.position.set(0, 0, 0.002);
    scene.add(worldAxes);

    const handleResize = () => {
      if (!canvasContainerRef.current || !rendererRef.current || !cameraRef.current) return;
      const w = canvasContainerRef.current.clientWidth;
      const h = canvasContainerRef.current.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };

    window.addEventListener('resize', handleResize);

    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      if (robotGroupRef.current) {
        robotGroupRef.current.position.lerp(targetRobotPosRef.current, 0.4);
        robotGroupRef.current.quaternion.slerp(targetRobotQuatRef.current, 0.4);

        if (leftWheelRef.current) {
          leftWheelRef.current.rotation.y += leftWheelVelocityRef.current * 0.02;
        }
        if (rightWheelRef.current) {
          rightWheelRef.current.rotation.y += rightWheelVelocityRef.current * 0.02;
        }
      }

      // Rotate active aerial drone rotors
      if (droneRotorsRef.current.length > 0) {
        droneRotorsRef.current.forEach((rotor, idx) => {
          // Counter-rotate alternate rotors for realistic quadcopter dynamics
          const dir = idx % 2 === 0 ? 1 : -1;
          rotor.rotation.z += dir * 0.35;
        });
      }

      if (followRobotRef.current && robotGroupRef.current) {
        const rp = robotGroupRef.current.position;
        controls.target.x += (rp.x - controls.target.x) * 0.1;
        controls.target.y += (rp.y - controls.target.y) * 0.1;
        controls.target.z += (rp.z + 0.1 - controls.target.z) * 0.1;
      }

      // --- Dynamic 3D Trajectory Trail Overlay & Prediction Horizon ---
      if (robotGroupRef.current) {
        const rp = robotGroupRef.current.position;
        const currentSpeed = simStateRef.current?.robot?.linear_speed || 0;
        const trailHistory = trailHistoryRef.current;
        const lastPt = trailHistory[trailHistory.length - 1];

        // Sample if moved > 0.04m
        const distSq = lastPt ? ((rp.x - lastPt.x)**2 + (rp.y - lastPt.y)**2 + (rp.z - lastPt.z)**2) : 999;
        if (distSq > 0.0016) {
          trailHistory.push({ x: rp.x, y: rp.y, z: Math.max(0.015, rp.z), speed: currentSpeed });
          if (trailHistory.length > 500) trailHistory.shift();
        }

        // Render trajectory ribbon segments with velocity-gradient vertex colors
        if (showTrajectoryTrailRef.current && trailHistory.length >= 2) {
          if (!trailMeshRef.current || trailMeshRef.current.parent !== scene) {
            const geo = new THREE.BufferGeometry();
            const mat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
            const line = new THREE.LineSegments(geo, mat);
            line.name = 'robot_trajectory_ribbon';
            scene.add(line);
            trailMeshRef.current = line;
          }

          const positions: number[] = [];
          const colors: number[] = [];

          for (let i = 0; i < trailHistory.length - 1; i++) {
            const p1 = trailHistory[i];
            const p2 = trailHistory[i + 1];
            positions.push(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);

            // Speed color gradient: green (<0.25 m/s) -> yellow (0.25-0.7 m/s) -> red (>0.7 m/s)
            const spd = (p1.speed + p2.speed) * 0.5;
            let r = 0.13, g = 0.77, b = 0.37; // #22c55e green
            if (spd > 0.7) {
              r = 0.94; g = 0.27; b = 0.27; // #ef4444 red
            } else if (spd > 0.25) {
              r = 0.92; g = 0.70; b = 0.03; // #eab308 yellow
            }
            colors.push(r, g, b, r, g, b);
          }

          const geo = trailMeshRef.current.geometry;
          geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
          geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
          geo.attributes.position.needsUpdate = true;
          geo.attributes.color.needsUpdate = true;
          trailMeshRef.current.visible = true;
        } else if (trailMeshRef.current) {
          trailMeshRef.current.visible = false;
        }

        // Forward Prediction Horizon Ribbon (Dashed projected arc for next 2.0s)
        if (showTrajectoryTrailRef.current) {
          const v = simStateRef.current?.robot?.linear_speed || 0;
          const w = simStateRef.current?.robot?.angular_speed || 0;
          const rq = robotGroupRef.current.quaternion;
          const euler = new THREE.Euler().setFromQuaternion(rq, 'ZYX');
          let heading = euler.z;

          const hPositions: THREE.Vector3[] = [];
          let hx = rp.x;
          let hy = rp.y;
          const dt = 0.1;
          for (let step = 0; step <= 20; step++) {
            hPositions.push(new THREE.Vector3(hx, hy, Math.max(0.015, rp.z)));
            hx += v * Math.cos(heading) * dt;
            hy += v * Math.sin(heading) * dt;
            heading += w * dt;
          }

          if (!horizonMeshRef.current || horizonMeshRef.current.parent !== scene) {
            const hGeo = new THREE.BufferGeometry().setFromPoints(hPositions);
            const hMat = new THREE.LineDashedMaterial({ color: 0x38bdf8, dashSize: 0.06, gapSize: 0.04, linewidth: 2 });
            const hLine = new THREE.Line(hGeo, hMat);
            hLine.name = 'robot_prediction_horizon';
            hLine.computeLineDistances();
            scene.add(hLine);
            horizonMeshRef.current = hLine;
          } else {
            horizonMeshRef.current.geometry.setFromPoints(hPositions);
            horizonMeshRef.current.computeLineDistances();
            horizonMeshRef.current.visible = (v > 0.02 || Math.abs(w) > 0.05);
          }
        } else if (horizonMeshRef.current) {
          horizonMeshRef.current.visible = false;
        }
      }

      controls.update();

      const containerEl = canvasContainerRef.current;
      const w = containerEl ? containerEl.clientWidth : width;
      const h = containerEl ? containerEl.clientHeight : height;

      // Primary Scene Render
      renderer.setViewport(0, 0, w, h);
      renderer.setScissorTest(false);
      renderer.render(scene, camera);

      // Onboard Camera Feed PiP Render (when Camera sensor is enabled and HUD open)
      if (
        showCameraHUDRef.current &&
        robotEyeCameraRef.current &&
        projectRef.current?.sensors?.camera?.enabled !== false
      ) {
        const pipW = Math.min(260, Math.floor(w * 0.28));
        const pipH = Math.floor(pipW * 0.65);
        const pipX = w - pipW - 14;
        const pipY = 14;

        renderer.clearDepth();
        renderer.setScissorTest(true);
        renderer.setScissor(pipX, pipY, pipW, pipH);
        renderer.setViewport(pipX, pipY, pipW, pipH);

        robotEyeCameraRef.current.aspect = pipW / pipH;
        robotEyeCameraRef.current.fov = projectRef.current?.sensors?.camera?.fov_deg || 65;
        robotEyeCameraRef.current.updateProjectionMatrix();

        renderer.render(scene, robotEyeCameraRef.current);
        renderer.setScissorTest(false);
      }
    };
    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animId);
      controls.dispose();
      transformControls.dispose();
      transformControlsRef.current = null;
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
      setSceneInstance(null);
    };
  }, []);

  // Update TransformControls with current mode and snapping
  useEffect(() => {
    if (!transformControlsRef.current) return;
    const tc = transformControlsRef.current;
    
    tc.setTranslationSnap(snapTranslate > 0 ? snapTranslate : (null as any));
    tc.setRotationSnap(snapRotate > 0 ? (snapRotate * Math.PI) / 180 : (null as any));

    if (selectedObjectId === 'target') {
      const beacon = sceneInstance?.getObjectByName('target_beacon');
      if (beacon) {
        tc.attach(beacon);
        tc.setMode('translate');
      } else {
        tc.detach();
      }
    } else if (selectedObjectId && selectedObjectId !== 'robot') {
      const mesh = objectMeshesRef.current.get(selectedObjectId);
      if (mesh) {
        tc.attach(mesh);
        tc.setMode(transformMode);
      } else {
        tc.detach();
      }
    } else {
      tc.detach();
    }
  }, [selectedObjectId, transformMode, snapTranslate, snapRotate, sceneInstance, project.objects]);

  // Rebuild scene objects when project changes
  useEffect(() => {
    if (sceneInstance && project) {
      buildSceneObjects(sceneInstance, project);
    }
  }, [sceneInstance, project, buildSceneObjects]);

  // Target Goal Beacon Rendering
  useEffect(() => {
    if (!sceneInstance) return;
    let beaconGroup = sceneInstance.getObjectByName('target_beacon') as THREE.Group;

    if (targetPosition) {
      if (!beaconGroup) {
        beaconGroup = new THREE.Group();
        beaconGroup.name = 'target_beacon';
        (beaconGroup as any).userData = { id: 'target', isTarget: true };

        const ringGeo = new THREE.RingGeometry(0.28, 0.38, 32);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0x10b981,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
        });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.name = 'target_ring';
        beaconGroup.add(ringMesh);

        const discGeo = new THREE.CircleGeometry(0.12, 24);
        const discMat = new THREE.MeshBasicMaterial({
          color: 0x06b6d4,
          transparent: true,
          opacity: 0.9,
        });
        const discMesh = new THREE.Mesh(discGeo, discMat);
        beaconGroup.add(discMesh);

        const pillarGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.6, 16);
        const pillarMat = new THREE.MeshBasicMaterial({
          color: 0x10b981,
          transparent: true,
          opacity: 0.7,
        });
        const pillarMesh = new THREE.Mesh(pillarGeo, pillarMat);
        pillarMesh.position.set(0, 0, 0.3);
        pillarMesh.rotation.x = Math.PI / 2;
        beaconGroup.add(pillarMesh);

        sceneInstance.add(beaconGroup);
      }
      beaconGroup.position.set(targetPosition[0], targetPosition[1], 0);
      beaconGroup.visible = true;
    } else if (beaconGroup) {
      beaconGroup.visible = false;
    }
  }, [sceneInstance, targetPosition]);

  // Trajectory Trail
  useEffect(() => {
    if (!sceneInstance) return;
    const oldTrail = sceneInstance.getObjectByName('trajectory_trail');
    if (oldTrail) {
      sceneInstance.remove(oldTrail);
      (oldTrail as any).geometry?.dispose();
      (oldTrail as any).material?.dispose();
    }

    if (trajectoryTrail && trajectoryTrail.length > 1) {
      const points = trajectoryTrail.map((p) => new THREE.Vector3(p[0], p[1], p[2] || 0.02));
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x38bdf8,
        linewidth: 2,
        transparent: true,
        opacity: 0.85,
      });
      const lineMesh = new THREE.Line(lineGeo, lineMat);
      lineMesh.name = 'trajectory_trail';
      sceneInstance.add(lineMesh);
    }
  }, [sceneInstance, trajectoryTrail]);

  // Per-frame physics synchronization from simState
  useEffect(() => {
    if (!simState) return;

    if (simState.robot) {
      const [rx, ry, rz] = simState.robot.position;
      const [qw, qx, qy, qz] = simState.robot.quaternion;

      targetRobotPosRef.current.set(rx, ry, rz);
      const qlen = Math.hypot(qw, qx, qy, qz);
      if (qlen > 0.001) {
        targetRobotQuatRef.current.set(qx / qlen, qy / qlen, qz / qlen, qw / qlen);
      }

      leftWheelVelocityRef.current = simState.robot.left_wheel_velocity || 0;
      rightWheelVelocityRef.current = simState.robot.right_wheel_velocity || 0;
    }

    if (simState.objects) {
      const curSelected = selectedObjectIdRef.current;
      const isDragging = isDraggingTransformRef.current;

      Object.entries(simState.objects).forEach(([id, objState]) => {
        if (isDragging && id === curSelected) return;
        const mesh = objectMeshesRef.current.get(id);
        if (mesh && objState.position && objState.quaternion) {
          const [ox, oy, oz] = objState.position;
          const [qw, qx, qy, qz] = objState.quaternion;
          mesh.position.set(ox, oy, oz);
          const qlen = Math.hypot(qw, qx, qy, qz);
          if (qlen > 0.001) {
            mesh.quaternion.set(qx / qlen, qy / qlen, qz / qlen, qw / qlen);
          }
        }
      });
    }

    // Generic Robot Joint Positions Synchronization
    if (simState.joint_positions && jointMotionNodesRef.current.size > 0) {
      Object.entries(simState.joint_positions).forEach(([jName, val]) => {
        const node = jointMotionNodesRef.current.get(jName);
        if (node) {
          if (node.type === 'prismatic') {
            node.group.position.copy(node.axis).multiplyScalar(val);
          } else {
            node.group.quaternion.setFromAxisAngle(node.axis, val);
          }
        }
      });
    }

    // Live End-Effector Pose Extraction from active 3D model
    if (endEffectorMarkerRef.current) {
      endEffectorMarkerRef.current.updateMatrixWorld(true);
      const wp = new THREE.Vector3();
      const wq = new THREE.Quaternion();
      endEffectorMarkerRef.current.getWorldPosition(wp);
      endEffectorMarkerRef.current.getWorldQuaternion(wq);
      const euler = new THREE.Euler().setFromQuaternion(wq, 'ZYX');
      const lName = (endEffectorMarkerRef.current.userData as any)?.linkName || 'EE Tip';
      setLiveEePose({
        linkName: lName,
        x: wp.x,
        y: wp.y,
        z: wp.z,
        roll: (euler.x * 180) / Math.PI,
        pitch: (euler.y * 180) / Math.PI,
        yaw: (euler.z * 180) / Math.PI,
      });
    }

    // Real LiDAR Rays Visualization (Respects Sensors Enabled)
    if (sceneInstance && simState.lidar_readings && simState.lidar_readings.length > 0 && project.sensors?.lidar?.enabled !== false) {
      let lidarGroup = sceneInstance.getObjectByName('lidar_rays') as THREE.LineSegments;
      if (!lidarGroup) {
        const mat = new THREE.LineBasicMaterial({
          color: 0x06b6d4,
          transparent: true,
          opacity: 0.55,
        });
        lidarGroup = new THREE.LineSegments(new THREE.BufferGeometry(), mat);
        lidarGroup.name = 'lidar_rays';
        sceneInstance.add(lidarGroup);
        lidarLinesRef.current = lidarGroup;
      }

      const rPos = simState.robot.position;
      const rQuat = simState.robot.quaternion;
      const siny_cosp = 2.0 * (rQuat[0] * rQuat[3] + rQuat[1] * rQuat[2]);
      const cosy_cosp = 1.0 - 2.0 * (rQuat[2] * rQuat[2] + rQuat[3] * rQuat[3]);
      const yaw = Math.atan2(siny_cosp, cosy_cosp);

      const mount = project.sensors?.lidar?.mount_pos || [0.1, 0, 0.09];
      const cosY = Math.cos(yaw);
      const sinY = Math.sin(yaw);
      const originX = rPos[0] + mount[0] * cosY - mount[1] * sinY;
      const originY = rPos[1] + mount[0] * sinY + mount[1] * cosY;
      const originZ = rPos[2] + mount[2];

      const pts: THREE.Vector3[] = [];
      const angles = simState.lidar_angles || [];
      simState.lidar_readings.forEach((dist, idx) => {
        const beamAngleRad = ((angles[idx] ?? 0) * Math.PI) / 180;
        const beamYaw = yaw + beamAngleRad;
        const hitX = originX + dist * Math.cos(beamYaw);
        const hitY = originY + dist * Math.sin(beamYaw);
        const hitZ = originZ;
        pts.push(new THREE.Vector3(originX, originY, originZ));
        pts.push(new THREE.Vector3(hitX, hitY, hitZ));
      });
      lidarGroup.geometry.setFromPoints(pts);
      lidarGroup.visible = true;
    } else if (sceneInstance) {
      const oldLidar = sceneInstance.getObjectByName('lidar_rays');
      if (oldLidar) oldLidar.visible = false;
    }

    // Active Contact Points Visualization
    if (sceneInstance) {
      let contactGrp = sceneInstance.getObjectByName('contact_markers') as THREE.Group;
      if (!contactGrp) {
        contactGrp = new THREE.Group();
        contactGrp.name = 'contact_markers';
        sceneInstance.add(contactGrp);
        contactMarkersGroupRef.current = contactGrp;
      }
      while (contactGrp.children.length > 0) {
        const c = contactGrp.children[0] as THREE.Mesh;
        contactGrp.remove(c);
        c.geometry?.dispose();
        if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose());
        else (c.material as any)?.dispose();
      }

      if (simState.contacts && simState.contacts.length > 0) {
        const sphereGeo = new THREE.SphereGeometry(0.025, 10, 10);
        const sphereMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
        simState.contacts.slice(0, 10).forEach((con) => {
          const marker = new THREE.Mesh(sphereGeo, sphereMat);
          marker.position.set(con.position[0], con.position[1], con.position[2]);
          contactGrp.add(marker);
        });
      }
    }
  }, [simState, sceneInstance, project]);

  // Highlight selection
  useEffect(() => {
    if (robotGroupRef.current) {
      const chassis = robotGroupRef.current.children[0] as THREE.Mesh;
      if (chassis && chassis.material) {
        const mat = chassis.material as THREE.MeshStandardMaterial;
        mat.emissive.set(selectedObjectId === 'robot' ? 0x3b82f6 : 0x000000);
        mat.emissiveIntensity = selectedObjectId === 'robot' ? 0.5 : 0.0;
      }
    }

    objectMeshesRef.current.forEach((mesh, id) => {
      mesh.traverse((child: any) => {
        if (child.isMesh && child.material && 'emissive' in child.material) {
          const mat = child.material as THREE.MeshStandardMaterial;
          mat.emissive.set(id === selectedObjectId ? 0xffa500 : 0x000000);
          mat.emissiveIntensity = id === selectedObjectId ? 0.5 : 0.0;
        }
      });
    });
  }, [selectedObjectId]);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    pointerDownPos.current = { x: e.clientX, y: e.clientY };
    (document.activeElement as HTMLElement)?.blur();
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (transformControlsRef.current && (transformControlsRef.current as any).dragging) return;

    const dx = e.clientX - pointerDownPos.current.x;
    const dy = e.clientY - pointerDownPos.current.y;
    if (Math.hypot(dx, dy) > 5) return;

    if (!canvasContainerRef.current || !cameraRef.current || !sceneRef.current) return;
    const rect = canvasContainerRef.current.getBoundingClientRect();
    mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);

    const interactableObjects: THREE.Object3D[] = [];
    if (robotGroupRef.current) {
      interactableObjects.push(robotGroupRef.current);
    }
    if (genericRobotGroupRef.current) {
      interactableObjects.push(genericRobotGroupRef.current);
    }
    const beacon = sceneRef.current.getObjectByName('target_beacon');
    if (beacon) {
      interactableObjects.push(beacon);
    }
    if (obstaclesGroupRef.current) {
      interactableObjects.push(obstaclesGroupRef.current);
    }

    const intersects = raycasterRef.current.intersectObjects(interactableObjects, true);

    if (intersects.length > 0) {
      let hitObj: THREE.Object3D | null = intersects[0].object;
      let matchedId: string | null = null;

      while (hitObj && hitObj !== sceneRef.current) {
        if (hitObj.name === 'target_beacon' || (hitObj as any).userData?.isTarget) {
          matchedId = 'target';
          break;
        }
        if (hitObj.name === 'generic_robot_group' || (hitObj as any).userData?.id === 'generic_robot') {
          matchedId = 'generic_robot';
          break;
        }
        if (hitObj.name === 'robot' || (hitObj as any).userData?.isRobot) {
          matchedId = 'robot';
          break;
        }
        if ((hitObj as any).userData?.isObstacle && (hitObj as any).userData?.id) {
          matchedId = (hitObj as any).userData.id;
          break;
        }
        hitObj = hitObj.parent;
      }

      onSelectObject(matchedId);
    } else {
      onSelectObject(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDraggingFile) setIsDraggingFile(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(false);

    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    const fileName = file.name.toLowerCase();
    try {
      const text = await file.text();
      if (fileName.endsWith('.urdf')) {
        setImportNotification(`Importing URDF: ${file.name}...`);
        const res = await importURDF(text, true);
        if (res.status !== 'error') {
          setImportNotification(`Successfully imported URDF: ${res.robot?.name || file.name}`);
        } else {
          setImportNotification(`Import error: ${res.error || 'Invalid URDF'}`);
        }
      } else if (fileName.endsWith('.xml') || fileName.endsWith('.mjcf')) {
        setImportNotification(`Importing MJCF: ${file.name}...`);
        const res = await importMJCF(text, true);
        if (res.status !== 'error') {
          setImportNotification(`Successfully imported MJCF: ${res.robot?.name || file.name}`);
        } else {
          setImportNotification(`Import error: ${res.error || 'Invalid MJCF'}`);
        }
      } else {
        setImportNotification('Unsupported format. Please drop a .urdf or .xml/.mjcf robot file.');
      }
      setTimeout(() => setImportNotification(null), 4000);
    } catch (err: any) {
      setImportNotification(`Import failed: ${err.message}`);
      setTimeout(() => setImportNotification(null), 4000);
    }
  };

  const isW = activeKeys['KeyW'] || activeKeys['ArrowUp'];
  const isS = activeKeys['KeyS'] || activeKeys['ArrowDown'];
  const isA = activeKeys['KeyA'] || activeKeys['ArrowLeft'];
  const isD = activeKeys['KeyD'] || activeKeys['ArrowRight'];

  const rPose = simState.robot;
  const isCameraEnabled = project.sensors?.camera?.enabled !== false;

  return (
    <div
      className="rf-viewport-container"
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}
    >
      {/* File Drag-and-Drop Visual Dropzone */}
      {isDraggingFile && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundColor: 'rgba(6, 182, 212, 0.25)',
          border: '3px dashed #06b6d4',
          borderRadius: 8,
          zIndex: 50,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 12,
          pointerEvents: 'none',
          backdropFilter: 'blur(4px)'
        }}>
          <FileUp size={48} color="#06b6d4" />
          <div style={{ fontSize: 18, fontWeight: 700, color: '#fff' }}>
            Drop Robot Model to Instantiate
          </div>
          <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.85)' }}>
            Supports URDF (.urdf) &amp; native MuJoCo MJCF (.xml / .mjcf)
          </div>
        </div>
      )}

      {/* Import Notification Banner */}
      {importNotification && (
        <div style={{
          position: 'absolute',
          top: 55,
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          border: '1px solid #06b6d4',
          borderRadius: 6,
          padding: '8px 16px',
          color: '#06b6d4',
          fontSize: 12,
          fontWeight: 600,
          zIndex: 60,
          boxShadow: '0 8px 24px rgba(0,0,0,0.6)'
        }}>
          {importNotification}
        </div>
      )}

      {/* Dedicated Three.js canvas container with no React children */}
      <div
        ref={canvasContainerRef}
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0 }}
      />

      {/* Transform Gizmo Selector Bar */}
      {selectedObjectId && (
        <div style={{
          position: 'absolute',
          top: 12,
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          backgroundColor: 'rgba(20, 22, 28, 0.92)',
          backdropFilter: 'blur(8px)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 8,
          padding: '6px 12px',
          zIndex: 10,
          boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
          fontSize: 12
        }}>
          <span style={{ fontWeight: 600, color: selectedObjectId === 'target' ? 'var(--accent-green)' : 'var(--accent-orange)', display: 'flex', alignItems: 'center', gap: 5 }}>
            <span>{selectedObjectId === 'target' ? 'Target Waypoint' : 'Object'}:</span>
            <span style={{ color: '#fff' }}>
              {selectedObjectId === 'target'
                ? `[${targetPosition ? `${targetPosition[0].toFixed(2)}, ${targetPosition[1].toFixed(2)}` : '0, 0'}]`
                : (project.objects.find(o => o.id === selectedObjectId)?.name || selectedObjectId)}
            </span>
          </span>
          <div style={{ width: 1, height: 18, backgroundColor: 'var(--border-subtle)', margin: '0 4px' }} />
          <button
            className={`rf-btn ${transformMode === 'translate' ? 'rf-btn-primary' : ''}`}
            style={{ padding: '4px 10px', fontSize: 11 }}
            onClick={(e) => { e.stopPropagation(); setTransformMode('translate'); }}
            title="Translate / Move in 3D"
          >
            <Move size={13} />
            <span>Move</span>
          </button>
          {selectedObjectId !== 'target' && (
            <>
              <button
                className={`rf-btn ${transformMode === 'rotate' ? 'rf-btn-primary' : ''}`}
                style={{ padding: '4px 10px', fontSize: 11 }}
                onClick={(e) => { e.stopPropagation(); setTransformMode('rotate'); }}
                title="Rotate in 3D"
              >
                <RotateCw size={13} />
                <span>Rotate</span>
              </button>
              <button
                className={`rf-btn ${transformMode === 'scale' ? 'rf-btn-primary' : ''}`}
                style={{ padding: '4px 10px', fontSize: 11 }}
                onClick={(e) => { e.stopPropagation(); setTransformMode('scale'); }}
                title="Scale Object in 3D"
              >
                <Maximize2 size={13} />
                <span>Scale</span>
              </button>
            </>
          )}

          {/* Snapping controls */}
          <div style={{ width: 1, height: 18, backgroundColor: 'var(--border-subtle)', margin: '0 4px' }} />
          <button
            className={`rf-btn ${snapTranslate > 0 ? 'rf-btn-primary' : ''}`}
            style={{ padding: '4px 8px', fontSize: 11 }}
            onClick={(e) => {
              e.stopPropagation();
              setSnapTranslate((prev) => (prev === 0 ? 0.1 : prev === 0.1 ? 0.5 : prev === 0.5 ? 1.0 : 0));
            }}
            title="Grid Translation Snap (0.1m, 0.5m, 1.0m)"
          >
            <Magnet size={12} />
            <span>{snapTranslate === 0 ? 'Grid Off' : `${snapTranslate}m`}</span>
          </button>

          <button
            className={`rf-btn ${surfaceSnap ? 'rf-btn-primary' : ''}`}
            style={{ padding: '4px 8px', fontSize: 11 }}
            onClick={(e) => {
              e.stopPropagation();
              setSurfaceSnap(!surfaceSnap);
            }}
            title="Surface Stacking Snap: Snaps object flush to ground and obstacle top surfaces"
          >
            <Layers size={12} />
            <span>Surface: {surfaceSnap ? 'ON' : 'OFF'}</span>
          </button>

          <button
            className={`rf-btn ${attachmentSnap ? 'rf-btn-primary' : ''}`}
            style={{ padding: '4px 8px', fontSize: 11 }}
            onClick={(e) => {
              e.stopPropagation();
              setAttachmentSnap(!attachmentSnap);
            }}
            title="Robot Assembly & Flange Attachment Snap: Snaps object to robot chassis mount or EE flange"
          >
            <GitFork size={12} />
            <span>Attach: {attachmentSnap ? 'ON' : 'OFF'}</span>
          </button>

          <button
            className="rf-btn"
            style={{ padding: '4px 6px', fontSize: 12, color: 'var(--text-muted)' }}
            onClick={(e) => { e.stopPropagation(); onSelectObject(null); }}
            title="Deselect"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Snap Notification Toast Indicator */}
      {snapNotification && (
        <div
          style={{
            position: 'absolute',
            bottom: 24,
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: 'rgba(15, 23, 42, 0.90)',
            backdropFilter: 'blur(8px)',
            border: '1px solid var(--accent-blue)',
            borderRadius: 8,
            padding: '6px 14px',
            color: '#38bdf8',
            fontSize: 12,
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
            zIndex: 10,
            animation: 'fadeIn 0.2s ease-in-out'
          }}
        >
          <Magnet size={14} color="#38bdf8" />
          <span>{snapNotification}</span>
        </div>
      )}

      {/* Viewport Overlay Controls (Top Right): Camera Views & Modes */}
      <div style={{ position: 'absolute', top: 12, right: 12, display: 'flex', gap: 6, zIndex: 5 }}>
        {/* Auto-Drive Autonomous Waypoint-Seeking Toggle */}
        <button
          className={`rf-btn ${autoDriveEnabled ? 'rf-btn-success' : ''}`}
          style={{
            fontSize: 11,
            fontWeight: 600,
            padding: '4px 10px',
            backgroundColor: autoDriveEnabled ? 'rgba(16, 185, 129, 0.25)' : 'var(--bg-panel)',
            borderColor: autoDriveEnabled ? 'var(--accent-green)' : 'var(--border-subtle)',
            color: autoDriveEnabled ? '#10b981' : 'var(--text-secondary)'
          }}
          title="Toggle Autonomous Waypoint Navigation (Robot navigates to target)"
          onClick={(e) => {
            e.stopPropagation();
            onToggleAutoDrive?.();
          }}
        >
          <Zap size={13} color={autoDriveEnabled ? 'var(--accent-green)' : 'currentColor'} />
          <span>Auto-Drive: {autoDriveEnabled ? 'ON' : 'OFF'}</span>
        </button>

        {/* Active RL Policy Live Badge */}
        {simState?.rl_policy_active && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              fontSize: 11,
              padding: '4px 10px',
              borderRadius: 6,
              background: 'rgba(168, 85, 247, 0.25)',
              border: '1px solid rgba(168, 85, 247, 0.6)',
              color: '#c084fc',
              fontWeight: 600,
              boxShadow: '0 0 10px rgba(168, 85, 247, 0.25)'
            }}
            title={`Active PPO Neural Network Policy: ${simState.active_rl_policy_name || 'Policy Active'}`}
          >
            <Cpu size={13} color="#c084fc" />
            <span>RL Policy: {simState.active_rl_policy_name || 'Active'}</span>
          </div>
        )}

        {/* Live Isaac Lab Multi-Robot RL Training Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button
            className="rf-btn"
            style={{
              fontSize: 11,
              padding: '3px 8px',
              borderRadius: 6,
              backgroundColor: simState?.rl_training_active ? 'rgba(239, 68, 68, 0.25)' : 'rgba(56, 189, 248, 0.2)',
              border: simState?.rl_training_active ? '1px solid rgba(239, 68, 68, 0.6)' : '1px solid rgba(56, 189, 248, 0.6)',
              color: simState?.rl_training_active ? '#fca5a5' : '#38bdf8',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 4
            }}
            onClick={async (e) => {
              e.stopPropagation();
              try {
                await fetch('http://127.0.0.1:8000/api/rl/train-drone-live', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ drone_identifier: 'drone', target_altitude: 1.5, total_episodes: 4 })
                });
              } catch (err) {
                console.error('Failed to trigger live drone training:', err);
              }
            }}
            title="Train Drone Hover live in 3D scene (Isaac Lab curriculum: fail, retry, converge to hover, store PPO model)"
          >
            <Sparkles size={12} color={simState?.rl_training_active ? '#fca5a5' : '#38bdf8'} />
            <span>{simState?.rl_training_active ? 'Training...' : '🚁 Drone RL'}</span>
          </button>

          <button
            className="rf-btn"
            style={{
              fontSize: 11,
              padding: '3px 8px',
              borderRadius: 6,
              backgroundColor: 'rgba(168, 85, 247, 0.18)',
              border: '1px solid rgba(168, 85, 247, 0.5)',
              color: '#c084fc',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 4
            }}
            onClick={async (e) => {
              e.stopPropagation();
              try {
                await fetch('http://127.0.0.1:8000/api/rl/train-arm-live', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ arm_identifier: 'arm', total_episodes: 4 })
                });
              } catch (err) {
                console.error('Failed to trigger live arm training:', err);
              }
            }}
            title="Train Robotic Arm Pick & Place live in 3D scene (overshoot, slip, align, lift, store PPO model)"
          >
            <span>🦾 Arm RL</span>
          </button>

          <button
            className="rf-btn"
            style={{
              fontSize: 11,
              padding: '3px 8px',
              borderRadius: 6,
              backgroundColor: 'rgba(34, 197, 94, 0.18)',
              border: '1px solid rgba(34, 197, 94, 0.5)',
              color: '#4ade80',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 4
            }}
            onClick={async (e) => {
              e.stopPropagation();
              try {
                await fetch('http://127.0.0.1:8000/api/rl/train-rover-live', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ robot_identifier: 'robot', total_episodes: 4 })
                });
              } catch (err) {
                console.error('Failed to trigger live rover training:', err);
              }
            }}
            title="Train Ground Rover Navigation live in 3D scene (crash, oversteer, LiDAR avoidance, goal beacon, store PPO model)"
          >
            <span>🚗 Rover RL</span>
          </button>
        </div>

        {/* 3D Trajectory Ribbon & Prediction Horizon Controls */}
        <div style={{ display: 'flex', alignItems: 'center', background: 'var(--bg-panel)', padding: 2, borderRadius: 6, border: '1px solid var(--border-subtle)', gap: 2 }}>
          <button
            className={`rf-btn ${showTrajectoryTrail ? 'rf-btn-primary' : ''}`}
            style={{ fontSize: 11, padding: '3px 8px' }}
            onClick={(e) => {
              e.stopPropagation();
              setShowTrajectoryTrail(!showTrajectoryTrail);
            }}
            title="Toggle Dynamic 3D Trajectory Ribbon & Speed Heatmap"
          >
            <Route size={12} />
            <span>Trail: {showTrajectoryTrail ? 'ON' : 'OFF'}</span>
          </button>
          <button
            className="rf-btn"
            style={{ fontSize: 11, padding: '3px 6px' }}
            onClick={(e) => {
              e.stopPropagation();
              trailHistoryRef.current = [];
              if (trailMeshRef.current) trailMeshRef.current.visible = false;
              if (horizonMeshRef.current) horizonMeshRef.current.visible = false;
            }}
            title="Clear Trajectory Trail History"
          >
            <Trash2 size={12} />
          </button>
        </div>

        {/* Camera Views Preset */}
        <div style={{ display: 'flex', background: 'var(--bg-panel)', padding: 2, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
          <button
            className="rf-btn"
            style={{ fontSize: 11, padding: '3px 8px' }}
            onClick={() => setCameraPreset('top')}
            title="Top-Down Camera (Z axis)"
          >
            Top
          </button>
          <button
            className="rf-btn"
            style={{ fontSize: 11, padding: '3px 8px' }}
            onClick={() => setCameraPreset('front')}
            title="Front Camera (X axis)"
          >
            Front
          </button>
          <button
            className="rf-btn"
            style={{ fontSize: 11, padding: '3px 8px' }}
            onClick={() => setCameraPreset('side')}
            title="Side Camera (Y axis)"
          >
            Side
          </button>
          <button
            className="rf-btn"
            style={{ fontSize: 11, padding: '3px 8px' }}
            onClick={() => setCameraPreset('iso')}
            title="3D Orbit Perspective"
          >
            3D
          </button>
        </div>

        <button
          className="rf-btn"
          style={{
            fontSize: 11,
            backgroundColor: followRobot ? 'rgba(59, 130, 246, 0.25)' : 'var(--bg-panel)',
            borderColor: followRobot ? 'var(--accent-blue)' : 'var(--border-subtle)',
            color: followRobot ? '#fff' : 'var(--text-secondary)',
          }}
          title="Keep Camera Locked onto Robot (Toggle Follow)"
          onClick={(e) => {
            e.stopPropagation();
            setFollowRobot(!followRobot);
          }}
        >
          <Crosshair size={13} color={followRobot ? 'var(--accent-blue)' : 'currentColor'} />
          <span>Follow: {followRobot ? 'ON' : 'OFF'}</span>
        </button>

        <button
          className="rf-btn rf-btn-icon"
          title="Focus Camera on Robot"
          onClick={(e) => {
            e.stopPropagation();
            focusOnRobot();
          }}
        >
          <Camera size={14} />
        </button>
      </div>

      {/* Kinematic Studio Visualizations Overlay Bar */}
      {project.generic_robot && (
        <div
          style={{
            position: 'absolute',
            top: 50,
            right: 12,
            display: 'flex',
            gap: 4,
            alignItems: 'center',
            backgroundColor: 'rgba(20, 22, 28, 0.90)',
            backdropFilter: 'blur(6px)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 6,
            padding: '3px 8px',
            zIndex: 5,
            fontSize: 11,
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
          }}
        >
          <span
            style={{
              color: 'var(--text-muted)',
              fontWeight: 600,
              marginRight: 4,
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 10,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            <GitFork size={12} color="var(--accent-blue)" />
            <span>Kinematics:</span>
          </span>
          <button
            className={`rf-btn ${showKinematicFrames ? 'rf-btn-primary' : ''}`}
            style={{ fontSize: 10, padding: '2px 7px' }}
            onClick={(e) => {
              e.stopPropagation();
              setShowKinematicFrames(!showKinematicFrames);
            }}
            title="Toggle Link & Joint Coordinate Frames (AxesHelper)"
          >
            Frames
          </button>
          <button
            className={`rf-btn ${showJointAxes ? 'rf-btn-primary' : ''}`}
            style={{ fontSize: 10, padding: '2px 7px' }}
            onClick={(e) => {
              e.stopPropagation();
              setShowJointAxes(!showJointAxes);
            }}
            title="Toggle Joint Motion Axes (Arrow Vectors)"
          >
            Axes
          </button>
          <button
            className={`rf-btn ${showJointLimits ? 'rf-btn-primary' : ''}`}
            style={{ fontSize: 10, padding: '2px 7px' }}
            onClick={(e) => {
              e.stopPropagation();
              setShowJointLimits(!showJointLimits);
            }}
            title="Toggle Joint Angular & Linear Limit Indicators"
          >
            Limits
          </button>
          <button
            className={`rf-btn ${showLinkHierarchy ? 'rf-btn-primary' : ''}`}
            style={{ fontSize: 10, padding: '2px 7px' }}
            onClick={(e) => {
              e.stopPropagation();
              setShowLinkHierarchy(!showLinkHierarchy);
            }}
            title="Toggle Kinematic Tree Hierarchy Skeleton Lines"
          >
            Hierarchy
          </button>
          <button
            className={`rf-btn ${showEndEffectorMarker ? 'rf-btn-primary' : ''}`}
            style={{ fontSize: 10, padding: '2px 7px' }}
            onClick={(e) => {
              e.stopPropagation();
              setShowEndEffectorMarker(!showEndEffectorMarker);
            }}
            title="Toggle End-Effector Tip Pose Marker"
          >
            EE Tip
          </button>
        </div>
      )}

      {/* Real-time End-Effector Pose Readout Card */}
      {project.generic_robot && liveEePose && (
        <div
          style={{
            position: 'absolute',
            top: 88,
            right: 12,
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            backgroundColor: 'rgba(20, 22, 28, 0.92)',
            backdropFilter: 'blur(8px)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 6,
            padding: '7px 11px',
            zIndex: 5,
            fontSize: 11,
            fontFamily: 'var(--font-mono)',
            boxShadow: '0 4px 14px rgba(0,0,0,0.45)',
            pointerEvents: 'none',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
            <span style={{ color: 'var(--accent-green)', fontWeight: 600, fontSize: 10, display: 'flex', alignItems: 'center', gap: 4 }}>
              <Compass size={11} /> END-EFFECTOR POSE
            </span>
            <span style={{ color: 'var(--accent-blue)', fontSize: 10, fontWeight: 600 }}>
              {liveEePose.linkName}
            </span>
          </div>
          <div style={{ display: 'flex', gap: 10, color: 'var(--text-primary)' }}>
            <span>X: {liveEePose.x.toFixed(3)}m</span>
            <span>Y: {liveEePose.y.toFixed(3)}m</span>
            <span>Z: {liveEePose.z.toFixed(3)}m</span>
          </div>
          <div style={{ display: 'flex', gap: 8, color: 'var(--text-secondary)', fontSize: 10 }}>
            <span>Roll: {liveEePose.roll.toFixed(1)}°</span>
            <span>Pitch: {liveEePose.pitch.toFixed(1)}°</span>
            <span>Yaw: {liveEePose.yaw.toFixed(1)}°</span>
          </div>
        </div>
      )}

      {/* Real-time Telemetry HUD (Top Left) */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          backgroundColor: 'rgba(20, 22, 28, 0.88)',
          backdropFilter: 'blur(6px)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 8,
          padding: '8px 14px',
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
          pointerEvents: 'none',
          fontSize: 11,
          fontFamily: 'var(--font-mono)',
          boxShadow: '0 4px 16px rgba(0,0,0,0.4)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-secondary)' }}>
          <Compass size={13} color="var(--accent-blue)" />
          <span style={{ fontWeight: 600 }}>ROBOT TELEMETRY</span>
        </div>
        <div style={{ display: 'flex', gap: 14, color: 'var(--text-primary)' }}>
          <span>X: {rPose.position[0].toFixed(3)}m</span>
          <span>Y: {rPose.position[1].toFixed(3)}m</span>
          <span>Z: {rPose.position[2].toFixed(3)}m</span>
        </div>
        <div style={{ display: 'flex', gap: 14, color: 'var(--text-secondary)' }}>
          <span>Speed: {rPose.linear_speed.toFixed(2)} m/s</span>
          <span>Turn: {rPose.angular_speed.toFixed(2)} rad/s</span>
        </div>
        {simState.contacts && simState.contacts.length > 0 && (
          <div style={{ color: '#ef4444', fontWeight: 600 }}>
            Contacts: {simState.contacts.length}
          </div>
        )}
      </div>

      {/* Picture-in-Picture Robot Eye Vision Camera Feed HUD (Bottom Right) */}
      {isCameraEnabled && (
        <div
          style={{
            position: 'absolute',
            bottom: 14,
            right: 14,
            width: 260,
            backgroundColor: 'rgba(15, 17, 23, 0.92)',
            backdropFilter: 'blur(8px)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 8,
            overflow: 'hidden',
            boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
            zIndex: 6,
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          {/* Header */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '5px 10px',
            backgroundColor: 'rgba(26, 30, 40, 0.95)',
            borderBottom: '1px solid var(--border-subtle)',
            fontSize: 10,
            fontWeight: 600,
            color: 'var(--text-secondary)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#38bdf8' }}>
              <Video size={12} />
              <span>ROBOT EYE (RGB CAMERA)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 9, padding: '1px 4px', backgroundColor: 'rgba(56, 189, 248, 0.2)', color: '#38bdf8', borderRadius: 3 }}>
                FOV {project.sensors?.camera?.fov_deg || 65}°
              </span>
              <button
                className="rf-btn"
                style={{ padding: '1px 4px', fontSize: 10, border: 'none', background: 'transparent' }}
                onClick={(e) => {
                  e.stopPropagation();
                  setShowCameraHUD(!showCameraHUD);
                }}
                title={showCameraHUD ? 'Minimize Camera Feed' : 'Show Camera Feed'}
              >
                {showCameraHUD ? <EyeOff size={12} /> : <Eye size={12} />}
              </button>
            </div>
          </div>

          {/* Viewport placeholder for scissor-test rendering */}
          {showCameraHUD && (
            <div
              style={{
                width: 260,
                height: 169,
                position: 'relative',
                pointerEvents: 'none',
              }}
            >
              {/* Camera Crosshair & HUD Overlay */}
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: 24,
                height: 24,
                border: '1px dashed rgba(56, 189, 248, 0.4)',
                borderRadius: '50%'
              }} />
              <div style={{
                position: 'absolute',
                bottom: 4,
                left: 6,
                fontSize: 9,
                fontFamily: 'var(--font-mono)',
                color: 'rgba(255,255,255,0.7)',
                backgroundColor: 'rgba(0,0,0,0.5)',
                padding: '1px 4px',
                borderRadius: 2
              }}>
                LIVE SENSOR
              </div>
            </div>
          )}
        </div>
      )}

      {/* Interactive On-Screen D-Pad Controller (Bottom Left) */}
      <div
        style={{
          position: 'absolute',
          bottom: 14,
          left: 14,
          backgroundColor: 'rgba(20, 22, 28, 0.92)',
          backdropFilter: 'blur(6px)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 8,
          padding: '8px 12px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 4,
          zIndex: 5,
          boxShadow: '0 4px 16px rgba(0,0,0,0.5)'
        }}
      >
        <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 2 }}>
          Manual Drive (WASD)
        </div>

        <button
          className="rf-btn rf-btn-icon"
          style={{
            backgroundColor: isW ? 'var(--accent-blue)' : 'var(--bg-panel)',
            borderColor: isW ? 'var(--accent-blue)' : 'var(--border-subtle)',
            color: isW ? '#fff' : 'var(--text-primary)',
          }}
          title="Forward (W or Up Arrow)"
          onPointerDown={() => onManualTeleop?.(1.0, 0.0)}
        >
          <ArrowUp size={16} />
        </button>

        <div style={{ display: 'flex', gap: 6 }}>
          <button
            className="rf-btn rf-btn-icon"
            style={{
              backgroundColor: isA ? 'var(--accent-blue)' : 'var(--bg-panel)',
              borderColor: isA ? 'var(--accent-blue)' : 'var(--border-subtle)',
              color: isA ? '#fff' : 'var(--text-primary)',
            }}
            title="Turn Left (A or Left Arrow)"
            onPointerDown={() => onManualTeleop?.(0.0, 1.5)}
          >
            <ArrowLeft size={16} />
          </button>
          <button
            className="rf-btn rf-btn-icon"
            style={{
              backgroundColor: isS ? 'var(--accent-blue)' : 'var(--bg-panel)',
              borderColor: isS ? 'var(--accent-blue)' : 'var(--border-subtle)',
              color: isS ? '#fff' : 'var(--text-primary)',
            }}
            title="Reverse (S or Down Arrow)"
            onPointerDown={() => onManualTeleop?.(-1.0, 0.0)}
          >
            <ArrowDown size={16} />
          </button>
          <button
            className="rf-btn rf-btn-icon"
            style={{
              backgroundColor: isD ? 'var(--accent-blue)' : 'var(--bg-panel)',
              borderColor: isD ? 'var(--accent-blue)' : 'var(--border-subtle)',
              color: isD ? '#fff' : 'var(--text-primary)',
            }}
            title="Turn Right (D or Right Arrow)"
            onPointerDown={() => onManualTeleop?.(0.0, -1.5)}
          >
            <ArrowRight size={16} />
          </button>
        </div>
      </div>

      {/* Isaac Lab Live RL Training HUD (Center-Top Viewport Overlay) */}
      {(simState?.rl_training_active || (simState?.rl_training_status && simState.rl_training_status.includes('PPO') && !simState.rl_training_status.includes('RESET'))) && (
        <div
          style={{
            position: 'absolute',
            top: 52,
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            flexDirection: 'column',
            gap: 7,
            backgroundColor: 'rgba(15, 17, 23, 0.94)',
            backdropFilter: 'blur(10px)',
            border: simState?.rl_training_active ? '1px solid rgba(56, 189, 248, 0.6)' : '1px solid rgba(34, 197, 94, 0.6)',
            borderRadius: 10,
            padding: '10px 16px',
            zIndex: 10,
            boxShadow: simState?.rl_training_active
              ? '0 0 20px rgba(56, 189, 248, 0.35), 0 8px 32px rgba(0,0,0,0.6)'
              : '0 0 20px rgba(34, 197, 94, 0.35), 0 8px 32px rgba(0,0,0,0.6)',
            minWidth: 420,
            maxWidth: 620,
            pointerEvents: 'none',
          }}
        >
          {/* Header Row */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: 6 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: simState?.rl_training_active ? '#38bdf8' : '#22c55e',
                  boxShadow: simState?.rl_training_active ? '0 0 8px #38bdf8' : '0 0 8px #22c55e',
                }}
              />
              <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.8px', color: '#f8fafc', textTransform: 'uppercase' }}>
                ⚡ ISAAC LAB — LIVE PPO TRAINING
              </span>
            </div>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
              {simState?.rl_training_model_name || 'PPO-Quadcopter'}
            </span>
          </div>

          {/* Episode, Step, Reward Stats */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#38bdf8' }}>
              <span style={{ color: 'var(--text-muted)' }}>EPISODE:</span>
              <span style={{ fontWeight: 700 }}>
                {simState?.rl_training_episode || 0} / {simState?.rl_training_total_episodes || 4}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-secondary)' }}>
              <span style={{ color: 'var(--text-muted)' }}>STEP:</span>
              <span style={{ fontWeight: 600 }}>{simState?.rl_training_step || 0}</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ color: 'var(--text-muted)' }}>REWARD:</span>
              <span
                style={{
                  fontWeight: 700,
                  color: (simState?.rl_training_reward || 0) >= 0 ? '#4ade80' : '#f87171'
                }}
              >
                {(simState?.rl_training_reward || 0) > 0 ? '+' : ''}
                {(simState?.rl_training_reward || 0).toFixed(1)}
              </span>
            </div>
          </div>

          {/* Dynamic Status Banner */}
          <div
            style={{
              padding: '6px 10px',
              borderRadius: 6,
              fontSize: 11,
              fontWeight: 600,
              backgroundColor: (simState?.rl_training_status?.includes('CRASH') || simState?.rl_training_status?.includes('FAIL'))
                ? 'rgba(239, 68, 68, 0.25)'
                : (simState?.rl_training_status?.includes('CONVERGED') || simState?.rl_training_status?.includes('SAVED') || simState?.rl_training_status?.includes('DEPLOYED'))
                ? 'rgba(34, 197, 94, 0.25)'
                : 'rgba(56, 189, 248, 0.15)',
              border: (simState?.rl_training_status?.includes('CRASH') || simState?.rl_training_status?.includes('FAIL'))
                ? '1px solid rgba(239, 68, 68, 0.5)'
                : (simState?.rl_training_status?.includes('CONVERGED') || simState?.rl_training_status?.includes('SAVED') || simState?.rl_training_status?.includes('DEPLOYED'))
                ? '1px solid rgba(34, 197, 94, 0.5)'
                : '1px solid rgba(56, 189, 248, 0.3)',
              color: (simState?.rl_training_status?.includes('CRASH') || simState?.rl_training_status?.includes('FAIL'))
                ? '#fca5a5'
                : (simState?.rl_training_status?.includes('CONVERGED') || simState?.rl_training_status?.includes('SAVED') || simState?.rl_training_status?.includes('DEPLOYED'))
                ? '#86efac'
                : '#bae6fd',
              textAlign: 'center'
            }}
          >
            {simState?.rl_training_status || 'Autonomous Policy Optimization Active'}
          </div>

          {/* Episode Progress Bar */}
          <div style={{ width: '100%', height: 4, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 2, overflow: 'hidden' }}>
            <div
              style={{
                height: '100%',
                width: `${Math.min(100, Math.max(5, ((simState?.rl_training_episode || 0) / (simState?.rl_training_total_episodes || 4)) * 100))}%`,
                backgroundColor: simState?.rl_training_active ? '#38bdf8' : '#22c55e',
                transition: 'width 0.3s ease-out'
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};
