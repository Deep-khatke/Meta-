import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { createRoverModel, RoverModel } from '../components/robots/roverModel';
import { createDroneModel, DroneModel } from '../components/robots/droneModel';
import { createPreviewArm } from '../components/robots/armModel';
import { createHumanoidModel } from '../components/robots/humanoidModel';
import { createDogModel } from '../components/robots/dogModel';

/**
 * Robot design preview: a neutral studio for inspecting each robot model on its own.
 * Route: /robots?robot=rover|drone|arm&view=iso   (views: iso, front, side, top, back, detail)
 */
type ViewName = 'iso' | 'front' | 'side' | 'top' | 'back' | 'detail';
type RobotId = 'rover' | 'drone' | 'arm' | 'humanoid' | 'dog';

// Camera presets are authored for the rover (a 0.4 m robot) and scaled per robot.
const VIEWS: Record<ViewName, { pos: [number, number, number]; target: [number, number, number] }> = {
  iso: { pos: [0.66, -0.6, 0.4], target: [0, 0, 0.07] },
  front: { pos: [0.95, 0.0, 0.16], target: [0, 0, 0.07] },
  side: { pos: [0, -0.95, 0.14], target: [0, 0, 0.07] },
  back: { pos: [-0.8, 0.45, 0.3], target: [0, 0, 0.07] },
  top: { pos: [0.001, -0.02, 1.15], target: [0, 0, 0.0] },
  detail: { pos: [0.42, -0.3, 0.22], target: [0.12, 0, 0.07] },
};

// k scales the camera distance; tz is the height of the point to look at.
const FOCUS: Record<RobotId, { k: number; tz: number }> = {
  rover: { k: 1, tz: 0.07 },
  drone: { k: 1.15, tz: 0.3 },
  arm: { k: 2.3, tz: 0.4 },
  humanoid: { k: 3.1, tz: 0.65 },
  dog: { k: 2.1, tz: 0.27 },
};

const ROBOTS: Array<{ id: RobotId; label: string }> = [
  { id: 'rover', label: 'Rover' },
  { id: 'drone', label: 'Drone' },
  { id: 'arm', label: 'Arm' },
  { id: 'humanoid', label: 'Humanoid' },
  { id: 'dog', label: 'Robot dog' },
];

export const RobotPreview: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const params = new URLSearchParams(window.location.search);
  const [robot, setRobot] = useState<RobotId>((params.get('robot') as RobotId) || 'rover');
  const [view, setView] = useState<ViewName>((params.get('view') as ViewName) || 'iso');
  const [spin, setSpin] = useState<boolean>(params.get('spin') === '1');
  const [light, setLight] = useState<boolean>(params.get('bg') === 'light');
  const apiRef = useRef<{ setView: (v: ViewName) => void; setSpin: (s: boolean) => void } | null>(null);

  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.92;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(light ? 0xe9e9e9 : 0x151617);
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    (scene as unknown as { environmentIntensity: number }).environmentIntensity = 0.62;

    const camera = new THREE.PerspectiveCamera(32, el.clientWidth / el.clientHeight, 0.02, 50);
    camera.up.set(0, 0, 1);

    const key = new THREE.DirectionalLight(0xffffff, 1.7);
    key.position.set(1.4, -1.1, 2.2);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -1.4; key.shadow.camera.right = 1.4;
    key.shadow.camera.top = 1.4; key.shadow.camera.bottom = -1.4;
    key.shadow.camera.near = 0.2; key.shadow.camera.far = 8;
    key.shadow.bias = -0.0003; key.shadow.normalBias = 0.01;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xffffff, 0.8);
    rim.position.set(-1.6, 1.2, 1.0);
    scene.add(rim);

    // Studio floor: an unlit dark disc (so the backdrop stays dark), a shadow catcher and a faint grid.
    scene.add(new THREE.Mesh(
      new THREE.CircleGeometry(4, 96),
      new THREE.MeshBasicMaterial({ color: light ? 0xdedede : 0x3a3c3d }),
    ));
    const shadowCatcher = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.ShadowMaterial({ opacity: light ? 0.28 : 0.55 }));
    shadowCatcher.position.z = 0.0005;
    shadowCatcher.receiveShadow = true;
    scene.add(shadowCatcher);
    const grid = new THREE.GridHelper(3, 30, light ? 0xbcbcbc : 0x4a4d4e, light ? 0xcdcdcd : 0x3f4142);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.001;
    scene.add(grid);

    // The robot under review.
    let rover: RoverModel | null = null;
    let drone: DroneModel | null = null;
    let disposeArm: (() => void) | null = null;
    if (robot === 'rover') {
      rover = createRoverModel({
        chassisSize: [0.2, 0.14, 0.05], wheelRadius: 0.06, wheelWidth: 0.03, wheelBase: 0.28,
        casterRadius: 0.025, lidarMount: [0.1, 0, 0.09], cameraMount: [0.18, 0, 0.08],
      });
      rover.root.position.set(0, 0, 0.07);
      scene.add(rover.root);
    } else if (robot === 'drone') {
      drone = createDroneModel();
      drone.root.position.set(0, 0, 0.3); // hovering
      scene.add(drone.root);
    } else if (robot === 'humanoid') {
      const h = createHumanoidModel();
      scene.add(h.root);
      disposeArm = h.dispose;
    } else if (robot === 'dog') {
      const d = createDogModel();
      scene.add(d.root);
      disposeArm = d.dispose;
    } else {
      const arm = createPreviewArm();
      scene.add(arm.root);
      disposeArm = arm.dispose;
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.2;
    controls.maxDistance = 6;
    controls.maxPolarAngle = Math.PI * 0.495;

    const applyView = (v: ViewName) => {
      const f = FOCUS[robot];
      const p = VIEWS[v];
      const base = new THREE.Vector3(...VIEWS.iso.target);
      const tgt = new THREE.Vector3(p.target[0], p.target[1], p.target[2]).sub(base).multiplyScalar(f.k).add(new THREE.Vector3(0, 0, f.tz));
      const pos = new THREE.Vector3(...p.pos).sub(base).multiplyScalar(f.k).add(new THREE.Vector3(0, 0, f.tz));
      camera.position.copy(pos);
      controls.target.copy(tgt);
      controls.update();
    };
    applyView(view);

    let spinning = spin;
    apiRef.current = { setView: applyView, setSpin: (s) => { spinning = s; } };

    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (spinning) {
        if (rover) {
          rover.leftWheel.rotation.y += 0.05;
          rover.rightWheel.rotation.y += 0.05;
          rover.lidarHead.rotation.z += 0.12;
        }
        if (drone) drone.rotors.forEach((r, i) => { r.rotation.z += (i % 2 ? -1 : 1) * 0.5; });
      }
      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    const onResize = () => {
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      controls.dispose();
      rover?.dispose();
      drone?.dispose();
      disposeArm?.();
      pmrem.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      apiRef.current = null;
    };
    // The scene is rebuilt per robot and backdrop; view and spin are applied through apiRef.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [light, robot]);

  useEffect(() => { apiRef.current?.setView(view); }, [view]);
  useEffect(() => { apiRef.current?.setSpin(spin); }, [spin]);

  const btn = (active: boolean): React.CSSProperties => ({
    height: 30, padding: '0 12px', borderRadius: 8, fontSize: 12.5, fontWeight: 600, cursor: 'pointer',
    border: `1px solid ${active ? 'var(--accent-line)' : 'var(--border-subtle)'}`,
    background: active ? 'var(--accent-soft)' : 'var(--bg-surface)',
    color: active ? 'var(--accent)' : 'var(--text-secondary)',
  });

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#151617', color: 'var(--text-primary)', fontFamily: 'var(--font-sans)' }}>
      <div ref={mountRef} style={{ position: 'absolute', inset: 0 }} />

      <div style={{ position: 'absolute', top: 16, left: 16, right: 16, display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 15, fontWeight: 800, letterSpacing: '0.02em' }}>RoboForge</span>
          <span style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>Robot design preview</span>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {ROBOTS.map((r) => (
            <button key={r.id} type="button" style={btn(r.id === robot)} onClick={() => setRobot(r.id)}>{r.label}</button>
          ))}
        </div>
      </div>

      <div style={{ position: 'absolute', bottom: 18, left: 0, right: 0, display: 'flex', gap: 6, justifyContent: 'center', flexWrap: 'wrap' }}>
        {(Object.keys(VIEWS) as ViewName[]).map((v) => (
          <button key={v} type="button" style={btn(view === v)} onClick={() => setView(v)}>
            {v[0].toUpperCase() + v.slice(1)}
          </button>
        ))}
        <span style={{ width: 12 }} />
        <button type="button" style={btn(spin)} onClick={() => setSpin(!spin)}>Animate</button>
        <button type="button" style={btn(light)} onClick={() => setLight(!light)}>Light backdrop</button>
      </div>
    </div>
  );
};
