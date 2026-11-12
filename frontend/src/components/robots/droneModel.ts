import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { createRobotMaterials, enableShadows, disposeGeometries } from './palette';

/**
 * Procedural radial motion blur texture for drone propeller disc.
 * Creates a feathered transparent disc with swept blade trails and a bright circular tip arc.
 */
function createRotorBlurTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return new THREE.CanvasTexture(canvas);
  }

  const cx = 128;
  const cy = 128;
  const maxR = 122;

  ctx.clearRect(0, 0, 256, 256);

  // 1. Concentric radial gradient: transparent center, smooth density, bright outer blade-tip ring
  const grad = ctx.createRadialGradient(cx, cy, 18, cx, cy, maxR);
  grad.addColorStop(0, 'rgba(255, 255, 255, 0)');
  grad.addColorStop(0.25, 'rgba(230, 235, 245, 0.05)');
  grad.addColorStop(0.70, 'rgba(230, 235, 245, 0.16)');
  grad.addColorStop(0.92, 'rgba(245, 250, 255, 0.30)');
  grad.addColorStop(0.96, 'rgba(255, 255, 255, 0.60)'); // Crisp tip arc
  grad.addColorStop(1, 'rgba(255, 255, 255, 0)');

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, maxR, 0, Math.PI * 2);
  ctx.fill();

  // 2. Sweeping fan blade motion blur trails
  for (const angle of [0, Math.PI]) {
    const sweep = ctx.createRadialGradient(cx, cy, 25, cx, cy, maxR);
    sweep.addColorStop(0, 'rgba(255, 255, 255, 0)');
    sweep.addColorStop(0.65, 'rgba(240, 245, 255, 0.20)');
    sweep.addColorStop(0.95, 'rgba(255, 255, 255, 0.42)');
    sweep.addColorStop(1, 'rgba(255, 255, 255, 0)');

    ctx.fillStyle = sweep;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, maxR, angle, angle + Math.PI * 0.42, false);
    ctx.closePath();
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  return texture;
}

/**
 * X-frame quadcopter. Frame: Z up, +X forward, origin at the centre of mass (the simulated body centre).
 * The skid bottoms sit 0.08 m below the origin, matching the physics box, so a landed drone rests on the ground.
 * `rotors` are the four spinning propeller groups; spin them about their local Z.
 */
export interface DroneModel {
  root: THREE.Group;
  rotors: THREE.Object3D[];
  dispose: () => void;
}

export function createDroneModel(): DroneModel {
  const m = createRobotMaterials();
  const root = new THREE.Group();
  root.name = 'drone_model';

  const blurTexture = createRotorBlurTexture();
  const blurMat = new THREE.MeshBasicMaterial({
    map: blurTexture,
    transparent: true,
    opacity: 0.50,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  const rbox = (w: number, d: number, h: number, r: number, mat: THREE.Material) =>
    new THREE.Mesh(new RoundedBoxGeometry(w, d, h, 4, r), mat);
  const zcyl = (rt: number, rb: number, h: number, mat: THREE.Material, seg = 32) => {
    const g = new THREE.CylinderGeometry(rt, rb, h, seg);
    g.rotateX(Math.PI / 2);
    return new THREE.Mesh(g, mat);
  };

  // ---- Fuselage ------------------------------------------------------------------------------------
  const hull = rbox(0.17, 0.105, 0.05, 0.022, m.shell);
  hull.position.set(0.005, 0, 0);
  const seam = rbox(0.172, 0.107, 0.004, 0.002, m.graphite);
  seam.position.set(0.005, 0, -0.012);
  const canopy = rbox(0.1, 0.068, 0.016, 0.008, m.panel);
  canopy.position.set(-0.012, 0, 0.03);
  const battery = rbox(0.115, 0.07, 0.028, 0.01, m.dark);
  battery.position.set(-0.012, 0, -0.04);
  const strap = rbox(0.02, 0.074, 0.03, 0.004, m.graphite);
  strap.position.set(-0.012, 0, -0.04);
  root.add(hull, seam, canopy, battery, strap);

  // Nose: sensor window and an under-slung camera.
  const nose = rbox(0.004, 0.06, 0.016, 0.0035, m.glass);
  nose.position.set(0.0905, 0, 0.008);
  const camBall = new THREE.Mesh(new THREE.SphereGeometry(0.02, 28, 20), m.graphite);
  camBall.position.set(0.082, 0, -0.03);
  const camRing = zcyl(0.0105, 0.0105, 0.006, m.steel, 28);
  camRing.rotation.y = Math.PI / 2;
  camRing.position.set(0.1, 0, -0.03);
  const camLens = zcyl(0.0075, 0.0075, 0.007, m.glass, 24);
  camLens.rotation.y = Math.PI / 2;
  camLens.position.set(0.1, 0, -0.03);
  root.add(nose, camBall, camRing, camLens);

  // ---- Arms, motors and rotors ---------------------------------------------------------------------
  const armLen = 0.24;
  const rotors: THREE.Object3D[] = [];
  for (let k = 0; k < 4; k++) {
    const a = Math.PI / 4 + (k * Math.PI) / 2;
    const arm = new THREE.Group();
    arm.rotation.z = a;
    const bar = rbox(armLen * 0.86, 0.024, 0.014, 0.005, m.trim);
    bar.position.set(armLen * 0.43, 0, 0);
    const cap = rbox(0.05, 0.03, 0.018, 0.007, m.shell);
    cap.position.set(armLen - 0.036, 0, 0.002);
    arm.add(bar, cap);
    root.add(arm);

    const mx = Math.cos(a) * armLen;
    const my = Math.sin(a) * armLen;
    const motor = zcyl(0.022, 0.024, 0.03, m.dark, 36);
    motor.position.set(mx, my, 0.02);
    const collar = zcyl(0.025, 0.025, 0.006, m.panel, 36);
    collar.position.set(mx, my, 0.036);
    root.add(motor, collar);

    // Propeller assembly: Aerodynamic pitched blades, contrast tip markers, ghost blur blades,
    // metallic center spinner nut, and a dynamic motion blur disc.
    const rotor = new THREE.Group();
    rotor.position.set(mx, my, 0.048);

    // Center hub & polished spinner bullet nut
    const hub = zcyl(0.010, 0.010, 0.009, m.steel, 28);
    const spinner = new THREE.Mesh(new THREE.ConeGeometry(0.007, 0.009, 18), m.steel);
    spinner.rotation.x = Math.PI / 2;
    spinner.position.set(0, 0, 0.006);
    rotor.add(hub, spinner);

    // Primary aerodynamic blades with contrast white tip accents
    for (const sgn of [-1, 1]) {
      const bladeGroup = new THREE.Group();

      // Main blade root
      const bladeRoot = new THREE.Mesh(new THREE.BoxGeometry(0.038, 0.021, 0.0032), m.dark);
      bladeRoot.position.set(sgn * 0.024, 0, 0);
      bladeRoot.rotation.x = sgn * 0.15;

      // Mid airfoil
      const bladeMid = new THREE.Mesh(new THREE.BoxGeometry(0.052, 0.018, 0.0026), m.panel);
      bladeMid.position.set(sgn * 0.064, 0, 0);
      bladeMid.rotation.x = sgn * 0.15;

      // High-visibility crisp white tip marker (creates vivid motion trail)
      const bladeTip = new THREE.Mesh(new THREE.BoxGeometry(0.024, 0.013, 0.0020), m.shell);
      bladeTip.position.set(sgn * 0.101, 0, 0);
      bladeTip.rotation.x = sgn * 0.15;

      bladeGroup.add(bladeRoot, bladeMid, bladeTip);
      rotor.add(bladeGroup);

      // Trailing ghost blades for realistic visual persistence of motion
      const ghostMat1 = m.panel.clone();
      ghostMat1.transparent = true;
      ghostMat1.opacity = 0.28;
      ghostMat1.depthWrite = false;
      const ghost1 = new THREE.Mesh(new THREE.BoxGeometry(0.105, 0.016, 0.002), ghostMat1);
      ghost1.position.set(sgn * 0.056, 0, 0.001);
      ghost1.rotation.set(sgn * 0.14, 0, -sgn * 0.24);

      const ghostMat2 = m.panel.clone();
      ghostMat2.transparent = true;
      ghostMat2.opacity = 0.14;
      ghostMat2.depthWrite = false;
      const ghost2 = new THREE.Mesh(new THREE.BoxGeometry(0.098, 0.015, 0.002), ghostMat2);
      ghost2.position.set(sgn * 0.053, 0, 0.0015);
      ghost2.rotation.set(sgn * 0.14, 0, -sgn * 0.46);

      rotor.add(ghost1, ghost2);
    }

    // High-speed motion blur disk
    const blurDisc = new THREE.Mesh(new THREE.PlaneGeometry(0.244, 0.244), blurMat);
    blurDisc.position.set(0, 0, 0.002);
    rotor.add(blurDisc);

    root.add(rotor);
    rotors.push(rotor);
  }

  // ---- Landing skids (bottom at z = -0.08, matching the physics box) --------------------------------
  for (const sy of [-1, 1]) {
    const skid = rbox(0.24, 0.012, 0.012, 0.005, m.dark);
    skid.position.set(0, sy * 0.078, -0.074);
    root.add(skid);
    for (const sx of [-1, 1]) {
      const leg = zcyl(0.0045, 0.0045, 0.04, m.trim, 16);
      leg.position.set(sx * 0.072, sy * 0.07, -0.052);
      leg.rotation.x = -sy * 0.12;
      root.add(leg);
    }
  }

  enableShadows(root);
  // Transparent motion-blur elements and ghost blades must not cast shadows.
  rotors.forEach((r) => r.children.forEach((c) => {
    const mesh = c as THREE.Mesh;
    if (mesh.material && (mesh.material as THREE.Material).transparent) {
      mesh.castShadow = false;
      mesh.receiveShadow = false;
    }
  }));

  return {
    root,
    rotors,
    dispose: () => {
      blurTexture.dispose();
      blurMat.dispose();
      disposeGeometries(root);
      m.dispose();
    },
  };
}
