import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { createRobotMaterials, enableShadows, disposeGeometries } from './palette';

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

    // Propeller: two pitched blades, a hub and a faint disc that suggests motion.
    const rotor = new THREE.Group();
    rotor.position.set(mx, my, 0.048);
    for (const sgn of [-1, 1]) {
      const blade = new THREE.Mesh(new THREE.BoxGeometry(0.113, 0.019, 0.0032), m.panel);
      blade.position.x = sgn * 0.0585;
      blade.rotation.x = sgn * 0.14;
      rotor.add(blade);
    }
    const hub = zcyl(0.0095, 0.0095, 0.009, m.steel, 24);
    rotor.add(hub);
    const discMat = m.panel.clone();
    discMat.transparent = true;
    discMat.opacity = 0.09;
    discMat.depthWrite = false;
    discMat.side = THREE.DoubleSide;
    rotor.add(new THREE.Mesh(new THREE.CircleGeometry(0.12, 48), discMat));
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
  // The faint motion-hint discs must not cast shadows.
  rotors.forEach((r) => r.children.forEach((c) => {
    if ((c as THREE.Mesh).geometry?.type === 'CircleGeometry') { c.castShadow = false; c.receiveShadow = false; }
  }));
  return {
    root,
    rotors,
    dispose: () => {
      disposeGeometries(root);
      m.dispose();
    },
  };
}
