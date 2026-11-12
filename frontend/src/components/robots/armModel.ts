import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { createRobotMaterials, enableShadows, RobotMaterials } from './palette';

/**
 * Articulated-arm link styling. The arm is data-driven (built-in arms and imported URDFs), so instead of one
 * hard-coded arm this styles any link geometry: tubes become white/grey segments with dark collars and a
 * seam, the root becomes a pedestal, small boxes become a gripper, and every joint gets an actuator housing
 * aligned to its axis. Strictly white and grey.
 */
export interface ArmGeom {
  type: string;
  size: number[];
  origin_pos?: number[];
  origin_rpy?: number[];
}

export interface ArmLinkOpts {
  isRoot: boolean;
  /** Position of the link in the chain; alternates white and light-grey segments. */
  index: number;
  /** Axis of the joint that moves this link, in the link frame. */
  jointAxis?: THREE.Vector3;
}

const zcyl = (rt: number, rb: number, h: number, mat: THREE.Material, seg = 36) => {
  const g = new THREE.CylinderGeometry(rt, rb, h, seg);
  g.rotateX(Math.PI / 2); // axis along Z, like URDF cylinders
  return new THREE.Mesh(g, mat);
};
const rbox = (w: number, d: number, h: number, r: number, mat: THREE.Material) =>
  new THREE.Mesh(new RoundedBoxGeometry(Math.max(0.004, w), Math.max(0.004, d), Math.max(0.004, h), 3, Math.min(r, w / 2.2, d / 2.2, h / 2.2)), mat);

/** Actuator housing for a joint, with its rotation axis along `axis`. */
function jointHousing(radius: number, axis: THREE.Vector3, m: RobotMaterials): THREE.Group {
  const g = new THREE.Group();
  const len = radius * 1.7;
  const core = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, len, 40), m.dark);
  const capGeo = new THREE.CylinderGeometry(radius * 0.7, radius * 0.7, len * 1.12, 32);
  const cap = new THREE.Mesh(capGeo, m.steel);
  const ring = new THREE.Mesh(new THREE.CylinderGeometry(radius * 1.04, radius * 1.04, len * 0.14, 40), m.graphite);
  g.add(core, cap, ring);
  const ax = axis.clone();
  if (ax.lengthSq() < 1e-9) ax.set(0, 0, 1);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), ax.normalize());
  return g;
}

export function createArmLinkVisual(geom: ArmGeom, o: ArmLinkOpts, m: RobotMaterials): THREE.Group {
  const link = new THREE.Group();
  const body = new THREE.Group();
  const p = geom.origin_pos ?? [0, 0, 0];
  const r = geom.origin_rpy ?? [0, 0, 0];
  body.position.set(p[0] || 0, p[1] || 0, p[2] || 0);
  body.rotation.set(r[0] || 0, r[1] || 0, r[2] || 0);
  link.add(body);

  const seg = o.index % 2 === 0 ? m.shell : m.panel;

  if (geom.type === 'cylinder') {
    const rad = Math.max(0.005, geom.size[0] || 0.05);
    const h = Math.max(0.01, (geom.size[1] || 0.15) * 2);
    if (o.isRoot) {
      // Pedestal: dark base plate, white column, light turntable ring, corner bolts.
      const plate = zcyl(rad * 1.12, rad * 1.12, h * 0.26, m.dark, 56);
      plate.position.z = -h / 2 + h * 0.13;
      const column = zcyl(rad, rad * 0.95, h * 0.54, m.shell, 56);
      column.position.z = -h / 2 + h * 0.26 + h * 0.27;
      const seam = zcyl(rad * 1.01, rad * 1.01, h * 0.035, m.graphite, 56);
      seam.position.z = -h / 2 + h * 0.26;
      const ring = zcyl(rad * 0.8, rad * 0.8, h * 0.2, m.panel, 56);
      ring.position.z = h / 2 - h * 0.1;
      body.add(plate, column, seam, ring);
      for (let i = 0; i < 6; i++) {
        const th = (i / 6) * Math.PI * 2;
        const bolt = zcyl(rad * 0.045, rad * 0.045, h * 0.05, m.steel, 12);
        bolt.position.set(Math.cos(th) * rad * 0.96, Math.sin(th) * rad * 0.96, -h / 2 + h * 0.27);
        body.add(bolt);
      }
    } else {
      // Tube segment with dark end collars and a mid seam.
      const tube = zcyl(rad, rad * 0.93, h * 0.9, seg, 48);
      const c1 = zcyl(rad * 1.1, rad * 1.1, Math.min(0.026, h * 0.12), m.dark, 48);
      c1.position.z = -h / 2 + Math.min(0.026, h * 0.12) / 2;
      const c2 = c1.clone();
      c2.position.z = h / 2 - Math.min(0.026, h * 0.12) / 2;
      const mid = zcyl(rad * 1.012, rad * 1.012, 0.003, m.graphite, 48);
      mid.position.z = h * 0.12;
      body.add(tube, c1, c2, mid);
    }
  } else if (geom.type === 'box') {
    const sx = (geom.size[0] || 0.05) * 2;
    const sy = (geom.size[1] || 0.05) * 2;
    const sz = (geom.size[2] || 0.05) * 2;
    const slender = Math.min(sx, sy, sz) < 0.03;
    // Slender boxes are gripper fingers; larger boxes are housings (palm, wrist).
    const mesh = rbox(sx, sy, sz, 0.006, slender ? m.trim : m.dark);
    body.add(mesh);
    if (!slender) {
      const edge = rbox(sx * 1.0, sy * 1.0, 0.004, 0.002, m.steel);
      edge.position.z = sz / 2;
      body.add(edge);
    }
  } else {
    const rad = Math.max(0.01, geom.size[0] || 0.05);
    body.add(new THREE.Mesh(new THREE.SphereGeometry(rad, 28, 20), m.satin));
  }

  if (!o.isRoot && o.jointAxis) {
    const hubR = Math.max(0.028, (geom.type === 'cylinder' ? geom.size[0] || 0.05 : 0.04) * 1.12);
    link.add(jointHousing(hubR, o.jointAxis, m));
  }
  return link;
}

/** A posed 4-joint arm for the design preview (same proportions as the built-in arm). */
export function createPreviewArm(): { root: THREE.Group; dispose: () => void } {
  const m = createRobotMaterials();
  const root = new THREE.Group();
  const baseH = 0.12;
  const segs = [0.34, 0.289, 0.2465];
  const radii = [0.05, 0.046, 0.042];
  const pose = [0.55, -0.7, 1.15, -0.55];

  root.add(createArmLinkVisual({ type: 'cylinder', size: [0.12, baseH / 2], origin_pos: [0, 0, baseH / 2] }, { isRoot: true, index: 0 }, m));

  let parent: THREE.Group = root;
  let offset = baseH;
  for (let i = 0; i < 4; i++) {
    const axis = i % 2 === 0 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    const jg = new THREE.Group();
    jg.position.z = offset;
    if (axis.z === 1) jg.rotation.z = pose[i]; else jg.rotation.y = pose[i];
    parent.add(jg);
    if (i < 3) {
      jg.add(createArmLinkVisual({ type: 'cylinder', size: [radii[i], segs[i] / 2], origin_pos: [0, 0, segs[i] / 2] }, { isRoot: false, index: i, jointAxis: axis }, m));
      offset = segs[i];
    } else {
      jg.add(createArmLinkVisual({ type: 'box', size: [0.055, 0.02, 0.03], origin_pos: [0, 0, 0.03] }, { isRoot: false, index: i, jointAxis: axis }, m));
      for (const fy of [-1, 1]) {
        const f = createArmLinkVisual({ type: 'box', size: [0.012, 0.012, 0.045], origin_pos: [0, 0, 0.045] }, { isRoot: false, index: i }, m);
        f.position.set(0, fy * 0.035, 0.055);
        jg.add(f);
      }
    }
    parent = jg;
  }
  enableShadows(root);
  return { root, dispose: () => { root.traverse((o) => { const g = (o as THREE.Mesh).geometry; if (g) g.dispose(); }); m.dispose(); } };
}
