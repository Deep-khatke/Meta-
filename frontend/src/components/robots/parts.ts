import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type { RobotMaterials } from './palette';

/** Shared building blocks for legged robots: tapered limbs between two points, joint actuators, rounded boxes. */
export const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);

export function rbox(w: number, d: number, h: number, r: number, mat: THREE.Material): THREE.Mesh {
  const rr = Math.min(r, w / 2.2, d / 2.2, h / 2.2);
  return new THREE.Mesh(new RoundedBoxGeometry(w, d, h, 4, Math.max(0.001, rr)), mat);
}

/** Tapered tube from `a` (radius rA) to `b` (radius rB). */
export function limb(a: THREE.Vector3, b: THREE.Vector3, rA: number, rB: number, mat: THREE.Material, seg = 28): THREE.Mesh {
  const dir = b.clone().sub(a);
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(rB, rA, dir.length(), seg), mat);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return mesh;
}

/** Cylinder whose axis is Z. */
export function zcyl(rt: number, rb: number, h: number, mat: THREE.Material, seg = 32): THREE.Mesh {
  const g = new THREE.CylinderGeometry(rt, rb, h, seg);
  g.rotateX(Math.PI / 2);
  return new THREE.Mesh(g, mat);
}

/** Joint actuator at `p` rotating about `axis`: dark core, steel end caps, graphite gasket ring. */
export function joint(p: THREE.Vector3, radius: number, axis: THREE.Vector3, m: RobotMaterials): THREE.Group {
  const g = new THREE.Group();
  const len = radius * 1.7;
  g.add(
    new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, len, 36), m.dark),
    new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.68, radius * 0.68, len * 1.12, 28), m.steel),
    new THREE.Mesh(new THREE.CylinderGeometry(radius * 1.04, radius * 1.04, len * 0.14, 36), m.graphite),
  );
  g.position.copy(p);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.clone().normalize());
  return g;
}

export function ball(p: THREE.Vector3, r: number, mat: THREE.Material): THREE.Mesh {
  const s = new THREE.Mesh(new THREE.SphereGeometry(r, 28, 20), mat);
  s.position.copy(p);
  return s;
}
