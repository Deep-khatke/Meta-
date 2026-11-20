import * as THREE from 'three';
import { createRobotMaterials, enableShadows } from './palette';
import { V, rbox, limb, zcyl, joint, ball } from './parts';
import type { PresetLinks } from './humanoidModel';

/**
 * Quadruped robot dog, about 0.55 m tall at the back, standing on the floor (z = 0), facing +X. Strictly
 * white and grey. Split into links that match backend/robot_presets.py (see `DOG_ORIGINS`).
 */
const BZ = 0.44;
export const DOG_ORIGINS: Record<string, [number, number, number]> = { body: [0, 0, BZ] };
for (const [fr, hx] of [['f', 0.22], ['r', -0.22]] as const) {
  for (const [side, s] of [['l', 1], ['r', -1]] as const) {
    const p = `${fr}${side}`;
    DOG_ORIGINS[`${p}_hip_link`] = [hx, s * 0.12, BZ - 0.01];
    DOG_ORIGINS[`${p}_thigh`] = [hx, s * 0.148, BZ - 0.01];
    DOG_ORIGINS[`${p}_shin`] = [hx - 0.065, s * 0.148, 0.22];
  }
}

export function buildDogLinks(): PresetLinks {
  const m = createRobotMaterials();
  const X = V(1, 0, 0);
  const Y = V(0, 1, 0);
  const links: Record<string, THREE.Group> = {};
  for (const k of Object.keys(DOG_ORIGINS)) links[k] = new THREE.Group();
  const put = (link: string, ...objs: THREE.Object3D[]) => objs.forEach((o) => links[link].add(o));
  const at = <T extends THREE.Object3D>(o: T, x: number, y: number, z: number): T => { o.position.set(x, y, z); return o; };

  // ---- Body ----
  put('body',
    at(rbox(0.52, 0.22, 0.15, 0.058, m.shell), 0, 0, BZ),
    at(rbox(0.48, 0.19, 0.07, 0.035, m.dark), 0, 0, BZ - 0.07),
    at(rbox(0.526, 0.226, 0.006, 0.003, m.graphite), 0, 0, BZ - 0.032),
    at(rbox(0.3, 0.15, 0.012, 0.006, m.panel), -0.04, 0, BZ + 0.078),
    at(rbox(0.28, 0.01, 0.012, 0.004, m.graphite), -0.04, 0.07, BZ + 0.088),
    at(rbox(0.28, 0.01, 0.012, 0.004, m.graphite), -0.04, -0.07, BZ + 0.088),
    at(zcyl(0.035, 0.037, 0.03, m.dark, 40), -0.12, 0, BZ + 0.104),
    at(zcyl(0.0372, 0.0372, 0.01, m.glass, 40), -0.12, 0, BZ + 0.104),
    at(zcyl(0.028, 0.03, 0.004, m.panel, 36), -0.12, 0, BZ + 0.12),
    at(rbox(0.13, 0.17, 0.12, 0.048, m.shell), 0.31, 0, BZ + 0.015),
    at(rbox(0.008, 0.14, 0.06, 0.026, m.glass), 0.378, 0, BZ + 0.02),
    at(rbox(0.04, 0.2, 0.1, 0.03, m.dark), -0.28, 0, BZ));
  for (const [yy, rr] of [[-0.045, 0.0085], [0, 0.0115], [0.045, 0.0085]] as const) {
    const ring = at(zcyl(rr + 0.003, rr + 0.003, 0.004, m.steel, 28), 0.381, yy, BZ + 0.02);
    ring.rotation.y = Math.PI / 2;
    const lens = at(zcyl(rr, rr, 0.005, m.glass, 24), 0.383, yy, BZ + 0.02);
    lens.rotation.y = Math.PI / 2;
    put('body', ring, lens);
  }

  // ---- Legs ----
  for (const [fr, hx] of [['f', 0.22], ['r', -0.22]] as const) {
    for (const [side, s] of [['l', 1], ['r', -1]] as const) {
      const p = `${fr}${side}`;
      const hipAct = V(hx, s * 0.12, BZ - 0.01);
      const top = V(hx, s * 0.148, BZ - 0.01);
      const knee = V(hx - 0.065, s * 0.148, 0.22);
      const foot = V(hx + 0.015, s * 0.148, 0.024);
      put(`${p}_hip_link`, joint(hipAct, 0.05, X, m));
      const motor = rbox(0.07, 0.06, 0.09, 0.026, m.dark);
      motor.position.copy(top.clone().lerp(knee, 0.3));
      motor.position.y += s * 0.012;
      put(`${p}_thigh`, limb(top, knee, 0.04, 0.03, m.shell), motor, joint(knee, 0.032, Y, m));
      put(`${p}_shin`, limb(knee, foot, 0.026, 0.017, m.trim), ball(foot, 0.024, m.rubber));
    }
  }

  for (const [name, g] of Object.entries(links)) {
    const o = DOG_ORIGINS[name];
    g.children.forEach((c) => c.position.sub(new THREE.Vector3(o[0], o[1], o[2])));
    enableShadows(g);
  }
  return { links, dispose: () => m.dispose() };
}

/** Static, assembled dog in its standing pose (used by the /robots preview). */
export function createDogModel(): { root: THREE.Group; dispose: () => void } {
  const { links, dispose } = buildDogLinks();
  const root = new THREE.Group();
  root.name = 'dog_model';
  for (const [name, g] of Object.entries(links)) {
    const o = DOG_ORIGINS[name];
    g.position.set(o[0], o[1], o[2]);
    root.add(g);
  }
  return { root, dispose: () => { root.traverse((o) => { const geo = (o as THREE.Mesh).geometry; if (geo) geo.dispose(); }); dispose(); } };
}
