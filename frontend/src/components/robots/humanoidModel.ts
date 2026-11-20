import * as THREE from 'three';
import { createRobotMaterials, enableShadows } from './palette';
import { V, rbox, limb, joint, ball } from './parts';

/**
 * Bipedal humanoid, about 1.3 m tall, standing on the floor (z = 0), facing +X. Strictly white and grey.
 *
 * The model is split into links that match backend/robot_presets.py. `HUMANOID_ORIGINS` gives each link's
 * origin in the standing pose; every mesh is authored in world coordinates and then shifted into its link's
 * own frame, so the same meshes serve both the static preview and the joint-driven robot in the studio.
 */
export const HUMANOID_ORIGINS: Record<string, [number, number, number]> = {
  pelvis: [0, 0, 0.68], chest: [0, 0, 0.74], head: [0, 0, 1.12],
  l_shoulder: [0, 0.205, 1.07], l_upper_arm: [0, 0.205, 1.07], l_forearm: [-0.01, 0.228, 0.8],
  r_shoulder: [0, -0.205, 1.07], r_upper_arm: [0, -0.205, 1.07], r_forearm: [-0.01, -0.228, 0.8],
  l_thigh: [0, 0.095, 0.64], l_shin: [0.015, 0.1, 0.35], l_foot: [-0.01, 0.1, 0.085],
  r_thigh: [0, -0.095, 0.64], r_shin: [0.015, -0.1, 0.35], r_foot: [-0.01, -0.1, 0.085],
};

export interface PresetLinks {
  links: Record<string, THREE.Group>;
  dispose: () => void;
}

export function buildHumanoidLinks(): PresetLinks {
  const m = createRobotMaterials();
  const Y = V(0, 1, 0);
  const links: Record<string, THREE.Group> = {};
  for (const k of Object.keys(HUMANOID_ORIGINS)) links[k] = new THREE.Group();
  const put = (link: string, ...objs: THREE.Object3D[]) => objs.forEach((o) => links[link].add(o));
  const at = <T extends THREE.Object3D>(o: T, x: number, y: number, z: number): T => { o.position.set(x, y, z); return o; };

  // ---- Pelvis ----
  put('pelvis',
    at(rbox(0.17, 0.26, 0.11, 0.04, m.shell), 0, 0, 0.68),
    at(rbox(0.176, 0.266, 0.012, 0.005, m.graphite), 0, 0, 0.725));

  // ---- Chest and back ----
  put('chest',
    at(rbox(0.13, 0.2, 0.1, 0.04, m.dark), 0, 0, 0.79),
    at(rbox(0.22, 0.34, 0.26, 0.07, m.shell), 0.005, 0, 0.975),
    at(rbox(0.012, 0.21, 0.15, 0.03, m.panel), 0.112, 0, 0.99),
    at(rbox(0.008, 0.12, 0.04, 0.016, m.glass), 0.118, 0, 1.04),
    at(rbox(0.08, 0.24, 0.2, 0.04, m.panel), -0.155, 0, 0.98),
    at(rbox(0.12, 0.3, 0.055, 0.028, m.shell), 0, 0, 1.105));
  for (let i = 0; i < 4; i++) put('chest', at(rbox(0.004, 0.16, 0.006, 0.002, m.graphite), -0.196, 0, 0.93 + i * 0.03));

  // ---- Head ----
  put('head',
    limb(V(0, 0, 1.1), V(0, 0, 1.17), 0.034, 0.03, m.dark),
    at(rbox(0.18, 0.16, 0.17, 0.07, m.shell), 0.01, 0, 1.25),
    at(rbox(0.01, 0.13, 0.072, 0.034, m.glass), 0.0945, 0, 1.252),
    at(rbox(0.1, 0.1, 0.02, 0.01, m.panel), 0, 0, 1.337),
    joint(V(0.005, 0.088, 1.25), 0.03, Y, m),
    joint(V(0.005, -0.088, 1.25), 0.03, Y, m));

  // ---- Arms ----
  for (const [side, s] of [['l', 1], ['r', -1]] as const) {
    const shoulder = V(0, s * 0.205, 1.07);
    const elbow = V(-0.01, s * 0.228, 0.8);
    const wrist = V(0.075, s * 0.238, 0.6);
    put(`${side}_shoulder`, ball(shoulder, 0.056, m.dark), joint(shoulder, 0.05, Y, m));
    put(`${side}_upper_arm`, limb(shoulder, elbow, 0.045, 0.036, m.shell), joint(elbow, 0.038, Y, m));
    put(`${side}_forearm`, limb(elbow, wrist, 0.036, 0.028, m.panel), joint(wrist, 0.027, Y, m),
      at(rbox(0.04, 0.08, 0.075, 0.018, m.dark), wrist.x + 0.012, wrist.y, wrist.z - 0.05));
    for (let f = 0; f < 4; f++) {
      const fy = wrist.y + (f - 1.5) * 0.02;
      put(`${side}_forearm`, limb(V(wrist.x + 0.018, fy, wrist.z - 0.085), V(wrist.x + 0.03, fy, wrist.z - 0.15), 0.0085, 0.006, m.trim, 14));
    }
    put(`${side}_forearm`, limb(V(wrist.x + 0.03, wrist.y - s * 0.04, wrist.z - 0.05), V(wrist.x + 0.055, wrist.y - s * 0.062, wrist.z - 0.095), 0.01, 0.007, m.trim, 14));
  }

  // ---- Legs ----
  for (const [side, s] of [['l', 1], ['r', -1]] as const) {
    const hip = V(0, s * 0.095, 0.64);
    const knee = V(0.015, s * 0.1, 0.35);
    const ankle = V(-0.01, s * 0.1, 0.085);
    put(`${side}_thigh`, ball(hip, 0.06, m.dark), joint(hip, 0.052, Y, m), limb(hip, knee, 0.062, 0.048, m.shell), joint(knee, 0.052, Y, m));
    const calf = at(rbox(0.05, 0.07, 0.17, 0.022, m.shell), -0.045, s * 0.1, 0.215);
    calf.rotation.y = 0.08;
    put(`${side}_shin`, limb(knee, ankle, 0.047, 0.034, m.panel), calf, joint(ankle, 0.036, Y, m));
    put(`${side}_foot`,
      at(rbox(0.22, 0.095, 0.05, 0.022, m.dark), 0.045, s * 0.1, 0.035),
      at(rbox(0.07, 0.09, 0.034, 0.016, m.panel), 0.125, s * 0.1, 0.042),
      at(rbox(0.225, 0.1, 0.014, 0.006, m.rubber), 0.045, s * 0.1, 0.007));
  }

  // Shift every mesh from world coordinates into its link's own frame.
  for (const [name, g] of Object.entries(links)) {
    const o = HUMANOID_ORIGINS[name];
    g.children.forEach((c) => c.position.sub(new THREE.Vector3(o[0], o[1], o[2])));
    enableShadows(g);
  }
  return { links, dispose: () => m.dispose() };
}

/** Static, assembled humanoid in its standing pose (used by the /robots preview). */
export function createHumanoidModel(): { root: THREE.Group; dispose: () => void } {
  const { links, dispose } = buildHumanoidLinks();
  const root = new THREE.Group();
  root.name = 'humanoid_model';
  for (const [name, g] of Object.entries(links)) {
    const o = HUMANOID_ORIGINS[name];
    g.position.set(o[0], o[1], o[2]);
    root.add(g);
  }
  return { root, dispose: () => { root.traverse((o) => { const geo = (o as THREE.Mesh).geometry; if (geo) geo.dispose(); }); dispose(); } };
}
