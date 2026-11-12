import * as THREE from 'three';
import { createRobotMaterials, enableShadows } from './palette';
import { V, rbox, limb, zcyl, joint, ball } from './parts';

/**
 * Bipedal humanoid, about 1.3 m tall, standing on the floor (z = 0), facing +X. Strictly white and grey.
 * Proportions follow an adult human scaled down: hip at 0.64 m, shoulder at 1.07 m, head centre at 1.24 m.
 */
export function createHumanoidModel(): { root: THREE.Group; dispose: () => void } {
  const m = createRobotMaterials();
  const root = new THREE.Group();
  root.name = 'humanoid_model';
  const Y = V(0, 1, 0);

  // ---- Torso ---------------------------------------------------------------------------------------
  const pelvis = rbox(0.17, 0.26, 0.11, 0.04, m.shell);
  pelvis.position.set(0, 0, 0.68);
  const belt = rbox(0.176, 0.266, 0.012, 0.005, m.graphite);
  belt.position.set(0, 0, 0.725);
  const waist = rbox(0.13, 0.2, 0.1, 0.04, m.dark);
  waist.position.set(0, 0, 0.79);
  const chest = rbox(0.22, 0.34, 0.26, 0.07, m.shell);
  chest.position.set(0.005, 0, 0.975);
  const chestPlate = rbox(0.012, 0.21, 0.15, 0.03, m.panel);
  chestPlate.position.set(0.112, 0, 0.99);
  const chestGlass = rbox(0.008, 0.12, 0.04, 0.016, m.glass);
  chestGlass.position.set(0.118, 0, 1.04);
  const pack = rbox(0.08, 0.24, 0.2, 0.04, m.panel);
  pack.position.set(-0.155, 0, 0.98);
  for (let i = 0; i < 4; i++) {
    const vent = rbox(0.004, 0.16, 0.006, 0.002, m.graphite);
    vent.position.set(-0.196, 0, 0.93 + i * 0.03);
    root.add(vent);
  }
  const yoke = rbox(0.12, 0.3, 0.055, 0.028, m.shell);
  yoke.position.set(0, 0, 1.105);
  root.add(pelvis, belt, waist, chest, chestPlate, chestGlass, pack, yoke);

  // ---- Neck and head -------------------------------------------------------------------------------
  root.add(limb(V(0, 0, 1.1), V(0, 0, 1.17), 0.034, 0.03, m.dark));
  const head = rbox(0.18, 0.16, 0.17, 0.07, m.shell);
  head.position.set(0.01, 0, 1.25);
  const visor = rbox(0.01, 0.13, 0.072, 0.034, m.glass);
  visor.position.set(0.0945, 0, 1.252);
  const crown = rbox(0.1, 0.1, 0.02, 0.01, m.panel);
  crown.position.set(0, 0, 1.337);
  root.add(head, visor, crown);
  for (const s of [-1, 1]) {
    root.add(joint(V(0.005, s * 0.088, 1.25), 0.03, Y, m));
  }

  // ---- Arms ----------------------------------------------------------------------------------------
  for (const s of [-1, 1]) {
    const shoulder = V(0, s * 0.205, 1.07);
    const elbow = V(-0.01, s * 0.228, 0.8);
    const wrist = V(0.075, s * 0.238, 0.6);
    root.add(ball(shoulder, 0.056, m.dark), joint(shoulder, 0.05, Y, m));
    root.add(limb(shoulder, elbow, 0.045, 0.036, m.shell));
    root.add(joint(elbow, 0.038, Y, m));
    root.add(limb(elbow, wrist, 0.036, 0.028, m.panel));
    root.add(joint(wrist, 0.027, Y, m));

    // Hand: palm, four fingers and a thumb.
    const palm = rbox(0.04, 0.08, 0.075, 0.018, m.dark);
    palm.position.set(wrist.x + 0.012, wrist.y, wrist.z - 0.05);
    root.add(palm);
    for (let f = 0; f < 4; f++) {
      const fy = wrist.y + (f - 1.5) * 0.02;
      root.add(limb(V(wrist.x + 0.018, fy, wrist.z - 0.085), V(wrist.x + 0.03, fy, wrist.z - 0.15), 0.0085, 0.006, m.trim, 14));
    }
    root.add(limb(V(wrist.x + 0.03, wrist.y - s * 0.04, wrist.z - 0.05), V(wrist.x + 0.055, wrist.y - s * 0.062, wrist.z - 0.095), 0.01, 0.007, m.trim, 14));
  }

  // ---- Legs ----------------------------------------------------------------------------------------
  for (const s of [-1, 1]) {
    const hip = V(0, s * 0.095, 0.64);
    const knee = V(0.015, s * 0.1, 0.35);
    const ankle = V(-0.01, s * 0.1, 0.085);
    root.add(ball(hip, 0.06, m.dark), joint(hip, 0.052, Y, m));
    root.add(limb(hip, knee, 0.062, 0.048, m.shell));
    root.add(joint(knee, 0.052, Y, m));
    root.add(limb(knee, ankle, 0.047, 0.034, m.panel));
    const calf = rbox(0.05, 0.07, 0.17, 0.022, m.shell);
    calf.position.set(-0.045, s * 0.1, 0.215);
    calf.rotation.y = 0.08;
    root.add(calf);
    root.add(joint(ankle, 0.036, Y, m));

    const foot = rbox(0.22, 0.095, 0.05, 0.022, m.dark);
    foot.position.set(0.045, s * 0.1, 0.035);
    const toe = rbox(0.07, 0.09, 0.034, 0.016, m.panel);
    toe.position.set(0.125, s * 0.1, 0.042);
    const sole = rbox(0.225, 0.1, 0.014, 0.006, m.rubber);
    sole.position.set(0.045, s * 0.1, 0.007);
    root.add(foot, toe, sole);
  }

  // A small status strip on the chest would be coloured; the humanoid stays strictly neutral instead.
  void zcyl;
  enableShadows(root);
  return { root, dispose: () => { root.traverse((o) => { const g = (o as THREE.Mesh).geometry; if (g) g.dispose(); }); m.dispose(); } };
}
