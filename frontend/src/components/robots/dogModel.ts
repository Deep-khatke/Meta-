import * as THREE from 'three';
import { createRobotMaterials, enableShadows } from './palette';
import { V, rbox, limb, zcyl, joint, ball } from './parts';

/**
 * Quadruped robot dog, about 0.55 m tall at the back, standing on the floor (z = 0), facing +X.
 * Strictly white and grey. Every leg has a hip actuator, a chunky thigh with a motor housing, a knee joint,
 * a slim shin and a rubber foot, in the bent "<" stance used by real quadrupeds.
 */
export function createDogModel(): { root: THREE.Group; dispose: () => void } {
  const m = createRobotMaterials();
  const root = new THREE.Group();
  root.name = 'dog_model';
  const bz = 0.44; // body centre height

  // ---- Body ----------------------------------------------------------------------------------------
  const shell = rbox(0.52, 0.22, 0.15, 0.058, m.shell);
  shell.position.set(0, 0, bz);
  const belly = rbox(0.48, 0.19, 0.07, 0.035, m.dark);
  belly.position.set(0, 0, bz - 0.07);
  const seam = rbox(0.526, 0.226, 0.006, 0.003, m.graphite);
  seam.position.set(0, 0, bz - 0.032);
  const plate = rbox(0.3, 0.15, 0.012, 0.006, m.panel);
  plate.position.set(-0.04, 0, bz + 0.078);
  root.add(shell, belly, seam, plate);
  for (const sy of [-1, 1]) {
    const rail = rbox(0.28, 0.01, 0.012, 0.004, m.graphite);
    rail.position.set(-0.04, sy * 0.07, bz + 0.088);
    root.add(rail);
  }

  // Top sensor puck.
  const puck = zcyl(0.035, 0.037, 0.03, m.dark, 40);
  puck.position.set(-0.12, 0, bz + 0.104);
  const puckBand = zcyl(0.0372, 0.0372, 0.01, m.glass, 40);
  puckBand.position.set(-0.12, 0, bz + 0.104);
  const puckCap = zcyl(0.028, 0.03, 0.004, m.panel, 36);
  puckCap.position.set(-0.12, 0, bz + 0.12);
  root.add(puck, puckBand, puckCap);

  // Head module with a camera visor.
  const head = rbox(0.13, 0.17, 0.12, 0.048, m.shell);
  head.position.set(0.31, 0, bz + 0.015);
  const visor = rbox(0.008, 0.14, 0.06, 0.026, m.glass);
  visor.position.set(0.378, 0, bz + 0.02);
  root.add(head, visor);
  for (const [yy, rr] of [[-0.045, 0.0085], [0, 0.0115], [0.045, 0.0085]] as const) {
    const ring = zcyl(rr + 0.003, rr + 0.003, 0.004, m.steel, 28);
    ring.rotation.y = Math.PI / 2;
    ring.position.set(0.381, yy, bz + 0.02);
    const lens = zcyl(rr, rr, 0.005, m.glass, 24);
    lens.rotation.y = Math.PI / 2;
    lens.position.set(0.383, yy, bz + 0.02);
    root.add(ring, lens);
  }
  const tail = rbox(0.04, 0.2, 0.1, 0.03, m.dark);
  tail.position.set(-0.28, 0, bz);
  root.add(tail);

  // ---- Legs ----------------------------------------------------------------------------------------
  const X = V(1, 0, 0);
  const Y = V(0, 1, 0);
  for (const hx of [0.22, -0.22]) {
    for (const s of [-1, 1]) {
      const hipAct = V(hx, s * 0.12, bz - 0.01);
      const top = V(hx, s * 0.148, bz - 0.01);
      const knee = V(hx - 0.065, s * 0.148, 0.22);
      const foot = V(hx + 0.015, s * 0.148, 0.024);

      root.add(joint(hipAct, 0.05, X, m)); // hip abduction actuator at the body corner
      root.add(limb(top, knee, 0.04, 0.03, m.shell));
      const mid = top.clone().lerp(knee, 0.3);
      const motor = rbox(0.07, 0.06, 0.09, 0.026, m.dark);
      motor.position.copy(mid);
      motor.position.y += s * 0.012;
      root.add(motor);
      root.add(joint(knee, 0.032, Y, m));
      root.add(limb(knee, foot, 0.026, 0.017, m.trim));
      root.add(ball(foot, 0.024, m.rubber));
    }
  }

  enableShadows(root);
  return { root, dispose: () => { root.traverse((o) => { const g = (o as THREE.Mesh).geometry; if (g) g.dispose(); }); m.dispose(); } };
}
