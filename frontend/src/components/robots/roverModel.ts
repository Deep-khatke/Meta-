import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { createRobotMaterials, enableShadows, disposeGeometries, RobotMaterials } from './palette';

/**
 * Differential-drive rover ("ForgeBot").
 *
 * Frame: Z up, +X forward, origin at the chassis centre (the same frame the physics uses), so the model
 * drops straight onto the simulated body. All structural dimensions come from the physics config, which
 * keeps the wheels exactly on the ground. Only the visual detail is added on top.
 */
export interface RoverSpec {
  /** Half extents [x, y, z] of the physics chassis. */
  chassisSize: [number, number, number];
  wheelRadius: number;
  wheelWidth: number;
  /** Distance between the two driven wheels. */
  wheelBase: number;
  casterRadius: number;
  /** Wheel axle height relative to the chassis centre (physics uses -0.01). */
  wheelCenterZ?: number;
  lidarMount: [number, number, number];
  cameraMount: [number, number, number];
  lidarEnabled?: boolean;
  cameraEnabled?: boolean;
}

export interface RoverModel {
  root: THREE.Group;
  leftWheel: THREE.Group;
  rightWheel: THREE.Group;
  lidarAssembly: THREE.Group;
  /** The rotating part of the LiDAR; spin it about Z. */
  lidarHead: THREE.Group;
  cameraAssembly: THREE.Group;
  /** Coordinate-frame helpers, hidden by default. */
  frameAxes: THREE.Group;
  dispose: () => void;
}

/** Plan-view outline: rounded rear corners, chamfered nose. */
function planShape(a: number, b: number, rear: number, chamfer: number): THREE.Shape {
  const s = new THREE.Shape();
  const cy = chamfer * 0.9;
  s.moveTo(-a + rear, -b);
  s.lineTo(a - chamfer, -b);
  s.lineTo(a, -b + cy);
  s.lineTo(a, b - cy);
  s.lineTo(a - chamfer, b);
  s.lineTo(-a + rear, b);
  s.quadraticCurveTo(-a, b, -a, b - rear);
  s.lineTo(-a, -b + rear);
  s.quadraticCurveTo(-a, -b, -a + rear, -b);
  return s;
}

/** A bevelled slab with the given plan outline, spanning z0..z0+h with a soft edge of radius `bevel`. */
function slab(
  a: number, b: number, rear: number, chamfer: number,
  z0: number, h: number, bevel: number, mat: THREE.Material,
): THREE.Mesh {
  const shape = planShape(a - bevel, b - bevel, Math.max(0.004, rear - bevel), Math.max(0.004, chamfer - bevel));
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.001, h - 2 * bevel),
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 9,
    curveSegments: 18,
  });
  geo.translate(0, 0, z0 + bevel);
  return new THREE.Mesh(geo, mat);
}

function roundedBox(w: number, d: number, h: number, r: number, mat: THREE.Material): THREE.Mesh {
  return new THREE.Mesh(new RoundedBoxGeometry(w, d, h, 4, r), mat);
}

/** Cylinder whose axis is the local Y axis (three.js default). */
function cyl(rTop: number, rBottom: number, h: number, mat: THREE.Material, seg = 40): THREE.Mesh {
  return new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBottom, h, seg), mat);
}

/**
 * One drive wheel. The axle runs along the group's local Y, so `group.rotation.y` spins the wheel and the
 * viewport's existing wheel-velocity animation keeps working.
 */
function buildWheel(wr: number, ww: number, m: RobotMaterials, spokes = 5): THREE.Group {
  const g = new THREE.Group();

  // Tyre: a revolved profile with rounded shoulders.
  const r0 = wr * 0.6;
  const s = Math.min(0.008, ww * 0.28);
  const pts: THREE.Vector2[] = [new THREE.Vector2(r0, -ww / 2), new THREE.Vector2(wr - s, -ww / 2)];
  for (let k = 1; k <= 5; k++) {
    const t = (k / 5) * (Math.PI / 2);
    pts.push(new THREE.Vector2(wr - s + s * Math.sin(t), -ww / 2 + s - s * Math.cos(t)));
  }
  pts.push(new THREE.Vector2(wr, ww / 2 - s));
  for (let k = 1; k <= 5; k++) {
    const t = (k / 5) * (Math.PI / 2);
    pts.push(new THREE.Vector2(wr - s + s * Math.cos(t), ww / 2 - s + s * Math.sin(t)));
  }
  pts.push(new THREE.Vector2(r0, ww / 2));
  const tireMat = m.rubber.clone();
  tireMat.side = THREE.DoubleSide;
  g.add(new THREE.Mesh(new THREE.LatheGeometry(pts, 64), tireMat));

  // Staggered tread blocks. Their rotation is what makes the wheel visibly turn.
  const treadCount = 32;
  const tread = new THREE.InstancedMesh(
    new THREE.BoxGeometry((2 * Math.PI * wr) / treadCount * 0.62, ww * 0.46, 0.0045),
    m.rubber, treadCount,
  );
  const dummy = new THREE.Object3D();
  for (let i = 0; i < treadCount; i++) {
    const th = (i / treadCount) * Math.PI * 2;
    const R = wr + 0.0008;
    dummy.position.set(R * Math.sin(th), (i % 2 ? 1 : -1) * ww * 0.2, R * Math.cos(th));
    dummy.rotation.set(0, th, 0);
    dummy.updateMatrix();
    tread.setMatrixAt(i, dummy.matrix);
  }
  tread.castShadow = false;
  g.add(tread);

  // Rim: light outer ring, dark dish, steel-grey spokes and a centre cap.
  const rim = cyl(wr * 0.63, wr * 0.63, ww * 0.82, m.panel, 56);
  g.add(rim);
  const dish = cyl(wr * 0.5, wr * 0.5, ww * 0.86, m.graphite, 56);
  g.add(dish);
  for (let i = 0; i < spokes; i++) {
    const th = (i / spokes) * Math.PI * 2;
    const spoke = new THREE.Mesh(new THREE.BoxGeometry(wr * 0.13, ww * 0.9, wr * 0.5), m.panel);
    const rr = wr * 0.3;
    spoke.position.set(rr * Math.sin(th), 0, rr * Math.cos(th));
    spoke.rotation.set(0, th, 0);
    g.add(spoke);
  }
  const cap = cyl(wr * 0.18, wr * 0.18, ww * 0.96, m.steel, 32);
  g.add(cap);
  const capDot = cyl(wr * 0.07, wr * 0.07, ww * 1.0, m.graphite, 20);
  g.add(capDot);

  return g;
}

/** Fender arc that wraps the top of a wheel. Built in the wheel plane and extruded across the width. */
function buildFender(wr: number, width: number, m: RobotMaterials): THREE.Mesh {
  const R2 = wr + 0.0105;
  const R1 = wr + 0.0045;
  const a0 = -0.38;
  const a1 = Math.PI + 0.38;
  const sh = new THREE.Shape();
  sh.absarc(0, 0, R2, a0, a1, false);
  sh.absarc(0, 0, R1, a1, a0, true);
  const geo = new THREE.ExtrudeGeometry(sh, {
    depth: width - 0.004, bevelEnabled: true, bevelThickness: 0.002, bevelSize: 0.002, bevelSegments: 3, curveSegments: 40,
  });
  geo.translate(0, 0, -(width - 0.004) / 2);
  geo.rotateX(Math.PI / 2); // shape +y -> world +z; extrusion -> lateral
  return new THREE.Mesh(geo, m.shell);
}

function buildCaster(cr: number, m: RobotMaterials): THREE.Group {
  const g = new THREE.Group();
  const ball = new THREE.Mesh(new THREE.SphereGeometry(cr, 28, 20), m.satin);
  g.add(ball);
  const collar = cyl(cr * 0.98, cr * 0.98, 0.006, m.graphite, 28);
  collar.position.z = cr * 0.78;
  collar.rotation.x = Math.PI / 2;
  g.add(collar);
  return g;
}

export function createRoverModel(spec: RoverSpec): RoverModel {
  const m = createRobotMaterials();
  const [hx, hy, hz] = spec.chassisSize;
  const wr = spec.wheelRadius;
  const ww = spec.wheelWidth;
  const wheelZ = spec.wheelCenterZ ?? -0.01;
  const cr = spec.casterRadius;

  const root = new THREE.Group();
  root.name = 'rover_model';

  // ---- Body ---------------------------------------------------------------------------------------
  // The body sits between the wheels (wheels are outboard), so its half width stops just inside the tyre.
  const bodyA = hx - 0.004;
  const bodyB = Math.min(hy, spec.wheelBase / 2) - ww / 2 - 0.004;
  const bottom = -hz;
  const tubH = 0.032;
  const grooveH = 0.004;
  const shellBottom = bottom + tubH + grooveH;
  const shellH = hz - shellBottom;

  const tub = slab(bodyA + 0.002, bodyB + 0.001, 0.036, 0.06, bottom, tubH, 0.006, m.dark);
  const groove = slab(bodyA - 0.001, bodyB - 0.001, 0.032, 0.058, bottom + tubH, grooveH, 0.001, m.graphite);
  const shell = slab(bodyA, bodyB, 0.034, 0.06, shellBottom, shellH, 0.011, m.shell);
  root.add(tub, groove, shell);

  // Top deck: graphite seam under a light-grey inset plate.
  const deckZ = hz;
  const seam = slab(bodyA - 0.03, bodyB - 0.024, 0.026, 0.05, deckZ - 0.002, 0.006, 0.0015, m.graphite);
  const plate = slab(bodyA - 0.034, bodyB - 0.028, 0.024, 0.048, deckZ - 0.002, 0.0085, 0.002, m.panel);
  root.add(seam, plate);
  const deckTop = deckZ + 0.0065;

  // Rear vents.
  for (let i = 0; i < 4; i++) {
    const v = roundedBox(0.05, 0.0045, 0.0022, 0.001, m.graphite);
    v.position.set(-bodyA + 0.075, -0.027 + i * 0.018, deckTop - 0.0004);
    root.add(v);
  }

  // Front sensor visor (flat face of the chamfered nose). The RGB-D camera lives behind it, which keeps
  // the LiDAR's forward view clear and the silhouette clean.
  const visorW = (bodyB - 0.056) * 2 - 0.006;
  const visorZ = shellBottom + shellH * 0.5;
  const visor = roundedBox(0.005, visorW, 0.02, 0.0022, m.glass);
  visor.position.set(bodyA + 0.0006, 0, visorZ);
  root.add(visor);

  const cameraAssembly = new THREE.Group();
  cameraAssembly.name = 'camera_sensor';
  const lensAt = (y: number, ringR: number, lensR: number) => {
    const ring = cyl(ringR, ringR, 0.004, m.steel, 28);
    ring.rotation.z = Math.PI / 2;
    ring.position.set(bodyA + 0.0034, y, visorZ);
    const lens = cyl(lensR, lensR, 0.0048, m.glass, 24);
    lens.rotation.z = Math.PI / 2;
    lens.position.set(bodyA + 0.0038, y, visorZ);
    cameraAssembly.add(ring, lens);
  };
  lensAt(-visorW * 0.3, 0.0066, 0.0047); // left depth sensor
  lensAt(0, 0.0092, 0.0068); // RGB camera
  lensAt(visorW * 0.3, 0.0066, 0.0047); // right depth sensor
  cameraAssembly.visible = spec.cameraEnabled !== false;
  root.add(cameraAssembly);

  // Side seam lines and a rear service-port panel break up the flat sides.
  const seamZ = shellBottom + shellH * 0.4;
  for (const sy of [-1, 1]) {
    const seamLine = roundedBox(0.23, 0.0016, 0.0022, 0.0006, m.graphite);
    seamLine.position.set(-0.015, sy * (bodyB + 0.0002), seamZ);
    const port = roundedBox(0.05, 0.0016, 0.015, 0.0025, m.panel);
    port.position.set(-bodyA + 0.075, sy * (bodyB + 0.0002), shellBottom + shellH * 0.7);
    const portSlot = roundedBox(0.032, 0.0018, 0.0035, 0.001, m.graphite);
    portSlot.position.set(-bodyA + 0.075, sy * (bodyB + 0.0004), shellBottom + shellH * 0.7);
    root.add(seamLine, port, portSlot);
  }

  // ---- Wheels, fenders and axles ------------------------------------------------------------------
  const track = spec.wheelBase / 2;
  const leftWheel = buildWheel(wr, ww, m);
  leftWheel.position.set(0, track, wheelZ);
  const rightWheel = buildWheel(wr, ww, m);
  rightWheel.position.set(0, -track, wheelZ);
  root.add(leftWheel, rightWheel);

  for (const sy of [-1, 1]) {
    const fender = buildFender(wr, ww + 0.011, m);
    fender.position.set(0, sy * track, wheelZ);
    root.add(fender);
    // Axle stub bridging the gap between the body and the wheel.
    const axle = cyl(0.0085, 0.0085, ww * 0.9 + 0.008, m.steel, 20);
    axle.position.set(0, sy * (track - ww * 0.5 + 0.004), wheelZ);
    root.add(axle);
  }

  // ---- Ball casters -------------------------------------------------------------------------------
  const casterX = hx * 0.75;
  const casterY = spec.wheelBase * 0.35;
  const casterZ = wheelZ - wr + cr;
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      const c = buildCaster(cr, m);
      c.position.set(sx * casterX, sy * casterY, casterZ);
      root.add(c);
    }
  }

  // ---- LiDAR: low pedestal + spinning head ---------------------------------------------------------
  const lidarAssembly = new THREE.Group();
  lidarAssembly.name = 'lidar_sensor';
  const [lx, ly, lz] = spec.lidarMount;
  lidarAssembly.position.set(lx, ly, 0);
  const pedH = Math.max(0.006, lz - 0.015 - deckTop);
  const baseRing = cyl(0.036, 0.036, 0.004, m.panel, 40);
  baseRing.rotation.x = Math.PI / 2;
  baseRing.position.z = deckTop + 0.002;
  const pedestal = cyl(0.024, 0.027, pedH, m.graphite, 32);
  pedestal.rotation.x = Math.PI / 2;
  pedestal.position.z = deckTop + 0.004 + pedH / 2;
  lidarAssembly.add(baseRing, pedestal);

  const lidarHead = new THREE.Group();
  lidarHead.position.z = lz;
  const housing = cyl(0.031, 0.031, 0.03, m.dark, 48);
  housing.rotation.x = Math.PI / 2;
  const band = cyl(0.0318, 0.0318, 0.011, m.glass, 48);
  band.rotation.x = Math.PI / 2;
  band.position.z = 0.001;
  const topCap = cyl(0.026, 0.028, 0.004, m.panel, 40);
  topCap.rotation.x = Math.PI / 2;
  topCap.position.z = 0.017;
  const tab = roundedBox(0.012, 0.008, 0.0055, 0.002, m.steel);
  tab.position.set(0.02, 0, 0.0205);
  lidarHead.add(housing, band, topCap, tab);
  lidarAssembly.add(lidarHead);
  lidarAssembly.visible = spec.lidarEnabled !== false;
  root.add(lidarAssembly);

  // ---- Debug coordinate frames (hidden unless the viewport asks for them) --------------------------
  const frameAxes = new THREE.Group();
  frameAxes.name = 'rover_frame_axes';
  const baseAxes = new THREE.AxesHelper(0.35);
  baseAxes.position.set(0, 0, hz + 0.02);
  frameAxes.add(baseAxes);
  const lidarAxes = new THREE.AxesHelper(0.12);
  lidarAxes.position.set(lx, ly, lz);
  frameAxes.add(lidarAxes);
  frameAxes.visible = false;
  root.add(frameAxes);

  enableShadows(root);

  return {
    root, leftWheel, rightWheel, lidarAssembly, lidarHead, cameraAssembly, frameAxes,
    dispose: () => {
      disposeGeometries(root);
      m.dispose();
      leftWheel.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (mesh.isMesh && (mesh.material as THREE.Material).side === THREE.DoubleSide) (mesh.material as THREE.Material).dispose();
      });
    },
  };
}
