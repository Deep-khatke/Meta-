import * as THREE from 'three';

/**
 * RoboForge robot material palette.
 *
 * Every robot in the product is built from these materials and nothing else, so the fleet reads as one
 * product family. The palette is strictly neutral: white, greys and graphite. There is no tinted
 * accent colour on any robot (status and selection are shown by the UI, not by painting the robot).
 */
export interface RobotMaterials {
  /** Glossy white painted shell. */
  shell: THREE.MeshPhysicalMaterial;
  /** Light grey painted panels and trim plates. */
  panel: THREE.MeshStandardMaterial;
  /** Mid grey structural parts (skirts, arms, frames). */
  trim: THREE.MeshStandardMaterial;
  /** Dark grey covers and housings. */
  dark: THREE.MeshStandardMaterial;
  /** Near-black graphite for seams, recesses and gaskets. */
  graphite: THREE.MeshStandardMaterial;
  /** Black glass for sensor windows. */
  glass: THREE.MeshPhysicalMaterial;
  /** Tyre rubber. */
  rubber: THREE.MeshStandardMaterial;
  /** Polished steel for caps, axles and rings. */
  steel: THREE.MeshStandardMaterial;
  /** Satin grey metal for utility parts such as casters and brackets. */
  satin: THREE.MeshStandardMaterial;
  dispose: () => void;
}

export function createRobotMaterials(): RobotMaterials {
  const shell = new THREE.MeshPhysicalMaterial({
    color: 0xeeeeee, roughness: 0.3, metalness: 0.0, clearcoat: 0.75, clearcoatRoughness: 0.16,
  });
  const panel = new THREE.MeshStandardMaterial({ color: 0xd0d0d0, roughness: 0.42, metalness: 0.12 });
  const trim = new THREE.MeshStandardMaterial({ color: 0x8e8e8e, roughness: 0.5, metalness: 0.35 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x4a4a4a, roughness: 0.55, metalness: 0.25 });
  const graphite = new THREE.MeshStandardMaterial({ color: 0x232323, roughness: 0.6, metalness: 0.25 });
  const glass = new THREE.MeshPhysicalMaterial({
    color: 0x0c0c0c, roughness: 0.06, metalness: 0.15, clearcoat: 1.0, clearcoatRoughness: 0.04,
  });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x1b1b1b, roughness: 0.92, metalness: 0.0 });
  const steel = new THREE.MeshStandardMaterial({ color: 0xc6c6c6, roughness: 0.28, metalness: 0.92 });
  const satin = new THREE.MeshStandardMaterial({ color: 0x6f6f6f, roughness: 0.46, metalness: 0.65 });

  return {
    shell, panel, trim, dark, graphite, glass, rubber, steel, satin,
    dispose: () => [shell, panel, trim, dark, graphite, glass, rubber, steel, satin].forEach((m) => m.dispose()),
  };
}

/** Mark every mesh under `root` as casting/receiving shadows (small instanced details opt out). */
export function enableShadows(root: THREE.Object3D): void {
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.isMesh && !(mesh as unknown as THREE.InstancedMesh).isInstancedMesh) {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }
  });
}

/** Dispose all geometries under `root` (materials are owned by the palette). */
export function disposeGeometries(root: THREE.Object3D): void {
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
  });
}
