import type * as THREE from 'three';
import { buildHumanoidLinks } from './humanoidModel';
import { buildDogLinks } from './dogModel';

/**
 * Designed per-link visuals for the articulated presets. The studio asks for these by link name; robots that
 * are not presets (imported URDFs, the built-in arm) return null and use the generic styling instead.
 * Detection is by link names, so it keeps working if a user renames the robot.
 */
export interface DesignedVisuals {
  get: (linkName: string) => THREE.Group | null;
}

export function designedLinkVisuals(linkNames: string[]): DesignedVisuals | null {
  const names = new Set(linkNames);
  let built: { links: Record<string, THREE.Group> } | null = null;
  if (names.has('pelvis') && names.has('chest') && names.has('l_thigh')) built = buildHumanoidLinks();
  else if (names.has('body') && names.has('fl_thigh') && names.has('rr_shin')) built = buildDogLinks();
  if (!built) return null;
  const links = built.links;
  return { get: (n) => links[n] ?? null };
}
