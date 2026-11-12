import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import * as THREE from 'three';
import { gsap } from 'gsap';
import {
  Heart, Sparkles, Volume2, VolumeX, Hand,
  Cookie, RotateCw
} from 'lucide-react';
import { GoogleSignInButton } from '../components/GoogleSignInButton';
import { useAuth } from '../context/AuthContext';

const C = {
  ivory: '#F1EEE7',
  graphite: '#17191A',
  black: '#202222',
  red: '#C4472D',
  cyan: '#e8622c',
  pink: '#d6d5cf',
  muted: '#9A9A96',
};

// Procedural Web Audio API Sound Synthesizer (Instant & Zero External Files)
class RoboAudioSynth {
  private ctx: AudioContext | null = null;
  public enabled: boolean = true;

  private init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) this.ctx = new AudioCtx();
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  playHappyChirp() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(540, now);
    osc.frequency.exponentialRampToValueAtTime(880, now + 0.08);
    osc.frequency.exponentialRampToValueAtTime(1350, now + 0.2);
    gain.gain.setValueAtTime(0.12, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.26);
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start(now);
    osc.stop(now + 0.26);
  }

  playPetPurr() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(130, now);
    osc.frequency.linearRampToValueAtTime(190, now + 0.15);
    osc.frequency.linearRampToValueAtTime(140, now + 0.32);
    gain.gain.setValueAtTime(0.1, now);
    gain.gain.linearRampToValueAtTime(0.16, now + 0.15);
    gain.gain.linearRampToValueAtTime(0.001, now + 0.38);
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start(now);
    osc.stop(now + 0.38);
  }

  playTreatChomp() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'square';
    osc.frequency.setValueAtTime(320, now);
    osc.frequency.exponentialRampToValueAtTime(640, now + 0.08);
    osc.frequency.exponentialRampToValueAtTime(420, now + 0.16);
    osc.frequency.exponentialRampToValueAtTime(1150, now + 0.28);
    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.32);
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start(now);
    osc.stop(now + 0.32);
  }

  playHighFive() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(980, now);
    osc.frequency.exponentialRampToValueAtTime(1520, now + 0.14);
    gain.gain.setValueAtTime(0.14, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start(now);
    osc.stop(now + 0.3);
  }
}

const roboAudio = new RoboAudioSynth();

export type RobotMood = 'happy' | 'hearts' | 'curious' | 'wink' | 'shy' | 'excited' | 'surprised';

interface RobotBuddyViewProps {
  mood: RobotMood;
  isCoveringEyes: boolean;
  actionTrigger: { type: string; id: number } | null;
  onPetBot: () => void;
  onHighFiveBot: () => void;
  onBotClick: () => void;
}

/**
 * Procedural Real 3D Three.js Robot Companion
 * Faithfully models the chibi EVE / Astro companion robot from the reference photo:
 * - Glossy porcelain white oblate helmet shell
 * - Curved inset piano-black glass visor with glowing cyan honeycomb crescent eyes
 * - Ear cups with angled antennae rods & sphere tips
 * - Levitation neck gap with glowing cyan energy ring
 * - Tapered pear / inverted egg torso
 * - Detached floating paddle flipper arms (right arm waving, covers eyes on password)
 * - Fluid gaze tracking, idle hover, and full 3D interactive stunts
 */
function Real3DRoboBuddy({
  mood,
  isCoveringEyes,
  actionTrigger,
  onPetBot,
  onHighFiveBot,
  onBotClick
}: RobotBuddyViewProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const treatElRef = useRef<HTMLDivElement>(null);
  const [hearts, setHearts] = useState<Array<{ id: number; x: number; y: number }>>([]);

  // Refs for Three.js objects
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  // Mesh refs for animation
  const robotGroupRef = useRef<THREE.Group | null>(null);
  const headGroupRef = useRef<THREE.Group | null>(null);
  const bodyGroupRef = useRef<THREE.Group | null>(null);
  const rightArmRef = useRef<THREE.Group | null>(null);
  const leftArmRef = useRef<THREE.Group | null>(null);
  const shadowMeshRef = useRef<THREE.Mesh | null>(null);
  const neckRingRef = useRef<THREE.Mesh | null>(null);

  // Visor Canvas & Texture
  const visorCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const visorTextureRef = useRef<THREE.CanvasTexture | null>(null);

  // Cursor & Gaze State
  const mouseRef = useRef({ x: 0, y: 0, targetX: 0, targetY: 0 });
  const blinkStateRef = useRef({ isBlinking: false, nextBlinkTime: 2.5 });
  const animFrameIdRef = useRef<number | null>(null);

  // Helper to draw eye expressions on the high-res visor canvas
  const updateVisorCanvas = useCallback((
    currentMood: RobotMood,
    gazeX: number,
    gazeY: number,
    isBlink: boolean
  ) => {
    const canvas = visorCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // Clear background
    ctx.clearRect(0, 0, W, H);

    // Visor rounded stadium shape parameters
    const vx = 90;
    const vy = 80;
    const vw = W - 180;
    const vh = H - 160;
    const vr = 220; // soft curved stadium corners

    // 1. Draw Visor Bezel & Glossy Glass Screen
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(vx, vy, vw, vh, vr);
    ctx.clip();

    // Deep piano black background with subtle vertical glass gradient
    const bgGrad = ctx.createLinearGradient(0, vy, 0, vy + vh);
    bgGrad.addColorStop(0, '#101622');
    bgGrad.addColorStop(0.3, '#090d14');
    bgGrad.addColorStop(1, '#05070a');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(vx, vy, vw, vh);

    // Glossy curved specular highlight arc at top
    const specGrad = ctx.createLinearGradient(0, vy, 0, vy + 120);
    specGrad.addColorStop(0, 'rgba(255, 255, 255, 0.16)');
    specGrad.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
    ctx.fillStyle = specGrad;
    ctx.beginPath();
    ctx.ellipse(W / 2, vy + 40, vw * 0.42, 60, 0, 0, Math.PI * 2);
    ctx.fill();

    // Visor sleek inner accent border
    ctx.strokeStyle = 'rgba(232, 98, 44, 0.28)';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.restore();

    // Visor outer bezel rim
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(vx, vy, vw, vh, vr);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.restore();

    // 2. Draw Cyan Eyes
    const eyeBaseY = H / 2 + 10;
    const eyeSpacing = 160;
    const leftEyeX = W / 2 - eyeSpacing + gazeX * 28;
    const rightEyeX = W / 2 + eyeSpacing + gazeX * 28;
    const eyeY = eyeBaseY + gazeY * 18;

    // Helper to draw the honeycomb / dot grid pattern inside eyes
    const fillWithHoneycomb = (eyeCenterX: number, eyeCenterY: number) => {
      // Base vibrant gradient fill
      const grad = ctx.createLinearGradient(eyeCenterX, eyeCenterY - 60, eyeCenterX, eyeCenterY + 50);
      grad.addColorStop(0, '#e8622c');
      grad.addColorStop(1, '#e8622c');
      ctx.fillStyle = grad;
      ctx.fill();

      // Honeycomb dot matrix texture
      ctx.save();
      ctx.fillStyle = 'rgba(7, 24, 44, 0.38)';
      const hexStep = 10;
      for (let px = eyeCenterX - 90; px <= eyeCenterX + 90; px += hexStep) {
        for (let py = eyeCenterY - 70; py <= eyeCenterY + 60; py += hexStep * 0.866) {
          const rowOffset = (Math.floor(py / (hexStep * 0.866)) % 2) * (hexStep / 2);
          ctx.beginPath();
          ctx.arc(px + rowOffset, py, 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    };

    // Helper to draw a single crescent arch eye (exact match to reference photo!)
    const drawCrescentArch = (cx: number, cy: number, scaleY: number = 1) => {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(1, scaleY);
      ctx.translate(-cx, -cy);

      ctx.beginPath();
      // Outer arc
      ctx.arc(cx, cy + 34, 88, Math.PI * 1.14, Math.PI * 1.86, false);
      // Inner arc
      ctx.arc(cx, cy + 56, 76, Math.PI * 1.84, Math.PI * 1.16, true);
      ctx.closePath();

      ctx.save();
      ctx.clip();
      fillWithHoneycomb(cx, cy);
      ctx.restore();

      // Outer glow border
      ctx.strokeStyle = '#e8622c';
      ctx.lineWidth = 4;
      ctx.shadowColor = '#e8622c';
      ctx.shadowBlur = 24;
      ctx.stroke();

      ctx.restore();
    };

    if (isBlink) {
      // Natural blink: closed slit
      [leftEyeX, rightEyeX].forEach(cx => {
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(cx - 65, eyeY - 3, 130, 8, 4);
        ctx.fillStyle = '#e8622c';
        ctx.shadowColor = '#e8622c';
        ctx.shadowBlur = 18;
        ctx.fill();
        ctx.restore();
      });
    } else if (currentMood === 'happy') {
      // Default happy state: Smiling crescent arches with honeycomb
      drawCrescentArch(leftEyeX, eyeY, 1.0);
      drawCrescentArch(rightEyeX, eyeY, 1.0);
    } else if (currentMood === 'hearts') {
      // Loved / Petting state: Beating heart eyes & blush
      [leftEyeX, rightEyeX].forEach(cx => {
        ctx.save();
        ctx.translate(cx, eyeY);
        ctx.scale(1.2, 1.2);
        ctx.beginPath();
        ctx.moveTo(0, 18);
        ctx.bezierCurveTo(-38, -25, -50, -48, 0, -48);
        ctx.bezierCurveTo(50, -48, 38, -25, 0, 18);
        ctx.fillStyle = '#d6d5cf';
        ctx.shadowColor = '#d6d5cf';
        ctx.shadowBlur = 24;
        ctx.fill();
        ctx.restore();
      });

      // Blushing cheeks
      [leftEyeX - 25, rightEyeX + 25].forEach(cx => {
        const blushGrad = ctx.createRadialGradient(cx, eyeY + 75, 4, cx, eyeY + 75, 34);
        blushGrad.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
        blushGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
        ctx.fillStyle = blushGrad;
        ctx.beginPath();
        ctx.arc(cx, eyeY + 75, 34, 0, Math.PI * 2);
        ctx.fill();
      });
    } else if (currentMood === 'excited') {
      // Aerial spin / trick: Twinkling 5-pointed stars
      const drawStar = (cx: number, cy: number) => {
        ctx.save();
        ctx.translate(cx, cy);
        ctx.beginPath();
        for (let i = 0; i < 5; i++) {
          ctx.lineTo(Math.cos((18 + i * 72) * Math.PI / 180) * 44, -Math.sin((18 + i * 72) * Math.PI / 180) * 44);
          ctx.lineTo(Math.cos((54 + i * 72) * Math.PI / 180) * 20, -Math.sin((54 + i * 72) * Math.PI / 180) * 20);
        }
        ctx.closePath();
        ctx.fillStyle = '#e8622c';
        ctx.shadowColor = '#e8622c';
        ctx.shadowBlur = 28;
        ctx.fill();
        ctx.restore();
      };
      drawStar(leftEyeX, eyeY);
      drawStar(rightEyeX, eyeY);
    } else if (currentMood === 'curious') {
      // Email focus / curious: Inquisitive round eyes with tracking pupils
      [
        { cx: leftEyeX, r: 42 },
        { cx: rightEyeX, r: 35 }
      ].forEach(({ cx, r }) => {
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, eyeY, r, 0, Math.PI * 2);
        ctx.save();
        ctx.clip();
        fillWithHoneycomb(cx, eyeY);
        ctx.restore();

        ctx.strokeStyle = '#e8622c';
        ctx.lineWidth = 4;
        ctx.shadowColor = '#e8622c';
        ctx.shadowBlur = 20;
        ctx.stroke();

        // Inner pupil
        ctx.beginPath();
        ctx.arc(cx + gazeX * 12, eyeY + gazeY * 10, r * 0.44, 0, Math.PI * 2);
        ctx.fillStyle = '#081728';
        ctx.fill();
        ctx.restore();
      });
    } else if (currentMood === 'wink') {
      // Friendly wink: Left arch, right wink curve
      drawCrescentArch(leftEyeX, eyeY, 1.0);
      ctx.save();
      ctx.beginPath();
      ctx.arc(rightEyeX, eyeY + 10, 52, Math.PI * 0.2, Math.PI * 0.8, false);
      ctx.strokeStyle = '#e8622c';
      ctx.lineWidth = 10;
      ctx.lineCap = 'round';
      ctx.shadowColor = '#e8622c';
      ctx.shadowBlur = 22;
      ctx.stroke();
      ctx.restore();
    } else if (currentMood === 'shy') {
      // Shy / Password: Bashful horizontal bars
      [leftEyeX, rightEyeX].forEach(cx => {
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(cx - 55, eyeY - 8, 110, 18, 9);
        ctx.fillStyle = '#e8622c';
        ctx.shadowColor = '#e8622c';
        ctx.shadowBlur = 20;
        ctx.fill();
        ctx.restore();
      });

      // Blushing cheeks
      [leftEyeX - 20, rightEyeX + 20].forEach(cx => {
        const blushGrad = ctx.createRadialGradient(cx, eyeY + 60, 4, cx, eyeY + 60, 30);
        blushGrad.addColorStop(0, 'rgba(255, 255, 255, 0.28)');
        blushGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
        ctx.fillStyle = blushGrad;
        ctx.beginPath();
        ctx.arc(cx, eyeY + 60, 30, 0, Math.PI * 2);
        ctx.fill();
      });
    } else if (currentMood === 'surprised') {
      // Surprised: Tall round ovals
      [leftEyeX, rightEyeX].forEach(cx => {
        ctx.save();
        ctx.beginPath();
        ctx.ellipse(cx, eyeY, 34, 48, 0, 0, Math.PI * 2);
        ctx.save();
        ctx.clip();
        fillWithHoneycomb(cx, eyeY);
        ctx.restore();
        ctx.strokeStyle = '#e8622c';
        ctx.lineWidth = 5;
        ctx.shadowColor = '#e8622c';
        ctx.shadowBlur = 24;
        ctx.stroke();
        ctx.restore();
      });
    }

    if (visorTextureRef.current) {
      visorTextureRef.current.needsUpdate = true;
    }
  }, []);

  // Initialize Three.js Scene, Geometry, Materials, and Lighting
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 700;

    // 1. Scene & Camera
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera positioned with ideal FOV to show full robot comfortably centered
    const camera = new THREE.PerspectiveCamera(44, width / height, 0.1, 100);
    camera.position.set(0, 0.04, 3.3);
    camera.lookAt(0, -0.02, 0);
    cameraRef.current = camera;

    // 2. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 3. Lighting (Porcelain studio setup)
    // Ambient fill
    const ambientLight = new THREE.AmbientLight(0xf0f0ee, 0.95);
    scene.add(ambientLight);

    // Key directional light (creates glossy shine on forehead & torso)
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
    keyLight.position.set(2.4, 3.6, 3.2);
    scene.add(keyLight);

    // Fill light (soft blueish fill from left)
    const fillLight = new THREE.DirectionalLight(0xdfe4ea, 0.8);
    fillLight.position.set(-2.6, 1.2, 2.0);
    scene.add(fillLight);

    // Rim backlight (cinematic cyan glow rimming robot silhouette)
    const rimLight = new THREE.DirectionalLight(0xe8622c, 2.0);
    rimLight.position.set(0, 1.6, -2.8);
    scene.add(rimLight);

    // Neck PointLight (illuminates levitation collar gap)
    const neckLight = new THREE.PointLight(0xe8622c, 2.6, 1.6);
    neckLight.position.set(0, -0.07, 0.04);
    scene.add(neckLight);

    // 4. Materials
    // Glossy White Porcelain Shell
    const porcelainMat = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 0.14,
      metalness: 0.02,
      clearcoat: 0.88,
      clearcoatRoughness: 0.08,
      reflectivity: 0.9
    });

    // Dark Accent / Ear Mount Material
    const darkAccentMat = new THREE.MeshStandardMaterial({
      color: 0x131516,
      roughness: 0.35,
      metalness: 0.4
    });

    // Cyan Levitation Light Material
    const cyanGlowMat = new THREE.MeshBasicMaterial({
      color: 0xe8622c
    });

    // 5. Build Visor Offscreen Canvas & Texture
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 768;
    visorCanvasRef.current = canvas;

    const visorTexture = new THREE.CanvasTexture(canvas);
    visorTexture.colorSpace = THREE.SRGBColorSpace;
    visorTextureRef.current = visorTexture;

    // Draw initial eyes
    updateVisorCanvas('happy', 0, 0, false);

    // Visor Material
    const visorMat = new THREE.MeshStandardMaterial({
      map: visorTexture,
      emissiveMap: visorTexture,
      emissive: 0xe8622c,
      emissiveIntensity: 0.85,
      roughness: 0.06,
      metalness: 0.1,
      transparent: true,
      depthWrite: true
    });

    // 6. Assemble 3D Robot Hierarchy
    const robotGroup = new THREE.Group();
    robotGroupRef.current = robotGroup;
    scene.add(robotGroup);

    // ── HEAD GROUP ──────────────────────────────────────────────────────────
    const headGroup = new THREE.Group();
    headGroup.position.set(0, 0.16, 0);
    headGroupRef.current = headGroup;
    robotGroup.add(headGroup);

    // A. Outer Porcelain Helmet (Oblate spheroid wider horizontally)
    const headGeo = new THREE.SphereGeometry(0.44, 48, 36);
    headGeo.scale(1.16, 0.96, 1.04);
    const helmetMesh = new THREE.Mesh(headGeo, porcelainMat);
    headGroup.add(helmetMesh);

    // B. Curved Front Visor Screen (Bent plane hugging front of helmet shell)
    const visorW = 0.72;
    const visorH = 0.50;
    const visorGeo = new THREE.PlaneGeometry(visorW, visorH, 48, 36);
    const posAttr = visorGeo.attributes.position;
    const rx = 0.44 * 1.16;
    const ry = 0.44 * 0.96;
    const rz = 0.44 * 1.04;

    for (let i = 0; i < posAttr.count; i++) {
      const vx = posAttr.getX(i);
      const vy = posAttr.getY(i);
      const term = 1.0 - (vx / rx) * (vx / rx) - (vy / ry) * (vy / ry);
      const vz = term > 0 ? Math.sqrt(term) * rz + 0.005 : 0;
      posAttr.setZ(i, vz);
    }
    visorGeo.computeVertexNormals();

    const visorMesh = new THREE.Mesh(visorGeo, visorMat);
    visorMesh.position.set(0, 0, 0);
    headGroup.add(visorMesh);

    // C. Ear Cups & Antennae (Left & Right)
    const earDiscGeo = new THREE.CylinderGeometry(0.082, 0.082, 0.035, 32);
    const antennaStemGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.38, 16);
    antennaStemGeo.translate(0, 0.19, 0); // pivot at base
    const antennaTipGeo = new THREE.SphereGeometry(0.022, 16, 16);

    // Left Ear Mount & Antenna (Viewer's left)
    const leftEar = new THREE.Mesh(earDiscGeo, porcelainMat);
    leftEar.position.set(-0.495, 0.02, 0);
    leftEar.rotation.z = -Math.PI / 2;
    headGroup.add(leftEar);

    const leftEarInner = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.038, 24), darkAccentMat);
    leftEar.add(leftEarInner);

    const leftAntennaGroup = new THREE.Group();
    leftAntennaGroup.position.set(-0.49, 0.06, 0);
    leftAntennaGroup.rotation.z = 0.44; // angled outward ~25°
    leftAntennaGroup.rotation.x = -0.06;
    const leftStem = new THREE.Mesh(antennaStemGeo, porcelainMat);
    const leftTip = new THREE.Mesh(antennaTipGeo, porcelainMat);
    leftTip.position.set(0, 0.38, 0);
    leftAntennaGroup.add(leftStem, leftTip);
    headGroup.add(leftAntennaGroup);

    // Right Ear Mount & Antenna (Viewer's right)
    const rightEar = new THREE.Mesh(earDiscGeo, porcelainMat);
    rightEar.position.set(0.495, 0.02, 0);
    rightEar.rotation.z = Math.PI / 2;
    headGroup.add(rightEar);

    const rightEarInner = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.038, 24), darkAccentMat);
    rightEar.add(rightEarInner);

    const rightAntennaGroup = new THREE.Group();
    rightAntennaGroup.position.set(0.49, 0.06, 0);
    rightAntennaGroup.rotation.z = -0.44; // angled outward ~25°
    rightAntennaGroup.rotation.x = -0.06;
    const rightStem = new THREE.Mesh(antennaStemGeo, porcelainMat);
    const rightTip = new THREE.Mesh(antennaTipGeo, porcelainMat);
    rightTip.position.set(0, 0.38, 0);
    rightAntennaGroup.add(rightStem, rightTip);
    headGroup.add(rightAntennaGroup);

    // ── NECK LEVITATION GAP ──────────────────────────────────────────────────
    const neckRingGeo = new THREE.TorusGeometry(0.11, 0.016, 16, 32);
    neckRingGeo.rotateX(Math.PI / 2);
    const neckRing = new THREE.Mesh(neckRingGeo, cyanGlowMat);
    neckRing.position.set(0, -0.07, 0);
    neckRingRef.current = neckRing;
    robotGroup.add(neckRing);

    // ── BODY / TORSO ────────────────────────────────────────────────────────
    // Tapered pear / egg / teardrop profile matching reference photo
    const bodyGroup = new THREE.Group();
    bodyGroupRef.current = bodyGroup;
    robotGroup.add(bodyGroup);

    const bodyPoints: THREE.Vector2[] = [
      new THREE.Vector2(0.001, -0.68), // bottom rounded tip
      new THREE.Vector2(0.06, -0.66),
      new THREE.Vector2(0.12, -0.62),
      new THREE.Vector2(0.18, -0.55),
      new THREE.Vector2(0.23, -0.46),
      new THREE.Vector2(0.265, -0.34), // widest belly point
      new THREE.Vector2(0.26, -0.24),
      new THREE.Vector2(0.23, -0.16),
      new THREE.Vector2(0.17, -0.11),
      new THREE.Vector2(0.14, -0.095), // top collar rim
      new THREE.Vector2(0.08, -0.11),  // inner collar cavity
      new THREE.Vector2(0.001, -0.12)
    ];
    const bodyGeo = new THREE.LatheGeometry(bodyPoints, 48);
    bodyGeo.computeVertexNormals();
    const bodyMesh = new THREE.Mesh(bodyGeo, porcelainMat);
    bodyGroup.add(bodyMesh);

    // Subtle horizontal panel seam line on lower belly
    const seamRingGeo = new THREE.TorusGeometry(0.248, 0.004, 8, 48);
    seamRingGeo.rotateX(Math.PI / 2);
    const seamRing = new THREE.Mesh(seamRingGeo, darkAccentMat);
    seamRing.position.set(0, -0.40, 0);
    bodyGroup.add(seamRing);

    // ── FLOATING DETACHED PADDLE ARMS ───────────────────────────────────────
    // Build rounded paddle / flipper geometry
    const armShape = new THREE.Shape();
    armShape.moveTo(-0.038, 0.16);
    armShape.lineTo(0.038, 0.16);
    armShape.quadraticCurveTo(0.075, 0.02, 0.07, -0.12);
    armShape.quadraticCurveTo(0.055, -0.22, 0.0, -0.24); // rounded paddle tip
    armShape.quadraticCurveTo(-0.055, -0.22, -0.07, -0.12);
    armShape.quadraticCurveTo(-0.075, 0.02, -0.038, 0.16);

    const armExtrudeSettings = {
      depth: 0.028,
      bevelEnabled: true,
      bevelSegments: 8,
      steps: 1,
      bevelSize: 0.02,
      bevelThickness: 0.02
    };
    const armGeo = new THREE.ExtrudeGeometry(armShape, armExtrudeSettings);
    armGeo.center();

    // Right Arm (Viewer's left: Raised up and waving cheerfully!)
    const rightArmGroup = new THREE.Group();
    rightArmRef.current = rightArmGroup;
    rightArmGroup.position.set(-0.41, -0.24, 0.05);
    rightArmGroup.rotation.set(-0.15, 0.10, -0.68);
    const rightArmMesh = new THREE.Mesh(armGeo, porcelainMat);
    rightArmGroup.add(rightArmMesh);
    robotGroup.add(rightArmGroup);

    // Left Arm (Viewer's right: Relaxed hover at side)
    const leftArmGroup = new THREE.Group();
    leftArmRef.current = leftArmGroup;
    leftArmGroup.position.set(0.40, -0.34, 0.02);
    leftArmGroup.rotation.set(0.05, -0.10, 0.48);
    const leftArmMesh = new THREE.Mesh(armGeo, porcelainMat);
    leftArmGroup.add(leftArmMesh);
    robotGroup.add(leftArmGroup);

    // ── GROUND SHADOW DISC ──────────────────────────────────────────────────
    const shadowCanvas = document.createElement('canvas');
    shadowCanvas.width = 256;
    shadowCanvas.height = 256;
    const shadowCtx = shadowCanvas.getContext('2d');
    if (shadowCtx) {
      const grad = shadowCtx.createRadialGradient(128, 128, 10, 128, 128, 120);
      grad.addColorStop(0, 'rgba(0, 0, 0, 0.65)');
      grad.addColorStop(0.45, 'rgba(0, 0, 0, 0.28)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0.0)');
      shadowCtx.fillStyle = grad;
      shadowCtx.fillRect(0, 0, 256, 256);
    }
    const shadowTexture = new THREE.CanvasTexture(shadowCanvas);
    const shadowGeo = new THREE.PlaneGeometry(1.3, 1.3);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      map: shadowTexture,
      transparent: true,
      depthWrite: false
    });
    const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
    shadowMesh.position.set(0, -0.98, 0);
    shadowMeshRef.current = shadowMesh;
    scene.add(shadowMesh);

    // 7. Responsive Resize Handler
    const handleResize = () => {
      if (!container || !camera || !renderer) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // 8. Animation & Render Loop
    let clock = new THREE.Clock();

    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);

      const elapsed = clock.getElapsedTime();
      const delta = clock.getDelta();

      // Smooth cursor lerp
      mouseRef.current.x += (mouseRef.current.targetX - mouseRef.current.x) * 0.08;
      mouseRef.current.y += (mouseRef.current.targetY - mouseRef.current.y) * 0.08;

      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;

      // Organic idle hover bobbing
      const hoverY = Math.sin(elapsed * 2.4) * 0.024;
      if (robotGroupRef.current) {
        robotGroupRef.current.position.y = hoverY;
      }

      // Shadow scales inversely with hover height
      if (shadowMeshRef.current) {
        const shadowScale = 1.0 - hoverY * 2.2;
        shadowMeshRef.current.scale.set(shadowScale, shadowScale, shadowScale);
      }

      // Neck ring subtle pulse
      if (neckRingRef.current) {
        const ringScale = 1.0 + Math.sin(elapsed * 4.5) * 0.04;
        neckRingRef.current.scale.set(ringScale, ringScale, ringScale);
      }

      // Head tracks mouse cursor (gentle realistic pitch/yaw)
      if (headGroupRef.current) {
        headGroupRef.current.rotation.y = mx * 0.28; // ±16° max
        headGroupRef.current.rotation.x = -my * 0.16; // ±10° max
        headGroupRef.current.rotation.z = -mx * 0.06;
      }

      // Body subtle follow
      if (bodyGroupRef.current) {
        bodyGroupRef.current.rotation.y = mx * 0.12;
        bodyGroupRef.current.rotation.x = -my * 0.06;
      }

      // Natural wave animation on right arm when not covering eyes
      if (rightArmRef.current && !isCoveringEyes) {
        rightArmRef.current.rotation.z = -0.68 + Math.sin(elapsed * 3.6) * 0.12;
        rightArmRef.current.rotation.y = 0.10 + Math.cos(elapsed * 2.8) * 0.06;
      }

      // Natural float drift on left arm
      if (leftArmRef.current && !isCoveringEyes) {
        leftArmRef.current.rotation.z = 0.48 + Math.cos(elapsed * 2.6) * 0.06;
        leftArmRef.current.position.y = -0.34 + Math.sin(elapsed * 2.2) * 0.012;
      }

      // Natural blinking logic (every ~3.8 to 5 seconds)
      blinkStateRef.current.nextBlinkTime -= 0.016;
      if (blinkStateRef.current.nextBlinkTime <= 0) {
        if (!blinkStateRef.current.isBlinking) {
          blinkStateRef.current.isBlinking = true;
          updateVisorCanvas(mood, mx, my, true);
          setTimeout(() => {
            blinkStateRef.current.isBlinking = false;
            blinkStateRef.current.nextBlinkTime = 3.2 + Math.random() * 2.4;
            updateVisorCanvas(mood, mx, my, false);
          }, 130);
        }
      }

      renderer.render(scene, camera);
    };

    animate();

    // Cleanup on unmount
    return () => {
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      headGeo.dispose();
      visorGeo.dispose();
      bodyGeo.dispose();
      armGeo.dispose();
      earDiscGeo.dispose();
      antennaStemGeo.dispose();
      antennaTipGeo.dispose();
      neckRingGeo.dispose();
      porcelainMat.dispose();
      visorMat.dispose();
      darkAccentMat.dispose();
      cyanGlowMat.dispose();
      shadowMat.dispose();
    };
  }, []);

  // Update Visor Eyes when mood changes or cursor moves
  useEffect(() => {
    updateVisorCanvas(mood, mouseRef.current.x, mouseRef.current.y, false);
  }, [mood, updateVisorCanvas]);

  // Responsive Mouse Movement Tracking
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const container = mountRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();

      // Normalize mouse to [-1, 1] relative to viewport center
      const relX = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
      const relY = ((e.clientY - rect.top) / rect.height - 0.5) * 2;

      // Clamp target within gentle tracking range
      mouseRef.current.targetX = Math.max(-0.9, Math.min(0.9, relX));
      mouseRef.current.targetY = Math.max(-0.7, Math.min(0.7, relY));

      // Refresh visor eye gaze
      if (!blinkStateRef.current.isBlinking) {
        updateVisorCanvas(mood, mouseRef.current.targetX, mouseRef.current.targetY, false);
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [mood, updateVisorCanvas]);

  // Animate Password Peek (Arms Cover Visor "No Peeking!")
  useEffect(() => {
    const rightArm = rightArmRef.current;
    const leftArm = leftArmRef.current;
    if (!rightArm || !leftArm) return;

    if (isCoveringEyes) {
      // Both arms float up right in front of the visor!
      gsap.to(rightArm.position, {
        x: -0.16,
        y: 0.12,
        z: 0.44,
        duration: 0.38,
        ease: 'power2.out'
      });
      gsap.to(rightArm.rotation, {
        x: -0.5,
        y: 0.22,
        z: -0.42,
        duration: 0.38,
        ease: 'power2.out'
      });

      gsap.to(leftArm.position, {
        x: 0.16,
        y: 0.12,
        z: 0.44,
        duration: 0.38,
        ease: 'power2.out'
      });
      gsap.to(leftArm.rotation, {
        x: -0.5,
        y: -0.22,
        z: 0.42,
        duration: 0.38,
        ease: 'power2.out'
      });
    } else {
      // Return arms to cheerful wave / hover positions
      gsap.to(rightArm.position, {
        x: -0.41,
        y: -0.24,
        z: 0.05,
        duration: 0.42,
        ease: 'back.out(1.4)'
      });
      gsap.to(rightArm.rotation, {
        x: -0.15,
        y: 0.10,
        z: -0.68,
        duration: 0.42,
        ease: 'back.out(1.4)'
      });

      gsap.to(leftArm.position, {
        x: 0.40,
        y: -0.34,
        z: 0.02,
        duration: 0.42,
        ease: 'back.out(1.4)'
      });
      gsap.to(leftArm.rotation, {
        x: 0.05,
        y: -0.10,
        z: 0.48,
        duration: 0.42,
        ease: 'back.out(1.4)'
      });
    }
  }, [isCoveringEyes]);

  // Handle action triggers (flip, spin, highfive, treat)
  useEffect(() => {
    if (!actionTrigger) return;
    const { type } = actionTrigger;
    const robot = robotGroupRef.current;
    const head = headGroupRef.current;
    const rightArm = rightArmRef.current;
    if (!robot) return;

    if (type === 'flip') {
      // 360 aerial somersault flip
      gsap.to(robot.rotation, {
        x: Math.PI * 2,
        duration: 0.9,
        ease: 'power2.inOut',
        onComplete: () => {
          robot.rotation.x = 0;
        }
      });
      gsap.to(robot.position, {
        y: 0.32,
        duration: 0.45,
        yoyo: true,
        repeat: 1,
        ease: 'power1.out'
      });
      roboAudio.playHappyChirp();
    } else if (type === 'spin') {
      // 360 pirouette spin
      gsap.to(robot.rotation, {
        y: Math.PI * 2,
        duration: 0.85,
        ease: 'power2.inOut',
        onComplete: () => {
          robot.rotation.y = 0;
        }
      });
      roboAudio.playHappyChirp();
    } else if (type === 'highfive' && rightArm) {
      // Reach forward in a high-five gesture!
      gsap.to(rightArm.position, {
        z: 0.38,
        y: -0.12,
        duration: 0.22,
        yoyo: true,
        repeat: 1,
        ease: 'power2.out'
      });
      gsap.to(rightArm.rotation, {
        x: -0.85,
        z: -0.35,
        duration: 0.22,
        yoyo: true,
        repeat: 1,
        ease: 'power2.out'
      });
      roboAudio.playHighFive();
    } else if (type === 'treat') {
      const treatEl = treatElRef.current;
      if (treatEl) {
        treatEl.style.display = 'block';
        gsap.fromTo(
          treatEl,
          { opacity: 1, x: 140, y: -80, scale: 1.4 },
          {
            x: 0,
            y: 0,
            scale: 0.2,
            opacity: 0,
            duration: 0.65,
            ease: 'power2.in',
            onComplete: () => {
              treatEl.style.display = 'none';
              roboAudio.playTreatChomp();
              // Happy satisfied bounce
              if (head) {
                gsap.to(head.position, { y: 0.22, duration: 0.14, yoyo: true, repeat: 2 });
              }
            }
          }
        );
      }
    }
  }, [actionTrigger]);

  // Spawn floating DOM heart particle
  const spawnHeart = () => {
    const newHeart = {
      id: Date.now() + Math.random(),
      x: (Math.random() - 0.5) * 140,
      y: (Math.random() - 0.5) * 80
    };
    setHearts(prev => [...prev.slice(-10), newHeart]);
    setTimeout(() => {
      setHearts(prev => prev.filter(h => h.id !== newHeart.id));
    }, 1800);
  };

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        overflow: 'hidden'
      }}
      onClick={() => {
        onBotClick();
        spawnHeart();
      }}
      onMouseEnter={() => {
        onPetBot();
        spawnHeart();
      }}
      title="Click or pet your 3D RoboBuddy!"
    >
      {/* Soft Ambient Cyan Backlight Halo */}
      <div
        style={{
          position: 'absolute',
          width: '420px',
          height: '420px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(232, 98, 44, 0.07) 0%, rgba(232, 98, 44, 0.02) 55%, transparent 75%)',
          filter: 'blur(36px)',
          pointerEvents: 'none'
        }}
      />

      {/* Floating 3D Hearts on Petting */}
      {hearts.map(h => (
        <div
          key={h.id}
          style={{
            position: 'absolute',
            left: `calc(50% + ${h.x}px)`,
            top: `calc(45% + ${h.y}px)`,
            fontSize: '22px',
            color: '#d6d5cf',
            pointerEvents: 'none',
            zIndex: 30,
            animation: 'float-up 1.6s cubic-bezier(0.2, 0.8, 0.3, 1) forwards'
          }}
        >
          💖
        </div>
      ))}

      {/* Flying 3D Treat Snack Element */}
      <div
        ref={treatElRef}
        style={{
          position: 'absolute',
          width: '44px',
          height: '44px',
          borderRadius: '50%',
          background: 'radial-gradient(circle at 35% 35%, #fef08a 0%, #f59e0b 60%, #b45309 100%)',
          boxShadow: '0 0 24px #fbbf24, inset 0 0 8px #fef08a',
          display: 'none',
          zIndex: 40,
          pointerEvents: 'none'
        }}
      >
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px' }}>
          ⚡
        </div>
      </div>

      {/* Main Real Three.js Canvas Mount */}
      <div
        ref={mountRef}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative'
        }}
      />

      <style>{`
        @keyframes float-up {
          0% { transform: translateY(0) scale(0.6); opacity: 1; }
          100% { transform: translateY(-75px) scale(1.2); opacity: 0; }
        }
      `}</style>
    </div>
  );
}

type FormState = 'idle' | 'submitting' | 'error' | 'success';

export function LoginPage() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [state, setState] = useState<FormState>('idle');
  const [errorMsg, setErrorMsg] = useState('');

  // Interactive Robot Buddy State
  const [robotMood, setRobotMood] = useState<RobotMood>('happy');
  const [isCoveringEyes, setIsCoveringEyes] = useState(false);
  const [speechBubble, setSpeechBubble] = useState<string>("Hi! I'm your RoboBuddy. Pet me, play trick, or let's sign in!");
  const [actionTrigger, setActionTrigger] = useState<{ type: string; id: number } | null>(null);
  const [soundOn, setSoundOn] = useState(true);
  const [petCount, setPetCount] = useState(0);

  // Sync sound
  useEffect(() => {
    roboAudio.enabled = soundOn;
  }, [soundOn]);

  const triggerAction = (type: string, message: string, moodOverride?: RobotMood) => {
    setActionTrigger({ type, id: Date.now() });
    setSpeechBubble(message);
    if (moodOverride) {
      setRobotMood(moodOverride);
      setTimeout(() => setRobotMood('happy'), 2600);
    }
  };

  const handlePet = () => {
    setPetCount(c => c + 1);
    setRobotMood('hearts');
    setSpeechBubble("Aww, that feels so nice! ❤️ Happy to see you!");
    roboAudio.playPetPurr();
    setTimeout(() => setRobotMood('happy'), 2600);
  };

  const handlePlayCatch = () => {
    triggerAction('flip', "Wheee! 360 aerial flip! Look at me fly!", 'excited');
  };

  const handleFeedTreat = () => {
    triggerAction('treat', "Nom nom! Supercharged with fresh energy! ⚡", 'hearts');
  };

  const handleHighFive = () => {
    triggerAction('highfive', "High five! We make an awesome robotics team! ✋", 'excited');
  };

  const handleBotClick = () => {
    triggerAction('spin', "Hehe! Ready to explore robotics together?", 'wink');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setErrorMsg('Please enter your email and password.');
      setState('error');
      setRobotMood('curious');
      setSpeechBubble("Oops! Looks like an email or password is missing.");
      return;
    }
    setState('submitting');
    setRobotMood('excited');
    setSpeechBubble("Verifying credentials with database...");
    roboAudio.playHappyChirp();

    try {
      await login(email, password);
      setState('success');
      setRobotMood('hearts');
      setSpeechBubble("Access granted! Opening 3D Robotics Studio...");
      triggerAction('flip', "Welcome back! Entering Studio! 🚀");
      setTimeout(() => navigate('/studio'), 900);
    } catch (err: any) {
      setState('error');
      const errText = err.message || 'Invalid email or password.';
      setErrorMsg(errText);
      setRobotMood('shy');
      setSpeechBubble(errText);
    }
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', fontFamily: 'Manrope, sans-serif' }}>
      {/* ── Left — Real 3D Companion Robot (Sized & Centered with Breathing Room) ── */}
      <div style={{
        flex: '0 0 60%',
        background: 'radial-gradient(circle at 48% 44%, #1b1e20 0%, #131516 70%, #0f1112 100%)',
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}>
        {/* Real Three.js 3D Companion Robot */}
        <Real3DRoboBuddy
          mood={robotMood}
          isCoveringEyes={isCoveringEyes}
          actionTrigger={actionTrigger}
          onPetBot={handlePet}
          onHighFiveBot={handleHighFive}
          onBotClick={handleBotClick}
        />

        {/* Brand Wordmark & Sound Toggle */}
        <div style={{
          position: 'absolute',
          top: '28px',
          left: '36px',
          right: '36px',
          zIndex: 20,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <Link to="/" style={{ textDecoration: 'none' }}>
            <span style={{
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '13px',
              letterSpacing: '0.22em',
              color: '#e8622c',
              fontWeight: 700,
            }}>
              ROBOFORGE
            </span>
          </Link>

          <button
            onClick={() => setSoundOn(s => !s)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'rgba(255, 255, 255, 0.08)',
              border: '1px solid rgba(255, 255, 255, 0.16)',
              borderRadius: 20,
              padding: '4px 12px',
              color: soundOn ? '#e8622c' : '#a9aca7',
              fontSize: 11,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            title={soundOn ? 'Mute Robot Sound' : 'Enable Robot Sound'}
          >
            {soundOn ? <Volume2 size={13} /> : <VolumeX size={13} />}
            <span>{soundOn ? 'Sound ON' : 'Muted'}</span>
          </button>
        </div>

        {/* Speech Bubble Floating Above Robot */}
        <div style={{
          position: 'absolute',
          top: '76px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 20,
          backgroundColor: 'rgba(19, 21, 22, 0.88)',
          border: '1px solid rgba(232, 98, 44, 0.4)',
          borderRadius: 24,
          padding: '8px 20px',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.45)',
          maxWidth: '85%',
          textAlign: 'center'
        }}>
          <Sparkles size={14} color="#e8622c" />
          <span style={{
            fontSize: 12,
            color: '#f4f3ef',
            fontWeight: 500,
            letterSpacing: '0.01em'
          }}>
            {speechBubble}
          </span>
        </div>

        {/* Interactive Action Bar (Pet, Play, Feed, High-Five) */}
        <div style={{
          position: 'absolute',
          bottom: '32px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 20,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          backgroundColor: 'rgba(19, 21, 22, 0.82)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: 30,
          padding: '6px 12px',
          boxShadow: '0 12px 32px rgba(0, 0, 0, 0.55)'
        }}>
          <button
            onClick={handlePet}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'rgba(255, 255, 255, 0.06)',
              border: '1px solid rgba(255, 255, 255, 0.14)',
              color: '#d6d5cf',
              padding: '6px 12px',
              borderRadius: 20,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            title="Stroke and pet RoboBuddy"
          >
            <Heart size={13} fill="currentColor" />
            <span>Pet ({petCount})</span>
          </button>

          <button
            onClick={handlePlayCatch}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'rgba(232, 98, 44, 0.15)',
              border: '1px solid rgba(232, 98, 44, 0.35)',
              color: '#e8622c',
              padding: '6px 12px',
              borderRadius: 20,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            title="Play an aerial spin trick"
          >
            <RotateCw size={13} />
            <span>Play Trick</span>
          </button>

          <button
            onClick={handleFeedTreat}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'rgba(245, 158, 11, 0.15)',
              border: '1px solid rgba(245, 158, 11, 0.35)',
              color: '#fbbf24',
              padding: '6px 12px',
              borderRadius: 20,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            title="Feed energy snack / treat"
          >
            <Cookie size={13} />
            <span>Feed Treat</span>
          </button>

          <button
            onClick={handleHighFive}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'rgba(255, 255, 255, 0.06)',
              border: '1px solid rgba(255, 255, 255, 0.14)',
              color: '#d6d5cf',
              padding: '6px 12px',
              borderRadius: 20,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            title="Give RoboBuddy a high five"
          >
            <Hand size={13} />
            <span>High Five</span>
          </button>
        </div>

        {/* Ambient Cyan Accent Bar */}
        <div style={{
          position: 'absolute',
          left: 0,
          top: '50%',
          width: '3px',
          height: '90px',
          background: '#e8622c',
          transform: 'translateY(-50%)',
          borderRadius: '0 2px 2px 0',
          
        }} />
      </div>

      {/* ── Right — Login Form ───────────────────────────────────────────── */}
      <div style={{
        flex: '0 0 40%',
        background: C.ivory,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '60px 48px',
      }}>
        <div style={{ width: '100%', maxWidth: '360px' }}>
          {/* Header */}
          <div style={{ marginBottom: '40px' }}>
            <h1 style={{
              fontSize: '32px',
              fontWeight: 800,
              letterSpacing: '-0.025em',
              margin: '0 0 8px',
              color: C.black,
            }}>
              Welcome back.
            </h1>
            <p style={{ fontSize: '14px', color: C.black, opacity: 0.5, margin: 0 }}>
              Sign in to your RoboForge workspace.
            </p>
          </div>

          {/* Google One-Click Sign In */}
          <div style={{ marginBottom: '18px' }}>
            <GoogleSignInButton
              text="signin_with"
              onSuccess={() => {
                setState('success');
                setRobotMood('excited');
                setSpeechBubble('Google Authenticated! Opening your workspace...');
                roboAudio.playHappyChirp();
                setTimeout(() => navigate('/studio'), 800);
              }}
              onError={(err) => {
                setErrorMsg(err);
                setState('error');
              }}
            />
            <div style={{ display: 'flex', alignItems: 'center', margin: '16px 0 2px', gap: '10px' }}>
              <div style={{ flex: 1, height: '1px', background: 'rgba(32,34,34,0.12)' }} />
              <span style={{ fontSize: '11px', fontFamily: 'IBM Plex Mono, monospace', color: C.muted, letterSpacing: '0.1em' }}>OR EMAIL</span>
              <div style={{ flex: 1, height: '1px', background: 'rgba(32,34,34,0.12)' }} />
            </div>
          </div>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            {/* Email Field (Bot looks curious) */}
            <div>
              <label style={{
                display: 'block',
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '10px',
                letterSpacing: '0.15em',
                color: C.muted,
                marginBottom: '8px',
              }}>
                EMAIL
              </label>
              <input
                type="email"
                value={email}
                onFocus={() => {
                  setRobotMood('curious');
                  setSpeechBubble("Typing your email! I'm watching right here 🔍");
                }}
                onBlur={() => setRobotMood('happy')}
                onChange={e => { setEmail(e.target.value); setState('idle'); }}
                placeholder="you@example.com"
                style={{
                  width: '100%',
                  padding: '12px 14px',
                  boxSizing: 'border-box',
                  border: `1px solid ${state === 'error' ? C.red : 'rgba(32,34,34,0.15)'}`,
                  borderRadius: '3px',
                  background: 'white',
                  fontFamily: 'Manrope, sans-serif',
                  fontSize: '14px',
                  color: C.black,
                  outline: 'none',
                  transition: 'border-color 0.2s',
                }}
              />
            </div>

            {/* Password Field (Bot covers its eyes!) */}
            <div>
              <label style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '10px',
                letterSpacing: '0.15em',
                color: C.muted,
                marginBottom: '8px',
              }}>
                <span>PASSWORD</span>
                {isCoveringEyes && (
                  <span style={{ color: '#e8622c', fontWeight: 600 }}>🙈 NO PEEKING!</span>
                )}
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showPw ? 'text' : 'password'}
                  value={password}
                  onFocus={() => {
                    setIsCoveringEyes(true);
                    setRobotMood('shy');
                    setSpeechBubble("No peeking! Your password is 100% confidential 🙈");
                  }}
                  onBlur={() => {
                    setIsCoveringEyes(false);
                    setRobotMood('happy');
                    setSpeechBubble("All clear! Ready to log in when you are!");
                  }}
                  onChange={e => { setPassword(e.target.value); setState('idle'); }}
                  placeholder="••••••••"
                  style={{
                    width: '100%',
                    padding: '12px 44px 12px 14px',
                    boxSizing: 'border-box',
                    border: `1px solid ${state === 'error' ? C.red : 'rgba(32,34,34,0.15)'}`,
                    borderRadius: '3px',
                    background: 'white',
                    fontFamily: 'Manrope, sans-serif',
                    fontSize: '14px',
                    color: C.black,
                    outline: 'none',
                    transition: 'border-color 0.2s',
                  }}
                />
                <button
                  type="button"
                  onClick={() => {
                    setShowPw(s => !s);
                    if (!showPw) {
                      setRobotMood('surprised');
                      setSpeechBubble("Whoa, peek-a-boo! 👀");
                    }
                  }}
                  style={{
                    position: 'absolute',
                    right: '12px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    fontFamily: 'IBM Plex Mono, monospace',
                    fontSize: '9px',
                    letterSpacing: '0.12em',
                    color: C.muted,
                  }}
                >
                  {showPw ? 'HIDE' : 'SHOW'}
                </button>
              </div>
            </div>

            {/* Error message */}
            {state === 'error' && (
              <div style={{
                background: 'rgba(196,71,45,0.08)',
                border: `1px solid rgba(196,71,45,0.25)`,
                borderRadius: '3px',
                padding: '10px 14px',
                fontFamily: 'IBM Plex Mono, monospace',
                fontSize: '11px',
                color: C.red,
                letterSpacing: '0.06em',
              }}>
                {errorMsg || 'Invalid credentials. Please try again.'}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={state === 'submitting' || state === 'success'}
              style={{
                background: state === 'success' ? '#2a7a42' : C.black,
                color: C.ivory,
                padding: '14px',
                borderRadius: '3px',
                border: 'none',
                cursor: state === 'submitting' ? 'wait' : 'pointer',
                fontFamily: 'Manrope, sans-serif',
                fontWeight: 700,
                fontSize: '14px',
                letterSpacing: '0.04em',
                transition: 'background 0.3s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
            >
              {state === 'submitting' && (
                <span style={{
                  width: '14px',
                  height: '14px',
                  border: '2px solid rgba(241,238,231,0.3)',
                  borderTopColor: C.ivory,
                  borderRadius: '50%',
                  animation: 'spin 0.7s linear infinite',
                  display: 'inline-block',
                }} />
              )}
              {state === 'success' ? 'Authenticated — Opening Studio' : state === 'submitting' ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          {/* Links */}
          <div style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: C.muted, cursor: 'pointer' }}>
              Forgot password?
            </span>
            <div style={{ width: '100%', height: '1px', background: 'rgba(32,34,34,0.1)' }} />
            <div style={{ fontSize: '13px', color: C.black, opacity: 0.55 }}>
              Don't have an account?{' '}
              <Link to="/signup" style={{ color: C.red, textDecoration: 'none', fontWeight: 600 }}>
                Create one
              </Link>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        input:focus { border-color: rgba(32,34,34,0.45) !important; box-shadow: 0 0 0 3px rgba(232, 98, 44, 0.15); }
      `}</style>
    </div>
  );
}
