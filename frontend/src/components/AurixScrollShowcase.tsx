import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Play, Pause, RotateCcw, Shield, Zap, Cpu, Sparkles, ChevronDown, CheckCircle2 } from 'lucide-react';

interface AurixScrollShowcaseProps {
  onExploreStudio?: () => void;
}

export function AurixScrollShowcase({ onExploreStudio }: AurixScrollShowcaseProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [images, setImages] = useState<HTMLImageElement[]>([]);
  const [loadProgress, setLoadProgress] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const frameRef = useRef(0);
  const deltaAcc = useRef(0);
  const touchStartY = useRef<number | null>(null);
  const animIntervalRef = useRef<number | null>(null);

  const TOTAL_FRAMES = 50;

  // Preload all 50 sequential frames
  useEffect(() => {
    let loadedCount = 0;
    const loadedImages: HTMLImageElement[] = [];

    for (let i = 1; i <= TOTAL_FRAMES; i++) {
      const img = new Image();
      const frameNum = String(i).padStart(3, '0');
      img.src = `/assets/aurix/frames/ezgif-frame-${frameNum}.jpg`;

      img.onload = () => {
        loadedCount++;
        setLoadProgress(Math.round((loadedCount / TOTAL_FRAMES) * 100));
        if (loadedCount === TOTAL_FRAMES) {
          setImages(loadedImages);
          setIsLoaded(true);
        }
      };

      img.onerror = () => {
        loadedCount++;
        if (loadedCount === TOTAL_FRAMES) {
          setImages(loadedImages);
          setIsLoaded(true);
        }
      };

      loadedImages[i - 1] = img;
    }
  }, []);

  // Draw current frame onto canvas preserving 16:9 full-view aspect ratio
  const drawFrame = useCallback((frameIdx: number) => {
    const canvas = canvasRef.current;
    if (!canvas || !images[frameIdx]) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = images[frameIdx];
    if (!img.complete || img.naturalWidth === 0) return;

    const cw = canvas.width;
    const ch = canvas.height;

    // Studio dark backdrop
    ctx.fillStyle = '#0b0f14';
    ctx.fillRect(0, 0, cw, ch);

    // Aspect-ratio contain calculation
    const imgAspect = img.naturalWidth / img.naturalHeight;
    const canvasAspect = cw / ch;

    let drawW: number;
    let drawH: number;
    let drawX: number;
    let drawY: number;

    if (imgAspect > canvasAspect) {
      drawW = cw;
      drawH = cw / imgAspect;
      drawX = 0;
      drawY = (ch - drawH) / 2;
    } else {
      drawH = ch;
      drawW = ch * imgAspect;
      drawX = (cw - drawW) / 2;
      drawY = 0;
    }

    ctx.drawImage(img, drawX, drawY, drawW, drawH);
  }, [images]);

  // Redraw when currentFrame or images change
  useEffect(() => {
    if (isLoaded) {
      drawFrame(currentFrame);
    }
  }, [currentFrame, isLoaded, drawFrame]);

  // Responsive high-DPI canvas resizing
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);

      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      if (isLoaded) {
        drawFrame(currentFrame);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isLoaded, currentFrame, drawFrame]);

  // Helper to update frame and keep ref in sync
  const setFrameSafe = (newFrame: number) => {
    const clamped = Math.max(0, Math.min(TOTAL_FRAMES - 1, newFrame));
    frameRef.current = clamped;
    setCurrentFrame(clamped);
  };

  /**
   * High-Precision Wheel & Scroll Lock:
   * When this section is docked in view:
   * 1. If scrolling DOWN and robot is NOT yet fully detached (frame < 49):
   *    -> Prevent default! Page DOES NOT MOVE! Only robot scrubs forward.
   * 2. If scrolling DOWN and robot IS fully detached (frame === 49):
   *    -> Allow natural page scroll downward to Mechanism Catalog!
   * 3. If scrolling UP and robot is NOT yet fully enclosed (frame > 0):
   *    -> Prevent default! Page DOES NOT MOVE! Only robot scrubs backward.
   * 4. If scrolling UP and robot IS fully enclosed (frame === 0):
   *    -> Allow natural page scroll upward to Platform section!
   */
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onWheel = (e: WheelEvent) => {
      if (isPlaying) return; // let user scroll freely during auto-loop
      const rect = container.getBoundingClientRect();

      // Check if container occupies the active viewport
      const isVisibleInView = rect.top <= 25 && rect.bottom >= window.innerHeight - 25;
      if (!isVisibleInView) return;

      const delta = e.deltaY;

      if (delta > 0) {
        // User scrolling DOWN
        if (frameRef.current < TOTAL_FRAMES - 1) {
          // Robot not yet fully detached -> LOCK PAGE & SCRUB FORWARD
          e.preventDefault();
          deltaAcc.current += delta;
          const threshold = 26; // balanced tactile resistance
          const steps = Math.floor(deltaAcc.current / threshold);
          if (steps > 0) {
            setFrameSafe(frameRef.current + steps);
            deltaAcc.current %= threshold;
          }
        }
        // If frameRef.current === TOTAL_FRAMES - 1: fully detached!
        // Do NOT preventDefault! Let page scroll down naturally to next section!
      } else if (delta < 0) {
        // User scrolling UP
        if (frameRef.current > 0) {
          // Robot not yet fully assembled -> LOCK PAGE & SCRUB BACKWARD
          e.preventDefault();
          deltaAcc.current += delta;
          const threshold = 26;
          const steps = Math.floor(deltaAcc.current / threshold);
          if (steps < 0) {
            setFrameSafe(frameRef.current + steps);
            deltaAcc.current %= threshold;
          }
        }
        // If frameRef.current === 0: fully assembled!
        // Do NOT preventDefault! Let page scroll up naturally to previous section!
      }
    };

    // Touch event handlers for mobile/trackpad swipe
    const onTouchStart = (e: TouchEvent) => {
      touchStartY.current = e.touches[0].clientY;
    };

    const onTouchMove = (e: TouchEvent) => {
      if (isPlaying || touchStartY.current === null) return;
      const rect = container.getBoundingClientRect();
      const isVisibleInView = rect.top <= 25 && rect.bottom >= window.innerHeight - 25;
      if (!isVisibleInView) return;

      const currentY = e.touches[0].clientY;
      const deltaY = touchStartY.current - currentY; // positive = swipe up = scroll down

      if (deltaY > 0 && frameRef.current < TOTAL_FRAMES - 1) {
        e.preventDefault();
        setFrameSafe(frameRef.current + 1);
        touchStartY.current = currentY;
      } else if (deltaY < 0 && frameRef.current > 0) {
        e.preventDefault();
        setFrameSafe(frameRef.current - 1);
        touchStartY.current = currentY;
      }
    };

    const onTouchEnd = () => {
      touchStartY.current = null;
    };

    // Attach with { passive: false } to allow e.preventDefault()
    window.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('touchstart', onTouchStart, { passive: true });
    window.addEventListener('touchmove', onTouchMove, { passive: false });
    window.addEventListener('touchend', onTouchEnd, { passive: true });

    return () => {
      window.removeEventListener('wheel', onWheel);
      window.removeEventListener('touchstart', onTouchStart);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onTouchEnd);
    };
  }, [isPlaying]);

  // Autoplay loop toggle
  const togglePlay = () => {
    if (isPlaying) {
      if (animIntervalRef.current) clearInterval(animIntervalRef.current);
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      let f = frameRef.current;
      let forward = true;

      animIntervalRef.current = window.setInterval(() => {
        if (forward) {
          f++;
          if (f >= TOTAL_FRAMES - 1) {
            forward = false;
          }
        } else {
          f--;
          if (f <= 0) {
            forward = true;
          }
        }
        setFrameSafe(f);
      }, 55);
    }
  };

  useEffect(() => {
    return () => {
      if (animIntervalRef.current) clearInterval(animIntervalRef.current);
    };
  }, []);

  const isDetached = currentFrame === TOTAL_FRAMES - 1;
  const progressRatio = currentFrame / (TOTAL_FRAMES - 1);
  const phase = progressRatio < 0.28 ? 1 : progressRatio < 0.65 ? 2 : 3;

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        height: '100vh',
        width: '100%',
        background: '#0b0f14',
        fontFamily: 'Manrope, sans-serif',
        color: '#f8fafc',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      {/* Subtle Ambient Radial Backlight Glow */}
      <div
        style={{
          position: 'absolute',
          width: '750px',
          height: '750px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(0, 229, 255, 0.16) 0%, rgba(196, 71, 45, 0.08) 45%, transparent 75%)',
          filter: 'blur(70px)',
          pointerEvents: 'none'
        }}
      />

      {/* Top Header HUD Bar */}
      <div
        style={{
          position: 'absolute',
          top: '32px',
          left: '48px',
          right: '48px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          zIndex: 20
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <span
            style={{
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '11px',
              letterSpacing: '0.22em',
              color: '#00e5ff',
              fontWeight: 700,
              background: 'rgba(0, 229, 255, 0.1)',
              border: '1px solid rgba(0, 229, 255, 0.3)',
              padding: '5px 14px',
              borderRadius: '20px'
            }}
          >
            AURIX ARCHITECTURE // 50-FRAME DECONSTRUCTION
          </span>
          <span
            style={{
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '12px',
              color: isDetached ? '#00e5ff' : '#94a3b8',
              fontWeight: isDetached ? 700 : 500
            }}
          >
            FRAME {String(currentFrame + 1).padStart(2, '0')} / {TOTAL_FRAMES}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={togglePlay}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              background: isPlaying ? 'rgba(196, 71, 45, 0.25)' : 'rgba(255, 255, 255, 0.08)',
              border: `1px solid ${isPlaying ? '#c4472d' : 'rgba(255, 255, 255, 0.15)'}`,
              borderRadius: '24px',
              padding: '6px 16px',
              color: isPlaying ? '#ff6b4a' : '#f8fafc',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              backdropFilter: 'blur(8px)',
              transition: 'all 0.2s'
            }}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            <span>{isPlaying ? 'Pause Loop' : 'Auto Explode Loop'}</span>
          </button>

          <button
            onClick={() => setFrameSafe(0)}
            style={{
              background: 'rgba(255, 255, 255, 0.08)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              borderRadius: '50%',
              width: '34px',
              height: '34px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#94a3b8',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            title="Reset to Enclosed State"
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      {/* Main Full-Bleed Canvas Viewport (Unobstructed, Robot Centered) */}
      <div
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {!isLoaded ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px' }}>
            <div
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '50%',
                border: '3px solid rgba(0, 229, 255, 0.2)',
                borderTopColor: '#00e5ff',
                animation: 'aurix-spin 0.8s linear infinite'
              }}
            />
            <span style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '12px', color: '#00e5ff' }}>
              CACHING EXPLODED KINEMATICS ({loadProgress}%)
            </span>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            style={{
              width: '100%',
              height: '100%',
              display: 'block',
              objectFit: 'contain'
            }}
          />
        )}
      </div>

      {/* Dynamic Telemetry HUD Callout (Floating on Lower Left) */}
      <div
        style={{
          position: 'absolute',
          left: '48px',
          bottom: '48px',
          maxWidth: '380px',
          backgroundColor: 'rgba(11, 15, 20, 0.86)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: '16px',
          padding: '24px',
          boxShadow: '0 16px 40px rgba(0, 0, 0, 0.65)',
          zIndex: 20,
          transition: 'all 0.3s ease'
        }}
      >
        {phase === 1 && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#00e5ff', marginBottom: '8px' }}>
              <Shield size={16} />
              <span style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px', letterSpacing: '0.12em', fontWeight: 700 }}>
                PHASE 01 // ENCLOSED TITANIUM EXOSKELETON
              </span>
            </div>
            <h3 style={{ fontSize: '19px', fontWeight: 800, margin: '0 0 8px', color: '#f8fafc' }}>
              Titanium-Composite Armor Shell
            </h3>
            <p style={{ fontSize: '13px', color: '#94a3b8', lineHeight: 1.5, margin: '0 0 16px' }}>
              High-tensile ballistic armor cowlings with anti-abrasive ceramic coating and hermetic environmental atmospheric seals.
            </p>
            <div style={{ display: 'flex', gap: '8px' }}>
              <span style={chipStyle}>TENSILE: 520 MPa</span>
              <span style={chipStyle}>SEAL: IP69K</span>
              <span style={chipStyle}>MASS: 42 kg</span>
            </div>
          </div>
        )}

        {phase === 2 && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#38bdf8', marginBottom: '8px' }}>
              <Zap size={16} />
              <span style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px', letterSpacing: '0.12em', fontWeight: 700 }}>
                PHASE 02 // DUAL PLASMA CORE
              </span>
            </div>
            <h3 style={{ fontSize: '19px', fontWeight: 800, margin: '0 0 8px', color: '#f8fafc' }}>
              Solid-State Sub-Atomic Reactor
            </h3>
            <p style={{ fontSize: '13px', color: '#94a3b8', lineHeight: 1.5, margin: '0 0 16px' }}>
              Cryogenically stabilized cyan core supplying ultra-dense power to the 64-channel neural bus and multi-axis planetary motor stages.
            </p>
            <div style={{ display: 'flex', gap: '8px' }}>
              <span style={chipStyle}>DENSITY: 1.4 MW/kg</span>
              <span style={chipStyle}>TEMP: 4.2 K</span>
              <span style={chipStyle}>EFFICIENCY: 99.4%</span>
            </div>
          </div>
        )}

        {phase === 3 && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: isDetached ? '#00e5ff' : '#c4472d', marginBottom: '8px' }}>
              <Cpu size={16} />
              <span style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px', letterSpacing: '0.12em', fontWeight: 700 }}>
                {isDetached ? 'PHASE 03 // FULLY DETACHED' : 'PHASE 03 // DECONSTRUCTED SKELETON'}
              </span>
            </div>
            <h3 style={{ fontSize: '19px', fontWeight: 800, margin: '0 0 8px', color: '#f8fafc' }}>
              {isDetached ? 'Components Fully Detached' : 'Hydro-Pneumatic Planetary Servos'}
            </h3>
            <p style={{ fontSize: '13px', color: '#94a3b8', lineHeight: 1.5, margin: '0 0 16px' }}>
              {isDetached
                ? 'Every armor plate, core module, and servo limb is completely separated along the axis. Continue scrolling down to explore the Mechanism Catalog.'
                : 'Every armor panel and hydro-servo floats outward in a cinematic exploded view for field replacement and micro-component calibration.'}
            </p>
            <div style={{ display: 'flex', gap: '8px' }}>
              <span style={chipStyle}>AXES: 28-DOF</span>
              <span style={chipStyle}>TORQUE: 420 Nm</span>
              <span style={chipStyle}>STATUS: {isDetached ? 'DETACHED' : 'EXPLODING'}</span>
            </div>
          </div>
        )}
      </div>

      {/* Non-Intrusive Bottom Status Indicator (No Slider Bar!) */}
      <div
        style={{
          position: 'absolute',
          bottom: '36px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 20,
          pointerEvents: 'none'
        }}
      >
        {!isDetached ? (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              backgroundColor: 'rgba(11, 15, 20, 0.75)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '20px',
              padding: '6px 16px',
              color: '#94a3b8',
              fontSize: '11px',
              fontFamily: 'IBM Plex Mono, monospace',
              letterSpacing: '0.14em'
            }}
          >
            <span>SCROLL DOWN TO DETACH ROBOT</span>
            <ChevronDown size={14} color="#00e5ff" style={{ animation: 'bounce-hint 1.4s ease-in-out infinite' }} />
          </div>
        ) : (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              backgroundColor: 'rgba(0, 229, 255, 0.12)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(0, 229, 255, 0.4)',
              borderRadius: '20px',
              padding: '6px 18px',
              color: '#00e5ff',
              fontSize: '11px',
              fontFamily: 'IBM Plex Mono, monospace',
              letterSpacing: '0.14em',
              fontWeight: 700,
              boxShadow: '0 0 20px rgba(0, 229, 255, 0.25)'
            }}
          >
            <CheckCircle2 size={14} />
            <span>ROBOT FULLY DETACHED · SCROLL DOWN TO CONTINUE</span>
            <ChevronDown size={14} />
          </div>
        )}
      </div>

      <style>{`
        @keyframes aurix-spin {
          to { transform: rotate(360deg); }
        }
        @keyframes bounce-hint {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(4px); }
        }
      `}</style>
    </div>
  );
}

const chipStyle: React.CSSProperties = {
  fontFamily: 'IBM Plex Mono, monospace',
  fontSize: '10px',
  color: '#cbd5e1',
  backgroundColor: 'rgba(255, 255, 255, 0.06)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  padding: '4px 8px',
  borderRadius: '4px',
  letterSpacing: '0.06em'
};
