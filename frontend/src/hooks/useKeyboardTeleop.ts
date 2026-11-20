import { useEffect, useRef, useState } from 'react';
import { SimulationSocket } from '../services/api';

export function useKeyboardTeleop(
  socket: SimulationSocket | null,
  onTogglePlay: () => void,
  onReset: () => void,
  onAutoPlay?: () => void,
  onFocusRobot?: () => void,
  isDroneMode: boolean = false,
  droneFlightMode: 'stabilize' | 'acro' | 'alt_hold' = 'stabilize',
  activeDroneId: string = 'drone',
  onToggleFpv?: () => void
) {
  const [activeKeys, setActiveKeys] = useState<{ [key: string]: boolean }>({});
  const activeKeysRef = useRef<{ [key: string]: boolean }>({});
  const isDroneModeRef = useRef(isDroneMode);
  const droneFlightModeRef = useRef(droneFlightMode);
  const activeDroneIdRef = useRef(activeDroneId);

  useEffect(() => {
    activeKeysRef.current = activeKeys;
  }, [activeKeys]);

  useEffect(() => {
    isDroneModeRef.current = isDroneMode;
    droneFlightModeRef.current = droneFlightMode;
    activeDroneIdRef.current = activeDroneId;
  }, [isDroneMode, droneFlightMode, activeDroneId]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // If typing in an input, allow Escape to blur and regain keyboard control
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName)) {
        if (e.key === 'Escape') {
          (document.activeElement as HTMLElement)?.blur();
        }
        return;
      }

      const key = e.key.toLowerCase();

      // Drone mode quick hotkeys
      if (isDroneModeRef.current) {
        if (key === 't' && !e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          socket?.sendDroneAction('takeoff', 1.5, activeDroneIdRef.current);
          return;
        }
        if (key === 'l' && !e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          socket?.sendDroneAction('land', 0.0, activeDroneIdRef.current);
          return;
        }
        if (key === 'h' && !e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          socket?.sendDroneAction('hover', 1.5, activeDroneIdRef.current);
          return;
        }
      }

      if (e.code === 'Space') {
        if (!isDroneModeRef.current) {
          e.preventDefault();
          onTogglePlay();
          return;
        } else {
          // In drone mode, Space is climb up
          e.preventDefault();
          onAutoPlay?.();
          setActiveKeys((prev) => ({ ...prev, ' ': true }));
          return;
        }
      }

      if (key === 'r' && !e.ctrlKey && !e.metaKey) {
        onReset();
        return;
      }

      if (key === 'f' && !e.ctrlKey && !e.metaKey) {
        if (onToggleFpv) {
          e.preventDefault();
          onToggleFpv();
          return;
        } else if (onFocusRobot) {
          onFocusRobot();
          return;
        }
      }

      const monitoredKeys = isDroneModeRef.current
        ? ['w', 'a', 's', 'd', 'q', 'e', 'c', 'shift', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']
        : ['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'];

      if (monitoredKeys.includes(key)) {
        e.preventDefault();
        onAutoPlay?.();
        setActiveKeys((prev) => ({ ...prev, [key]: true }));
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      const key = e.key.toLowerCase();
      const code = e.code;
      if (code === 'Space') {
        setActiveKeys((prev) => {
          const next = { ...prev };
          delete next[' '];
          return next;
        });
      } else {
        setActiveKeys((prev) => {
          const next = { ...prev };
          delete next[key];
          return next;
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [socket, onTogglePlay, onReset, onAutoPlay, onFocusRobot, onToggleFpv]);

  // Periodic teleoperation dispatch loop
  const wasMovingRef = useRef(false);
  useEffect(() => {
    const interval = setInterval(() => {
      const keys = activeKeysRef.current;
      const isDrone = isDroneModeRef.current;
      const droneMode = droneFlightModeRef.current;
      const dId = activeDroneIdRef.current;

      if (isDrone) {
        // Quadcopter 4-axis flight inputs
        let pitch = 0.0;
        let roll = 0.0;
        let yaw = 0.0;
        let climb = 0.0;

        if (keys['w'] || keys['arrowup']) pitch += 1.0;
        if (keys['s'] || keys['arrowdown']) pitch -= 1.0;
        if (keys['d']) roll += 1.0;
        if (keys['a']) roll -= 1.0;
        if (keys['e'] || keys['arrowright']) yaw -= 1.0;
        if (keys['q'] || keys['arrowleft']) yaw += 1.0;
        if (keys[' '] || keys['shift']) climb += 1.0;
        if (keys['c']) climb -= 1.0;

        const flying = pitch !== 0.0 || roll !== 0.0 || yaw !== 0.0 || climb !== 0.0;
        if (flying) {
          onAutoPlay?.();
          socket?.sendDroneTeleop(pitch, roll, yaw, climb, droneMode, dId);
          wasMovingRef.current = true;
        } else if (wasMovingRef.current) {
          socket?.sendDroneTeleop(0.0, 0.0, 0.0, 0.0, droneMode, dId);
          wasMovingRef.current = false;
        }
      } else {
        // Ground rover differential teleoperation
        let linear = 0.0;
        let angular = 0.0;

        if (keys['w'] || keys['arrowup']) linear += 1.0;
        if (keys['s'] || keys['arrowdown']) linear -= 1.0;
        if (keys['a'] || keys['arrowleft']) angular += 1.0;
        if (keys['d'] || keys['arrowright']) angular -= 1.0;

        const moving = linear !== 0.0 || angular !== 0.0;

        if (moving) {
          onAutoPlay?.();
          socket?.sendTeleop(linear, angular);
          wasMovingRef.current = true;
        } else if (wasMovingRef.current) {
          socket?.sendTeleop(0.0, 0.0);
          wasMovingRef.current = false;
        }
      }
    }, 40);

    return () => clearInterval(interval);
  }, [socket, onAutoPlay]);

  return activeKeys;
}
