import { useEffect, useRef, useState } from 'react';
import { SimulationSocket } from '../services/api';

export function useKeyboardTeleop(
  socket: SimulationSocket | null,
  onTogglePlay: () => void,
  onReset: () => void,
  onAutoPlay?: () => void,
  onFocusRobot?: () => void
) {
  const [activeKeys, setActiveKeys] = useState<{ [key: string]: boolean }>({});
  const activeKeysRef = useRef<{ [key: string]: boolean }>({});

  useEffect(() => {
    activeKeysRef.current = activeKeys;
  }, [activeKeys]);

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

      if (e.code === 'Space') {
        e.preventDefault();
        onTogglePlay();
        return;
      }

      if (key === 'r' && !e.ctrlKey && !e.metaKey) {
        onReset();
        return;
      }

      if (key === 'f' && !e.ctrlKey && !e.metaKey && onFocusRobot) {
        onFocusRobot();
        return;
      }

      if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)) {
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
      if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)) {
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
  }, [onTogglePlay, onReset, onAutoPlay, onFocusRobot]);

  // Periodic teleoperation dispatch loop
  useEffect(() => {
    const interval = setInterval(() => {
      const keys = activeKeysRef.current;
      let linear = 0.0;
      let angular = 0.0;

      if (keys['w'] || keys['arrowup']) linear += 1.0;
      if (keys['s'] || keys['arrowdown']) linear -= 1.0;
      if (keys['a'] || keys['arrowleft']) angular += 1.0; // Left rotation
      if (keys['d'] || keys['arrowright']) angular -= 1.0; // Right rotation

      if (linear !== 0.0 || angular !== 0.0) {
        onAutoPlay?.();
      }

      if (socket) {
        socket.sendTeleop(linear, angular);
      }
    }, 40);

    return () => clearInterval(interval);
  }, [socket, onAutoPlay]);

  return activeKeys;
}
