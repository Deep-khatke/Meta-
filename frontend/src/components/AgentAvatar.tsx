import React, { useEffect, useState } from 'react';

interface AgentAvatarProps {
  mood?: 'idle' | 'thinking' | 'speaking' | 'talking' | 'success';
  size?: number;
  showBadge?: boolean;
}

export const AgentAvatar: React.FC<AgentAvatarProps> = ({
  mood = 'idle',
  size = 48,
  showBadge = true,
}) => {
  const [blink, setBlink] = useState(false);

  // Periodic natural eye blinking when idle
  useEffect(() => {
    if (mood !== 'idle') return;
    const interval = setInterval(() => {
      setBlink(true);
      setTimeout(() => setBlink(false), 160);
    }, 3800);
    return () => clearInterval(interval);
  }, [mood]);

  const eyeColor =
    mood === 'thinking' ? '#a9aca7' : mood === 'success' ? '#5cc596' : '#e8622c';
  const glowColor =
    mood === 'thinking'
      ? 'rgba(255, 255, 255, 0.171)'
      : mood === 'success'
        ? 'rgba(74, 222, 128, 0.45)'
        : 'rgba(232, 98, 44, 0.35)';

  return (
    <div
      style={{
        position: 'relative',
        width: size,
        height: size,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        filter: `drop-shadow(0 0 10px ${glowColor})`,
        animation: mood === 'thinking' ? 'agentBob 1.2s ease-in-out infinite' : 'agentBreathe 3s ease-in-out infinite',
      }}
    >
      <style>{`
        @keyframes agentBreathe {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-3px); }
        }
        @keyframes agentBob {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          25% { transform: translateY(-4px) rotate(-1.5deg); }
          75% { transform: translateY(-2px) rotate(1.5deg); }
        }
        @keyframes antennaPulse {
          0%, 100% { opacity: 0.4; transform: scale(0.9); }
          50% { opacity: 1.0; transform: scale(1.2); }
        }
        @keyframes radarScan {
          0% { transform: translateX(-8px); opacity: 0.3; }
          50% { transform: translateX(8px); opacity: 1.0; }
          100% { transform: translateX(-8px); opacity: 0.3; }
        }
      `}</style>

      <svg width={size} height={size} viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        {/* Antenna */}
        <line x1="32" y1="14" x2="32" y2="4" stroke="#8a8d89" strokeWidth="2.5" strokeLinecap="round" />
        <circle
          cx="32"
          cy="4"
          r="3.5"
          fill={eyeColor}
          style={{ animation: mood === 'thinking' ? 'antennaPulse 0.8s infinite' : 'none' }}
        />
        {mood === 'thinking' && (
          <circle cx="32" cy="4" r="6" stroke={eyeColor} strokeWidth="1" strokeDasharray="2 2" opacity="0.6" />
        )}

        {/* Head Chassis */}
        <rect x="10" y="14" width="44" height="38" rx="12" fill="#181d29" stroke="#363b3e" strokeWidth="2" />
        <rect x="12" y="16" width="40" height="34" rx="10" fill="url(#chassisGrad)" />

        {/* Ear Bolts */}
        <rect x="6" y="27" width="4" height="12" rx="2" fill="#6b6e6a" />
        <rect x="54" y="27" width="4" height="12" rx="2" fill="#6b6e6a" />

        {/* Visor Screen */}
        <rect x="16" y="21" width="32" height="18" rx="6" fill="#090c13" stroke="#1b1e20" strokeWidth="1" />

        {/* Eyes / Face display */}
        {mood === 'thinking' ? (
          // Thinking radar sweep
          <g style={{ animation: 'radarScan 1.1s ease-in-out infinite' }}>
            <rect x="24" y="27" width="16" height="5" rx="2.5" fill={eyeColor} />
            <circle cx="32" cy="29.5" r="4" fill="#ffffff" />
          </g>
        ) : mood === 'success' ? (
          // Happy curved eyes ^ ^
          <g stroke={eyeColor} strokeWidth="2.5" strokeLinecap="round">
            <path d="M22 31 Q25 26 28 31" />
            <path d="M36 31 Q39 26 42 31" />
          </g>
        ) : blink ? (
          // Blinking eyes (flat lines)
          <g stroke={eyeColor} strokeWidth="2" strokeLinecap="round">
            <line x1="22" y1="30" x2="28" y2="30" />
            <line x1="36" y1="30" x2="42" y2="30" />
          </g>
        ) : (
          // Normal friendly glowing eyes
          <g>
            <circle cx="25" cy="29" r="4" fill={eyeColor} />
            <circle cx="26" cy="28" r="1.5" fill="#ffffff" />
            <circle cx="39" cy="29" r="4" fill={eyeColor} />
            <circle cx="40" cy="28" r="1.5" fill="#ffffff" />
          </g>
        )}

        {/* Cute Mouth */}
        {mood === 'speaking' ? (
          <path d="M28 43 Q32 47 36 43" stroke={eyeColor} strokeWidth="2" strokeLinecap="round" fill="none" />
        ) : mood === 'success' ? (
          <path d="M26 43 Q32 48 38 43" stroke="#5cc596" strokeWidth="2" strokeLinecap="round" fill="none" />
        ) : (
          <line x1="29" y1="44" x2="35" y2="44" stroke="#6b6e6a" strokeWidth="1.8" strokeLinecap="round" />
        )}

        {/* Gradients */}
        <defs>
          <linearGradient id="chassisGrad" x1="12" y1="16" x2="52" y2="50" gradientUnits="userSpaceOnUse">
            <stop stopColor="#222a3d" />
            <stop offset="1" stopColor="#131516" />
          </linearGradient>
        </defs>
      </svg>

      {/* Online / Status Pill Badge */}
      {showBadge && (
        <span
          style={{
            position: 'absolute',
            bottom: -2,
            right: -2,
            width: 10,
            height: 10,
            borderRadius: '50%',
            backgroundColor: mood === 'thinking' ? '#a9aca7' : '#3fb27f',
            border: '2px solid #0f1112',
            boxShadow: `0 0 6px ${mood === 'thinking' ? '#a9aca7' : '#3fb27f'}`,
          }}
        />
      )}
    </div>
  );
};
