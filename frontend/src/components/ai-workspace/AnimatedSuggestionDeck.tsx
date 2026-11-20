import React from 'react';
import {
  Sparkles,
  Car,
  Plane,
  Wrench,
  Zap,
  HelpCircle,
  ArrowRight,
  Compass
} from 'lucide-react';

interface AnimatedSuggestionDeckProps {
  suggestions?: string[];
  onSelectSuggestion: (text: string) => void;
  title?: string;
  isEducatorMode?: boolean;
}

export const AnimatedSuggestionDeck: React.FC<AnimatedSuggestionDeckProps> = ({
  suggestions,
  onSelectSuggestion,
  title = 'What would you like to build or explore?',
  isEducatorMode = false,
}) => {
  if (!suggestions || suggestions.length === 0) return null;

  const getSuggestionIcon = (s: string) => {
    const lower = s.toLowerCase();
    if (lower.includes('drone') || lower.includes('fly') || lower.includes('altitude') || lower.includes('hover')) {
      return <Plane size={13} color="#38BDF8" />;
    }
    if (lower.includes('rover') || lower.includes('wheel') || lower.includes('steer') || lower.includes('drive')) {
      return <Car size={13} color="#F97316" />;
    }
    if (lower.includes('arm') || lower.includes('joint') || lower.includes('dof') || lower.includes('kinematics')) {
      return <Wrench size={13} color="#A855F7" />;
    }
    if (lower.includes('pid') || lower.includes('slip') || lower.includes('physics') || lower.includes('lidar')) {
      return <Zap size={13} color="#FBBF24" />;
    }
    return <Sparkles size={13} color="#38BDF8" />;
  };

  return (
    <div
      style={{
        marginTop: 14,
        padding: '12px 14px',
        borderRadius: 12,
        backgroundColor: isEducatorMode ? 'rgba(14, 30, 48, 0.45)' : 'rgba(23, 25, 28, 0.65)',
        border: `1px solid ${isEducatorMode ? 'rgba(56, 189, 248, 0.22)' : 'rgba(255, 255, 255, 0.08)'}`,
        backdropFilter: 'blur(10px)',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 11.5,
          fontWeight: 700,
          color: isEducatorMode ? '#38BDF8' : '#E4E4E7',
          marginBottom: 10,
          letterSpacing: '0.02em',
        }}
      >
        <Compass size={13} color={isEducatorMode ? '#38BDF8' : '#D35432'} />
        <span>{title}</span>
      </div>

      {/* Suggestion Cards Grid with Staggered Smooth Animation */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: 8,
        }}
      >
        {suggestions.map((suggestion, index) => {
          const delaySec = (index * 0.09).toFixed(2);
          return (
            <button
              key={index}
              onClick={() => onSelectSuggestion(suggestion)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 12px',
                borderRadius: 8,
                backgroundColor: isEducatorMode ? 'rgba(22, 42, 66, 0.6)' : '#191B20',
                border: `1px solid ${isEducatorMode ? 'rgba(56, 189, 248, 0.25)' : 'rgba(255, 255, 255, 0.07)'}`,
                color: '#F4F4F5',
                fontSize: 11.5,
                fontWeight: 500,
                textAlign: 'left',
                cursor: 'pointer',
                opacity: 0,
                animation: `rf-suggestion-enter 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards`,
                animationDelay: `${delaySec}s`,
                transition: 'transform 0.18s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.18s ease, background-color 0.18s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.borderColor = isEducatorMode ? 'rgba(56, 189, 248, 0.65)' : 'rgba(211, 84, 50, 0.55)';
                e.currentTarget.style.backgroundColor = isEducatorMode ? 'rgba(30, 58, 92, 0.85)' : '#23252C';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.borderColor = isEducatorMode ? 'rgba(56, 189, 248, 0.25)' : 'rgba(255, 255, 255, 0.07)';
                e.currentTarget.style.backgroundColor = isEducatorMode ? 'rgba(22, 42, 66, 0.6)' : '#191B20';
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
                <span style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
                  {getSuggestionIcon(suggestion)}
                </span>
                <span style={{ lineHeight: 1.35 }}>{suggestion}</span>
              </div>
              <ArrowRight size={11} color="#71717A" style={{ marginLeft: 6, flexShrink: 0 }} />
            </button>
          );
        })}
      </div>
    </div>
  );
};

