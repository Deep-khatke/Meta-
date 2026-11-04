import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { LogOut, ShieldCheck, Settings, User } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';

interface UserProfileBadgeProps {
  variant?: 'light' | 'dark';
}

export const UserProfileBadge: React.FC<UserProfileBadgeProps> = ({ variant = 'dark' }) => {
  const { user, isAuthenticated, logout } = useAuth();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [imgError, setImgError] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  // Reset imgError if user picture changes
  useEffect(() => {
    setImgError(false);
  }, [user?.picture]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const isLight = variant === 'light';

  if (!isAuthenticated || !user) {
    return (
      <Link
        to="/login"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          textDecoration: 'none',
          padding: '6px 14px',
          borderRadius: 100,
          background: isLight ? 'rgba(32, 34, 34, 0.05)' : 'rgba(255, 255, 255, 0.08)',
          border: isLight ? '1px solid rgba(32, 34, 34, 0.15)' : '1px solid rgba(255, 255, 255, 0.18)',
          color: isLight ? '#202222' : '#F1EEE7',
          fontSize: 13,
          fontFamily: 'Manrope, sans-serif',
          fontWeight: 600,
          cursor: 'pointer',
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'scale(1.03)';
          e.currentTarget.style.borderColor = '#C4472D';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'scale(1)';
          e.currentTarget.style.borderColor = isLight ? 'rgba(32, 34, 34, 0.15)' : 'rgba(255, 255, 255, 0.18)';
        }}
      >
        <svg width="15" height="15" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
          <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
          <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
          <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" />
          <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
        </svg>
        <span>Sign in</span>
      </Link>
    );
  }

  const displayName = user.name || user.email?.split('@')[0] || 'User';
  const firstName = displayName.split(' ')[0];
  const initials = displayName
    .split(' ')
    .filter(Boolean)
    .map((n) => n[0])
    .join('')
    .slice(0, 2)
    .toUpperCase() || 'U';

  const hasValidPicture = Boolean(user.picture && user.picture.trim().length > 5 && !imgError);

  return (
    <div ref={dropdownRef} style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
      <button
        type="button"
        onClick={() => setDropdownOpen(!dropdownOpen)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          background: isLight ? 'rgba(32, 34, 34, 0.05)' : 'rgba(255, 255, 255, 0.08)',
          border: isLight ? '1px solid rgba(32, 34, 34, 0.15)' : '1px solid rgba(255, 255, 255, 0.18)',
          borderRadius: 100,
          padding: '3px 12px 3px 3px',
          color: isLight ? '#202222' : '#F1EEE7',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          boxShadow: isLight ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
        }}
        title={`Signed in as ${displayName} (${user.email})`}
      >
        {/* Generous, high-res avatar (34x34 px) */}
        {hasValidPicture ? (
          <img
            src={user.picture}
            alt={displayName}
            onError={() => setImgError(true)}
            style={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              objectFit: 'cover',
              border: '1.5px solid #C4472D',
              flexShrink: 0,
            }}
          />
        ) : (
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #FF6B4A 0%, #C4472D 100%)',
              color: '#FFFFFF',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 13,
              fontWeight: 700,
              fontFamily: 'Manrope, sans-serif',
              boxShadow: '0 2px 6px rgba(196, 71, 45, 0.3)',
              flexShrink: 0,
            }}
          >
            {initials}
          </div>
        )}

        <span
          style={{
            fontSize: 13,
            fontWeight: 700,
            fontFamily: 'Manrope, sans-serif',
            maxWidth: 120,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {firstName}
        </span>
      </button>

      {/* User profile dropdown popup */}
      {dropdownOpen && (
        <div
          style={{
            position: 'absolute',
            top: 'calc(100% + 10px)',
            right: 0,
            backgroundColor: '#17191A',
            border: '1px solid #333',
            borderRadius: 12,
            boxShadow: '0 16px 40px rgba(0,0,0,0.85)',
            zIndex: 9999,
            width: 250,
            padding: 10,
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
            fontFamily: 'Manrope, sans-serif',
          }}
        >
          <div style={{ padding: '10px 12px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#F1EEE7' }}>{displayName}</div>
            <div style={{ fontSize: 12, color: '#9A9A96', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {user.email}
            </div>
            <div
              style={{
                marginTop: 8,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                background: user.provider === 'google' ? 'rgba(52, 168, 83, 0.15)' : 'rgba(0, 229, 255, 0.15)',
                color: user.provider === 'google' ? '#34A853' : '#00e5ff',
                padding: '3px 8px',
                borderRadius: 10,
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              <ShieldCheck size={12} />
              <span>{user.provider === 'google' ? 'Google Account' : 'RoboForge Member'}</span>
            </div>
          </div>

          <button
            type="button"
            onClick={() => {
              setDropdownOpen(false);
              navigate('/login');
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '9px 12px',
              background: 'transparent',
              border: 'none',
              borderRadius: 6,
              color: '#F1EEE7',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
              textAlign: 'left',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
          >
            <Settings size={14} color="#9A9A96" />
            <span>Switch Account</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setDropdownOpen(false);
              logout();
              navigate('/login');
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '9px 12px',
              background: 'transparent',
              border: 'none',
              borderRadius: 6,
              color: '#ef4444',
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
              textAlign: 'left',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.1)')}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
          >
            <LogOut size={14} />
            <span>Sign out</span>
          </button>
        </div>
      )}
    </div>
  );
};
