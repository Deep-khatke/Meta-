import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { AlertCircle, Loader2 } from 'lucide-react';

declare global {
  interface Window {
    google?: any;
  }
}

interface GoogleSignInButtonProps {
  onSuccess?: () => void;
  onError?: (err: string) => void;
  text?: 'signin_with' | 'signup_with' | 'continue_with';
  width?: number | string;
}

export const GoogleSignInButton: React.FC<GoogleSignInButtonProps> = ({
  onSuccess,
  onError,
  text = 'continue_with',
  width = '100%',
}) => {
  const { loginWithGoogleCredential, loginWithGoogleUserInfo } = useAuth();
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const googleClientId =
    import.meta.env.VITE_GOOGLE_CLIENT_ID ||
    '567256736468-3m5c2l0i74tronhurhm14mu174qkjbtc.apps.googleusercontent.com';

  useEffect(() => {
    let isMounted = true;

    const loadGsiScript = () => {
      if (window.google?.accounts) {
        initOneTap();
        return;
      }

      const existingScript = document.getElementById('google-gsi-client') as HTMLScriptElement;
      if (existingScript) {
        existingScript.addEventListener('load', () => {
          if (isMounted) initOneTap();
        });
        return;
      }

      const script = document.createElement('script');
      script.id = 'google-gsi-client';
      script.src = 'https://accounts.google.com/gsi/client';
      script.async = true;
      script.defer = true;
      script.onload = () => {
        if (isMounted) initOneTap();
      };
      document.body.appendChild(script);
    };

    const initOneTap = () => {
      if (!window.google?.accounts?.id || !googleClientId) return;
      try {
        window.google.accounts.id.initialize({
          client_id: googleClientId,
          callback: async (response: any) => {
            if (response.credential) {
              setIsLoading(true);
              setErrorMsg(null);
              try {
                await loginWithGoogleCredential(response.credential);
                onSuccess?.();
              } catch (err: any) {
                const msg = err.message || 'Google authentication failed.';
                setErrorMsg(msg);
                onError?.(msg);
              } finally {
                setIsLoading(false);
              }
            }
          },
          auto_select: false,
          cancel_on_tap_outside: true,
        });

        // Optionally prompt One-Tap non-intrusively
        window.google.accounts.id.prompt();
      } catch (err) {
        console.warn('Google One-Tap initialization notice:', err);
      }
    };

    loadGsiScript();

    return () => {
      isMounted = false;
    };
  }, [googleClientId]);

  // Click handler using Google OAuth2 Token Client (opens official Google popup dialog)
  const handleGoogleClick = () => {
    setErrorMsg(null);
    setIsLoading(true);

    if (window.google?.accounts?.oauth2) {
      try {
        const client = window.google.accounts.oauth2.initTokenClient({
          client_id: googleClientId,
          scope: 'openid profile email',
          callback: async (tokenResponse: any) => {
            if (tokenResponse?.error) {
              setIsLoading(false);
              setErrorMsg(`Google authorization notice: ${tokenResponse.error}`);
              onError?.(tokenResponse.error);
              return;
            }
            if (tokenResponse?.access_token) {
              try {
                // Fetch verified profile directly from Google UserInfo endpoint
                const userInfoRes = await fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
                  headers: { Authorization: `Bearer ${tokenResponse.access_token}` },
                });
                const userInfo = await userInfoRes.json();
                if (userInfo?.sub && userInfo?.email) {
                  await loginWithGoogleUserInfo({
                    sub: userInfo.sub,
                    email: userInfo.email,
                    name: userInfo.name || userInfo.given_name || userInfo.email.split('@')[0],
                    picture: userInfo.picture,
                  });
                  onSuccess?.();
                } else {
                  throw new Error('Incomplete profile received from Google.');
                }
              } catch (err: any) {
                const msg = err.message || 'Failed to retrieve Google profile.';
                setErrorMsg(msg);
                onError?.(msg);
              } finally {
                setIsLoading(false);
              }
            } else {
              setIsLoading(false);
            }
          },
        });
        client.requestAccessToken();
        return;
      } catch (err: any) {
        console.warn('Google OAuth2 client error:', err);
      }
    }

    // Fallback: trigger One-Tap prompt
    if (window.google?.accounts?.id) {
      try {
        window.google.accounts.id.prompt((notification: any) => {
          if (notification.isNotDisplayed() || notification.isSkippedMoment()) {
            setIsLoading(false);
          }
        });
        return;
      } catch (_) {}
    }

    setIsLoading(false);
    setErrorMsg('Google services are initializing. Please click again.');
  };

  const buttonLabel =
    text === 'signup_with'
      ? 'Sign up with Google'
      : text === 'signin_with'
      ? 'Sign in with Google'
      : 'Continue with Google';

  return (
    <div style={{ width: typeof width === 'number' ? `${width}px` : width, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
      <button
        type="button"
        onClick={handleGoogleClick}
        disabled={isLoading}
        style={{
          width: '100%',
          height: '44px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          padding: '0 18px',
          borderRadius: '8px',
          background: '#FFFFFF',
          color: '#1F2937',
          border: '1px solid #D1D5DB',
          fontFamily: 'Manrope, sans-serif',
          fontSize: '14px',
          fontWeight: 600,
          cursor: isLoading ? 'wait' : 'pointer',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04)',
          transition: 'all 0.18s ease',
          boxSizing: 'border-box',
          userSelect: 'none',
        }}
        onMouseEnter={(e) => {
          if (!isLoading) {
            e.currentTarget.style.backgroundColor = '#F9FAFB';
            e.currentTarget.style.borderColor = '#9CA3AF';
            e.currentTarget.style.boxShadow = '0 3px 6px rgba(0,0,0,0.1)';
          }
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = '#FFFFFF';
          e.currentTarget.style.borderColor = '#D1D5DB';
          e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04)';
        }}
      >
        {isLoading ? (
          <Loader2 size={18} className="animate-spin" color="#4285F4" />
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
            <path
              fill="#4285F4"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            />
            <path
              fill="#34A853"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            />
            <path
              fill="#FBBC05"
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
            />
            <path
              fill="#EA4335"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
            />
          </svg>
        )}
        <span>{buttonLabel}</span>
      </button>

      {errorMsg && (
        <div
          style={{
            color: '#DC2626',
            fontSize: '11px',
            fontFamily: 'IBM Plex Mono, monospace',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            marginTop: '2px',
            background: 'rgba(220, 38, 38, 0.08)',
            padding: '6px 10px',
            borderRadius: '6px',
            width: '100%',
            boxSizing: 'border-box',
          }}
        >
          <AlertCircle size={14} style={{ flexShrink: 0 }} />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
};
