import React, { createContext, useContext, useState, useEffect } from 'react';

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  picture?: string;
  provider: 'google' | 'email';
  created_at?: number;
}

interface AuthContextType {
  user: UserProfile | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<UserProfile>;
  signup: (name: string, email: string, password: string) => Promise<UserProfile>;
  loginWithGoogleCredential: (credential: string) => Promise<UserProfile>;
  loginWithGoogleUserInfo: (userInfo: { sub: string; email: string; name: string; picture?: string }) => Promise<UserProfile>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Helper to decode JWT without external libraries
function parseJwt(token: string): any {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch (e) {
    console.error('Failed to parse JWT token:', e);
    return null;
  }
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserProfile | null>(() => {
    try {
      const savedUser = localStorage.getItem('roboforge_user');
      return savedUser ? JSON.parse(savedUser) : null;
    } catch {
      return null;
    }
  });
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Validate session against database on initial load
  useEffect(() => {
    const verifySession = async () => {
      const token = localStorage.getItem('roboforge_token');
      if (!token) {
        setIsLoading(false);
        return;
      }
      try {
        const res = await fetch('/api/auth/me', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (res.ok) {
          const data = await res.json();
          if (data.authenticated && data.user) {
            setUser(data.user);
            localStorage.setItem('roboforge_user', JSON.stringify(data.user));
          } else {
            // Token expired or invalid
            setUser(null);
            localStorage.removeItem('roboforge_user');
            localStorage.removeItem('roboforge_token');
          }
        }
      } catch (err) {
        console.warn('Session verification error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    verifySession();
  }, []);

  const login = async (email: string, password: string): Promise<UserProfile> => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Invalid email or password.');
      }
      const loggedInUser: UserProfile = data.user;
      setUser(loggedInUser);
      localStorage.setItem('roboforge_user', JSON.stringify(loggedInUser));
      if (data.token) {
        localStorage.setItem('roboforge_token', data.token);
      }
      return loggedInUser;
    } finally {
      setIsLoading(false);
    }
  };

  const signup = async (name: string, email: string, password: string): Promise<UserProfile> => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Account registration failed.');
      }
      const newUser: UserProfile = data.user;
      setUser(newUser);
      localStorage.setItem('roboforge_user', JSON.stringify(newUser));
      if (data.token) {
        localStorage.setItem('roboforge_token', data.token);
      }
      return newUser;
    } finally {
      setIsLoading(false);
    }
  };

  const loginWithGoogleCredential = async (credential: string): Promise<UserProfile> => {
    setIsLoading(true);
    try {
      const payload = parseJwt(credential);
      const res = await fetch('/api/auth/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ credential }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Google authentication failed.');
      }
      const googleUser: UserProfile = data.user || {
        id: payload?.sub || 'google_user',
        name: payload?.name || 'Google User',
        email: payload?.email || '',
        picture: payload?.picture,
        provider: 'google',
      };
      setUser(googleUser);
      localStorage.setItem('roboforge_user', JSON.stringify(googleUser));
      if (data.token) {
        localStorage.setItem('roboforge_token', data.token);
      }
      return googleUser;
    } finally {
      setIsLoading(false);
    }
  };

  const loginWithGoogleUserInfo = async (userInfo: { sub: string; email: string; name: string; picture?: string }): Promise<UserProfile> => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/auth/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userInfo),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Google authentication failed.');
      }
      const googleUser: UserProfile = data.user;
      setUser(googleUser);
      localStorage.setItem('roboforge_user', JSON.stringify(googleUser));
      if (data.token) {
        localStorage.setItem('roboforge_token', data.token);
      }
      return googleUser;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    const token = localStorage.getItem('roboforge_token');
    setUser(null);
    localStorage.removeItem('roboforge_user');
    localStorage.removeItem('roboforge_token');
    if (token) {
      fetch('/api/auth/logout', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
      }).catch(() => {});
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        signup,
        loginWithGoogleCredential,
        loginWithGoogleUserInfo,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
