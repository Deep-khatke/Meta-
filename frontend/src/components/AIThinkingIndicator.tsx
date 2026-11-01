import React, { useEffect, useState } from 'react';
import { Sparkles } from 'lucide-react';

/**
 * A friendly, living "thinking / generating" indicator for the AI co-pilot.
 * A breathing gradient orb with an orbiting spark, a shimmering status line that cycles through
 * warm, human phrases, and bouncing typing dots. Purely visual; all animation is CSS.
 */
const PHRASES = [
  'Reading your scene',
  'Thinking it through',
  'Checking the robot',
  'Reasoning about physics',
  'Planning the next step',
  'Working on it',
  'Putting it together',
];

interface Props {
  /** Optional override message; when provided, the cycling phrases are paused. */
  message?: string;
}

export const AIThinkingIndicator: React.FC<Props> = ({ message }) => {
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    if (message) return;
    const t = setInterval(() => setIdx((i) => (i + 1) % PHRASES.length), 1800);
    return () => clearInterval(t);
  }, [message]);

  return (
    <div className="rf-ai-thinking">
      <div className="rf-ai-orb" aria-hidden>
        <Sparkles size={13} />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <span className="rf-ai-thinking-text" key={message || idx}>
          {message || PHRASES[idx]}
        </span>
        <span className="rf-ai-dots" aria-hidden>
          <span />
          <span />
          <span />
        </span>
      </div>
    </div>
  );
};
