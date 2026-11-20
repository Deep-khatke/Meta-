import React, { useState } from 'react';
import {
  ThumbsUp,
  Lightbulb,
  Play,
  HelpCircle,
  CheckCircle2,
  Sparkles,
  GraduationCap
} from 'lucide-react';

interface EducatorFeedbackBarProps {
  messageId: string;
  onSendFollowUp: (text: string) => void;
  feedbackGiven?: 'understood' | 'simpler' | 'experiment';
  onRecordFeedback?: (messageId: string, rating: 'understood' | 'simpler' | 'experiment') => void;
}

export const EducatorFeedbackBar: React.FC<EducatorFeedbackBarProps> = ({
  messageId,
  onSendFollowUp,
  feedbackGiven: initialFeedback,
  onRecordFeedback,
}) => {
  const [feedback, setFeedback] = useState<'understood' | 'simpler' | 'experiment' | null>(
    initialFeedback || null
  );

  const handleSelect = (choice: 'understood' | 'simpler' | 'experiment') => {
    setFeedback(choice);
    onRecordFeedback?.(messageId, choice);

    if (choice === 'simpler') {
      onSendFollowUp(
        'Can you explain that again using a simpler everyday analogy like a bicycle, a shopping cart, or a toy?'
      );
    } else if (choice === 'experiment') {
      onSendFollowUp(
        'Can you demonstrate this concept live in the 3D physics simulation right now so I can see it in action?'
      );
    }
  };

  return (
    <div
      style={{
        marginTop: 12,
        padding: '10px 14px',
        borderRadius: 10,
        backgroundColor: 'rgba(56, 189, 248, 0.08)',
        border: '1px solid rgba(56, 189, 248, 0.22)',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        animation: 'rf-feedback-pop 0.3s ease-out',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <GraduationCap size={14} color="#38BDF8" />
          <span style={{ fontSize: 11.5, fontWeight: 700, color: '#38BDF8' }}>
            RoboForge Educator Check-in
          </span>
        </div>
        {feedback === 'understood' ? (
          <span
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 10.5,
              fontWeight: 600,
              color: '#34D399',
            }}
          >
            <CheckCircle2 size={12} color="#34D399" />
            <span>Concept Understood! Adapting pace.</span>
          </span>
        ) : (
          <span style={{ fontSize: 10.5, color: '#94A3B8' }}>
            Did this make sense?
          </span>
        )}
      </div>

      {feedback === 'understood' ? (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 10px',
            borderRadius: 6,
            background: 'rgba(52, 211, 153, 0.12)',
            border: '1px solid rgba(52, 211, 153, 0.3)',
            fontSize: 11,
            color: '#A7F3D0',
          }}
        >
          <Sparkles size={12} color="#34D399" />
          <span>
            Awesome! Mastery registered. You are building real robotics intuition.
          </span>
        </div>
      ) : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <button
            onClick={() => handleSelect('understood')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '4px 10px',
              borderRadius: 6,
              background: '#182436',
              border: '1px solid rgba(56, 189, 248, 0.28)',
              color: '#F1F5F9',
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#1E3552';
              e.currentTarget.style.borderColor = '#38BDF8';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#182436';
              e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.28)';
            }}
          >
            <ThumbsUp size={11} color="#38BDF8" />
            <span>👍 Clear & Understood!</span>
          </button>

          <button
            onClick={() => handleSelect('simpler')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '4px 10px',
              borderRadius: 6,
              background: '#182436',
              border: '1px solid rgba(251, 191, 36, 0.28)',
              color: '#F1F5F9',
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#292B22';
              e.currentTarget.style.borderColor = '#FBBF24';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#182436';
              e.currentTarget.style.borderColor = 'rgba(251, 191, 36, 0.28)';
            }}
          >
            <Lightbulb size={11} color="#FBBF24" />
            <span>💡 Explain with simpler analogy</span>
          </button>

          <button
            onClick={() => handleSelect('experiment')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '4px 10px',
              borderRadius: 6,
              background: '#182436',
              border: '1px solid rgba(52, 211, 153, 0.28)',
              color: '#F1F5F9',
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#163328';
              e.currentTarget.style.borderColor = '#34D399';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#182436';
              e.currentTarget.style.borderColor = 'rgba(52, 211, 153, 0.28)';
            }}
          >
            <Play size={11} color="#34D399" />
            <span>🧪 Test it in 3D Sim!</span>
          </button>
        </div>
      )}
    </div>
  );
};

