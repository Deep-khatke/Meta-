import React, { useState, useEffect, useRef } from 'react';
import {
  Sparkles,
  Bot,
  User,
  Car,
  Wrench,
  Plane,
  Layers,
  Upload,
  Activity,
  Box,
  TrendingUp,
  Cpu,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Copy,
  Check,
  RefreshCw,
  Loader2,
  Zap,
} from 'lucide-react';
import { AIPromptComposer } from './AIPromptComposer';
import { AIExecutionStep, AIExecutionStepItem } from './AIExecutionStep';
import { ArtifactTabType } from './AIArtifactPanel';
import { AnimatedSuggestionDeck } from './AnimatedSuggestionDeck';
import { EducatorFeedbackBar } from './EducatorFeedbackBar';
import { GraduationCap } from 'lucide-react';
import { AIChatRequest, AIChatResponse } from '../../types';
import { sendAIChatMessage } from '../../services/api';

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: number;
  steps?: AIExecutionStepItem[];
  artifactTrigger?: {
    tab: ArtifactTabType;
    label: string;
  };
  codeSnippet?: {
    lang: string;
    code: string;
  };
  isEducatorMode?: boolean;
  followUpSuggestions?: string[];
  feedback?: 'understood' | 'simpler' | 'experiment';
  isStreaming?: boolean;
}

interface AIConversationPanelProps {
  messages: ChatMessage[];
  onSendMessage: (text: string, options?: any) => void;
  isLoading: boolean;
  onStopGeneration?: () => void;
  projectName?: string;
  activeRobotName?: string;
  onTriggerArtifact: (tab: ArtifactTabType) => void;
  isEducatorMode?: boolean;
  onToggleEducatorMode?: () => void;
  onRecordFeedback?: (messageId: string, rating: 'understood' | 'simpler' | 'experiment') => void;
}

export const AIConversationPanel: React.FC<AIConversationPanelProps> = ({
  messages,
  onSendMessage,
  isLoading,
  onStopGeneration,
  projectName = 'New Robotics Workspace',
  activeRobotName,
  onTriggerArtifact,
  isEducatorMode = false,
  onToggleEducatorMode,
  onRecordFeedback,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const suggestionCards = isEducatorMode ? [
    {
      title: 'How does a rover steer?',
      desc: 'Rowboat analogy & differential wheel speeds',
      prompt: 'Teach me how a differential drive rover steers using an everyday analogy',
      icon: <Car size={16} color="#38BDF8" />,
      accent: 'rgba(56, 189, 248, 0.15)',
    },
    {
      title: 'Why 4 rotors on a drone?',
      desc: 'Torque cancellation, lift & altitude hold physics',
      prompt: 'Why do quadcopter drones need 4 rotors spinning in opposite directions?',
      icon: <Plane size={16} color="#38BDF8" />,
      accent: 'rgba(56, 189, 248, 0.15)',
    },
    {
      title: 'What are Degrees of Freedom?',
      desc: 'Human arm vs 6-DOF industrial serial manipulator',
      prompt: 'Explain Degrees of Freedom (DoF) in robotics like I am a beginner',
      icon: <Wrench size={16} color="#A855F7" />,
      accent: 'rgba(168, 85, 247, 0.15)',
    },
    {
      title: 'How does LiDAR see walls?',
      desc: 'Laser tape measure in 360° point cloud',
      prompt: 'Explain how LiDAR sensors work and spawn an obstacle arena to demonstrate',
      icon: <Zap size={16} color="#FBBF24" />,
      accent: 'rgba(251, 191, 36, 0.15)',
    },
    {
      title: 'What is a PID Controller?',
      desc: 'Car cruise control maintaining speed uphill',
      prompt: 'Explain PID controllers using the car cruise control analogy',
      icon: <Activity size={16} color="#34D399" />,
      accent: 'rgba(52, 211, 153, 0.15)',
    },
    {
      title: 'Why do robots slip in physics?',
      desc: 'Friction, normal force & contact dynamics',
      prompt: 'Why do robot wheels slip on smooth surfaces in MuJoCo physics?',
      icon: <Box size={16} color="#F97316" />,
      accent: 'rgba(249, 115, 22, 0.15)',
    },
  ] : [
    {
      title: 'Train an autonomous rover',
      desc: 'Deploy PPO reinforcement learning with LiDAR obstacle avoidance',
      prompt: 'Train an autonomous rover to navigate obstacles using PPO in MuJoCo',
      icon: <Car size={16} color="#38BDF8" />,
      accent: 'rgba(56, 189, 248, 0.15)',
    },
    {
      title: 'Build a robotic arm',
      desc: 'Assemble a multi-joint manipulator with Damped Least-Squares IK',
      prompt: 'Construct a 6-DOF articulated serial manipulator with inverse kinematics',
      icon: <Wrench size={16} color="#D35432" />,
      accent: 'rgba(211, 84, 50, 0.15)',
    },
    {
      title: 'Simulate an autonomous drone',
      desc: 'Quadcopter flight with 4 aerodynamic rotor thrust physics & FPV',
      prompt: 'Spawn a quadcopter drone with authentic aerodynamic rotor physics and takeoff to 1.5m',
      icon: <Plane size={16} color="#A855F7" />,
      accent: 'rgba(168, 85, 247, 0.15)',
    },
    {
      title: 'Create a custom environment',
      desc: 'Generate a 3D warehouse arena with maze walls and obstacles',
      prompt: 'Create a warehouse environment with 4 obstacle pillars and safe boundaries',
      icon: <Layers size={16} color="#4ADE80" />,
      accent: 'rgba(74, 222, 128, 0.15)',
    },
    {
      title: 'Import a robot model',
      desc: 'Load URDF or MJCF robot kinematic definition into MuJoCo 3.17',
      prompt: 'Can you show me how to import and configure a custom URDF model?',
      icon: <Upload size={16} color="#F59E0B" />,
      accent: 'rgba(245, 158, 11, 0.15)',
    },
    {
      title: 'Analyse training performance',
      desc: 'Evaluate sim-to-real robustness scorecard across friction and noise',
      prompt: 'Analyse the reinforcement learning training metrics and sim-to-real scorecard',
      icon: <Activity size={16} color="#EC4899" />,
      accent: 'rgba(236, 72, 153, 0.15)',
    },
  ];

  const isEmpty = messages.length === 0;

  return (
    <div
      className="rf-ai-conversation-panel"
      style={{
        flex: 1,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#101113',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* ── Scrollable Center Area ── */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '24px 20px 20px 20px',
        }}
      >
        {isEmpty ? (
          /* ── Initial Welcome Screen ── */
          <div
            style={{
              width: '100%',
              maxWidth: 780,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              marginTop: 'auto',
              marginBottom: 'auto',
              padding: '20px 0',
              textAlign: 'center',
            }}
          >
            {/* RoboForge Logo Glow */}
            <div
              style={{
                width: 52,
                height: 52,
                borderRadius: 14,
                background: 'linear-gradient(135deg, #D35432 0%, #A8381C 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 6px 24px rgba(211, 84, 50, 0.45)',
                marginBottom: 18,
              }}
            >
              <Cpu size={26} color="#FFFFFF" />
            </div>

            {/* Large Typography */}
            <h1
              style={{
                fontSize: 28,
                fontWeight: 800,
                color: '#FFFFFF',
                letterSpacing: '-0.5px',
                marginBottom: 6,
              }}
            >
              RoboForge AI
            </h1>

            <h2
              style={{
                fontSize: 20,
                fontWeight: 600,
                color: '#E4E4E7',
                marginBottom: 8,
              }}
            >
              What will you build today?
            </h2>

            <p
              style={{
                fontSize: 13,
                color: '#A1A1AA',
                maxWidth: 480,
                lineHeight: 1.5,
                marginBottom: 28,
              }}
            >
              Build, simulate and train intelligent robots using natural language powered by Google Gemini and MuJoCo 3.17 physics.
            </p>

            {/* Prompt Composer on Welcome Screen */}
            <div style={{ width: '100%', marginBottom: 28 }}>
              <AIPromptComposer
                onSendMessage={onSendMessage}
                isLoading={isLoading}
                onStopGeneration={onStopGeneration}
                projectName={projectName}
                activeRobotName={activeRobotName}
                isEducatorMode={isEducatorMode}
                onToggleEducatorMode={onToggleEducatorMode}
              />
            </div>

            {/* 6 Elegant Suggestion Cards */}
            <div
              style={{
                width: '100%',
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: 12,
                textAlign: 'left',
              }}
            >
              {suggestionCards.map((card, idx) => (
                <div
                  key={idx}
                  onClick={() => onSendMessage(card.prompt)}
                  style={{
                    padding: '12px 14px',
                    borderRadius: 10,
                    backgroundColor: '#16171B',
                    border: '1px solid #2B2C32',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#1D1E24';
                    e.currentTarget.style.borderColor = '#3D3E48';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#16171B';
                    e.currentTarget.style.borderColor = '#2B2C32';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div
                      style={{
                        width: 28,
                        height: 28,
                        borderRadius: 6,
                        backgroundColor: card.accent,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      {card.icon}
                    </div>
                    <span style={{ fontSize: 12.5, fontWeight: 700, color: '#F5F5F5' }}>
                      {card.title}
                    </span>
                  </div>
                  <p style={{ fontSize: 11, color: '#A1A1AA', lineHeight: 1.4, margin: 0 }}>
                    {card.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          /* ── Active Conversation Stream ── */
          <div
            style={{
              width: '100%',
              maxWidth: 820,
              display: 'flex',
              flexDirection: 'column',
              gap: 20,
            }}
          >
            {messages.map((msg) => {
              const isEducatorMsg = msg.isEducatorMode || isEducatorMode;

              // Check if the message ends with or contains "What would you like to build or explore?"
              // so we can render it as a smooth animated suggestion deck rather than static flash text
              const exploreMatch = msg.text.match(/(?:What would you like to (?:build or )?explore\??)/i);
              let displayText = msg.text;
              let hasExploreInText = false;
              if (exploreMatch && exploreMatch.index !== undefined) {
                hasExploreInText = true;
                displayText = msg.text.slice(0, exploreMatch.index).trim();
              }

              const activeSuggestions = msg.followUpSuggestions || (hasExploreInText ? (
                isEducatorMsg ? [
                  '🚗 How does differential steering work? Build an arena',
                  '🚁 Why do drones have 4 rotors? Test hover in 3D',
                  '🦾 What does Degrees of Freedom mean? Move the robot arm',
                  '⚡ Explain how LiDAR sees through the room'
                ] : [
                  'Train an autonomous rover with PPO',
                  'Assemble a 6-DOF robotic arm',
                  'Simulate an autonomous drone hover',
                  'Spawn randomized obstacle arena'
                ]
              ) : undefined);

              return (
                <div
                  key={msg.id}
                  style={{
                    display: 'flex',
                    gap: 14,
                    alignItems: 'flex-start',
                  }}
                >
                  {/* Sender Avatar */}
                  <div
                    style={{
                      width: 30,
                      height: 30,
                      borderRadius: 8,
                      background: msg.sender === 'user'
                        ? '#2A2C34'
                        : isEducatorMsg
                        ? 'linear-gradient(135deg, #0284C7 0%, #0369A1 100%)'
                        : 'linear-gradient(135deg, #D35432 0%, #A8381C 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                      marginTop: 2,
                      boxShadow: msg.sender === 'assistant'
                        ? isEducatorMsg
                          ? '0 2px 8px rgba(2, 132, 199, 0.45)'
                          : '0 2px 8px rgba(211, 84, 50, 0.35)'
                        : 'none',
                    }}
                  >
                    {msg.sender === 'user' ? (
                      <User size={15} color="#D4D4D8" />
                    ) : isEducatorMsg ? (
                      <GraduationCap size={16} color="#FFFFFF" />
                    ) : (
                      <Bot size={15} color="#FFFFFF" />
                    )}
                  </div>

                  {/* Message Body */}
                  <div style={{ flex: 1, overflow: 'hidden' }}>
                    <div
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        color: msg.sender === 'user' ? '#A1A1AA' : isEducatorMsg ? '#38BDF8' : '#D35432',
                        marginBottom: 4,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                      }}
                    >
                      <span>{msg.sender === 'user' ? 'You' : 'RoboForge AI'}</span>
                      {msg.sender === 'assistant' && isEducatorMsg && (
                        <span
                          style={{
                            fontSize: 9.5,
                            padding: '1px 5px',
                            borderRadius: 4,
                            background: 'rgba(56, 189, 248, 0.18)',
                            color: '#38BDF8',
                            fontWeight: 700,
                            letterSpacing: '0.04em',
                          }}
                        >
                          EDUCATOR
                        </span>
                      )}
                    </div>

                    {/* Text Content */}
                    <div
                      style={{
                        fontSize: 13.5,
                        lineHeight: 1.6,
                        color: '#F4F4F5',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                      }}
                    >
                      {displayText}
                      {msg.isStreaming && (
                        <span
                          className="rf-streaming-cursor"
                          style={{
                            backgroundColor: isEducatorMsg ? '#38BDF8' : '#D35432',
                            boxShadow: isEducatorMsg ? '0 0 8px rgba(56, 189, 248, 0.6)' : '0 0 8px rgba(211, 84, 50, 0.6)',
                          }}
                        />
                      )}
                    </div>

                    {/* Execution Steps if present */}
                    {msg.steps && msg.steps.length > 0 && (
                      <AIExecutionStep steps={msg.steps} />
                    )}

                    {/* Interactive Artifact Pill Trigger */}
                    {msg.artifactTrigger && (
                      <div style={{ marginTop: 10 }}>
                        <button
                          onClick={() => onTriggerArtifact(msg.artifactTrigger!.tab)}
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 7,
                            padding: '6px 12px',
                            borderRadius: 8,
                            background: isEducatorMsg ? 'rgba(56, 189, 248, 0.15)' : 'rgba(211, 84, 50, 0.15)',
                            border: `1px solid ${isEducatorMsg ? 'rgba(56, 189, 248, 0.45)' : 'rgba(211, 84, 50, 0.45)'}`,
                            color: '#F4F4F5',
                            fontSize: 12,
                            fontWeight: 600,
                            cursor: 'pointer',
                            boxShadow: isEducatorMsg ? '0 2px 10px rgba(56, 189, 248, 0.25)' : '0 2px 10px rgba(211, 84, 50, 0.25)',
                            transition: 'all 0.15s ease',
                          }}
                          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = isEducatorMsg ? 'rgba(56, 189, 248, 0.25)' : 'rgba(211, 84, 50, 0.25)')}
                          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = isEducatorMsg ? 'rgba(56, 189, 248, 0.15)' : 'rgba(211, 84, 50, 0.15)')}
                        >
                          <Sparkles size={13} color={isEducatorMsg ? '#38BDF8' : '#D35432'} />
                          <span>{msg.artifactTrigger.label}</span>
                          <ExternalLink size={12} color="#A1A1AA" />
                        </button>
                      </div>
                    )}

                    {/* Smooth Animated Exploration Suggestion Deck (fades in once streaming is complete) */}
                    {!msg.isStreaming && msg.sender === 'assistant' && activeSuggestions && activeSuggestions.length > 0 && (
                      <AnimatedSuggestionDeck
                        suggestions={activeSuggestions}
                        onSelectSuggestion={(s) => onSendMessage(s)}
                        isEducatorMode={isEducatorMsg}
                        title={isEducatorMsg ? '🎓 What would you like to explore or test next?' : '💡 What would you like to build or explore?'}
                      />
                    )}

                    {/* Educator Feedback & Learning Satisfaction Loop (appears once streaming is complete) */}
                    {!msg.isStreaming && msg.sender === 'assistant' && isEducatorMsg && (
                      <EducatorFeedbackBar
                        messageId={msg.id}
                        onSendFollowUp={(text) => onSendMessage(text)}
                        feedbackGiven={msg.feedback}
                        onRecordFeedback={onRecordFeedback}
                      />
                    )}
                  </div>
                </div>
              );
            })}

            {/* Loading Indicator */}
            {isLoading && (
              <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
                <div
                  style={{
                    width: 30,
                    height: 30,
                    borderRadius: 8,
                    background: isEducatorMode ? 'linear-gradient(135deg, #0284C7 0%, #0369A1 100%)' : 'linear-gradient(135deg, #D35432 0%, #A8381C 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: isEducatorMode ? '0 2px 8px rgba(2, 132, 199, 0.45)' : '0 2px 8px rgba(211, 84, 50, 0.35)',
                  }}
                >
                  <Loader2 size={15} color="#FFFFFF" className="animate-spin" />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#A1A1AA', fontSize: 12.5 }}>
                  <span>{isEducatorMode ? 'RoboForge Educator is crafting an intuitive explanation & simulation demonstration…' : 'RoboForge AI is planning and generating physics trajectory…'}</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* ── Fixed Prompt Composer (in Conversation State) ── */}
      {!isEmpty && (
        <AIPromptComposer
          onSendMessage={onSendMessage}
          isLoading={isLoading}
          onStopGeneration={onStopGeneration}
          projectName={projectName}
          activeRobotName={activeRobotName}
          isEducatorMode={isEducatorMode}
          onToggleEducatorMode={onToggleEducatorMode}
        />
      )}
    </div>
  );
};

