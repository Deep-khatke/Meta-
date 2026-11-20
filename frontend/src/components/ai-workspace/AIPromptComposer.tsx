import React, { useState, useRef, useLayoutEffect, useEffect } from 'react';
import {
  ArrowUp,
  Square,
  Paperclip,
  Image as ImageIcon,
  FileCode,
  Sparkles,
  Bot,
  Sliders,
  ChevronDown,
  X,
  Zap,
  Cpu
} from 'lucide-react';

interface AIPromptComposerProps {
  onSendMessage: (text: string, options?: { model?: string; mode?: string; attachment?: any }) => void;
  isLoading: boolean;
  onStopGeneration?: () => void;
  projectName?: string;
  activeRobotName?: string;
  initialValue?: string;
  placeholder?: string;
  disabled?: boolean;
  isEducatorMode?: boolean;
  onToggleEducatorMode?: () => void;
}

export const AIPromptComposer: React.FC<AIPromptComposerProps> = ({
  onSendMessage,
  isLoading,
  onStopGeneration,
  projectName = 'New Robotics Workspace',
  activeRobotName,
  initialValue = '',
  placeholder,
  disabled = false,
  isEducatorMode = false,
  onToggleEducatorMode,
}) => {
  const [text, setText] = useState<string>(initialValue);
  const [selectedModel, setSelectedModel] = useState<string>('gemini-2.0-flash');
  const [selectedMode, setSelectedMode] = useState<string>(isEducatorMode ? 'educator' : 'autonomous');
  const [showModelMenu, setShowModelMenu] = useState<boolean>(false);
  const [showModeMenu, setShowModeMenu] = useState<boolean>(false);
  const [attachment, setAttachment] = useState<{ name: string; type: 'image' | 'urdf' | 'file'; data?: string } | null>(null);

  useEffect(() => {
    if (isEducatorMode) {
      setSelectedMode('educator');
    }
  }, [isEducatorMode]);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const modeMenuRef = useRef<HTMLDivElement>(null);

  // Close popup menus on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
      if (modeMenuRef.current && !modeMenuRef.current.contains(e.target as Node)) {
        setShowModeMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Sync initialValue
  useEffect(() => {
    if (initialValue) {
      setText(initialValue);
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  }, [initialValue]);

  // Auto-expand textarea
  useLayoutEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(Math.max(scrollHeight, 44), 160)}px`;
    }
  }, [text]);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed && !attachment) return;
    if (isLoading) return;

    onSendMessage(trimmed, {
      model: selectedModel,
      mode: selectedMode,
      attachment: attachment || undefined,
    });
    setText('');
    setAttachment(null);
    if (textareaRef.current) {
      textareaRef.current.style.height = '44px';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, type: 'file' | 'image' | 'urdf') => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setAttachment({
        name: file.name,
        type,
        data: event.target?.result as string,
      });
    };
    if (type === 'image') {
      reader.readAsDataURL(file);
    } else {
      reader.readAsText(file);
    }
    e.target.value = '';
  };

  const models = [
    { id: 'gemini-2.0-flash', name: 'Google Gemini 2.0 Flash', badge: 'Primary // Recommended', icon: <Sparkles size={12} color="#38BDF8" /> },
    { id: 'nvidia/nemotron-3.5-super-vl:free', name: 'NVIDIA Nemotron 3.5', badge: 'Reasoning', icon: <Cpu size={12} color="#76B900" /> },
    { id: 'meta/muse-spark-1.3', name: 'Meta Muse Spark 1.3', badge: 'Code', icon: <Bot size={12} color="#A855F7" /> },
    { id: 'ollama/local', name: 'Local Ollama', badge: 'Offline', icon: <Zap size={12} color="#F59E0B" /> },
  ];

  const modes = [
    { id: 'educator', label: '🎓 Educator Mode', desc: 'Teaches robotics step-by-step for beginners using analogies, while building in 3D simulation' },
    { id: 'autonomous', label: 'Autonomous Tool-Use', desc: 'Directly executes robotics tools & updates physics scene' },
    { id: 'plan', label: 'Plan & Reason First', desc: 'Outlines verified multi-step trajectory before acting' },
    { id: 'direct', label: 'Direct Advice / Q&A', desc: 'Answers robotics theory, FK/IK maths and ROS 2 specs' },
  ];

  const dynamicPlaceholder =
    placeholder ||
    (selectedMode === 'educator'
      ? 'Ask any robotics question, or say "Teach me how a rover steers"…'
      : 'Message RoboForge AI… (e.g. "Build an obstacle arena and train the drone to hover")');

  return (
    <div
      className="rf-ai-prompt-composer"
      style={{
        width: '100%',
        maxWidth: 820,
        margin: '0 auto',
        padding: '0 16px 14px 16px',
        position: 'relative',
        zIndex: 20,
      }}
    >
      <div
        style={{
          borderRadius: 16,
          backgroundColor: '#16171B',
          border: selectedMode === 'educator' ? '1px solid rgba(56, 189, 248, 0.45)' : '1px solid #303136',
          boxShadow: selectedMode === 'educator' ? '0 8px 30px rgba(56, 189, 248, 0.12), 0 0 0 1px rgba(56, 189, 248, 0.15)' : '0 8px 30px rgba(0, 0, 0, 0.45)',
          transition: 'all 0.2s ease',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
        }}
      >
        {/* Educator Mode Active Banner */}
        {selectedMode === 'educator' && (
          <div
            style={{
              padding: '5px 14px',
              background: 'linear-gradient(90deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.1) 100%)',
              borderBottom: '1px solid rgba(56, 189, 248, 0.2)',
              borderTopLeftRadius: 15,
              borderTopRightRadius: 15,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: 11,
              color: '#38BDF8',
              fontWeight: 600,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>🎓</span>
              <span>Educator Mode Active — Explaining robotics simply with analogies & autonomous demonstrations</span>
            </div>
            <span style={{ fontSize: 10, color: '#94A3B8', fontWeight: 500 }}>Beginner Friendly</span>
          </div>
        )}
        {/* Attachment Pill if loaded */}
        {attachment && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              margin: '8px 12px 0 12px',
              padding: '4px 10px',
              borderRadius: 8,
              background: '#202228',
              border: '1px solid #383A42',
              fontSize: 11,
              color: '#38BDF8',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              {attachment.type === 'image' ? <ImageIcon size={13} /> : <FileCode size={13} />}
              <span style={{ fontWeight: 600, color: '#F4F4F5' }}>{attachment.name}</span>
              <span style={{ fontSize: 9.5, color: '#71717A', textTransform: 'uppercase' }}>({attachment.type})</span>
            </div>
            <button
              onClick={() => setAttachment(null)}
              style={{ background: 'transparent', border: 'none', color: '#71717A', cursor: 'pointer', padding: 2 }}
            >
              <X size={13} />
            </button>
          </div>
        )}

        {/* Text Input Area */}
        <div style={{ padding: '10px 14px 4px 14px' }}>
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={dynamicPlaceholder}
            disabled={disabled || isLoading}
            rows={1}
            style={{
              width: '100%',
              background: 'transparent',
              border: 'none',
              outline: 'none',
              resize: 'none',
              color: '#F5F5F5',
              fontSize: 13.5,
              lineHeight: 1.5,
              fontFamily: 'inherit',
            }}
          />
        </div>

        {/* Bottom Controls Bar: Attachments, Model Selector, Mode Selector, Send Button */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '6px 12px 10px 12px',
          }}
        >
          {/* Left Actions: Attachments & Model/Mode Pills */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            {/* Attachment Button (Files/URDF) */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              title="Attach File or URDF Definition"
              style={{
                width: 28,
                height: 28,
                borderRadius: 6,
                background: 'transparent',
                border: '1px solid #2B2C32',
                color: '#A1A1AA',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
            >
              <Paperclip size={13} />
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => handleFileUpload(e, 'file')}
              accept=".urdf,.xml,.json,.py,.txt"
              style={{ display: 'none' }}
            />

            {/* Image Attachment (Gemini 2.0 Multimodal Vision) */}
            <button
              type="button"
              onClick={() => imageInputRef.current?.click()}
              title="Upload Image/Sketch for Gemini Multimodal Scene Generation"
              style={{
                width: 28,
                height: 28,
                borderRadius: 6,
                background: 'transparent',
                border: '1px solid #2B2C32',
                color: '#A1A1AA',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
            >
              <ImageIcon size={13} />
            </button>
            <input
              type="file"
              ref={imageInputRef}
              onChange={(e) => handleFileUpload(e, 'image')}
              accept="image/png,image/jpeg,image/webp"
              style={{ display: 'none' }}
            />

            {/* Model Selector Pill */}
            <div ref={modelMenuRef} style={{ position: 'relative' }}>
              <button
                type="button"
                onClick={() => {
                  setShowModelMenu(!showModelMenu);
                  setShowModeMenu(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  padding: '3px 8px',
                  borderRadius: 6,
                  background: '#1F2025',
                  border: '1px solid #303138',
                  color: '#D4D4D8',
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                <Sparkles size={11} color="#38BDF8" />
                <span>{models.find((m) => m.id === selectedModel)?.name.replace('Google ', '') || 'Gemini 2.0'}</span>
                <ChevronDown size={11} color="#71717A" />
              </button>

              {showModelMenu && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: 'calc(100% + 8px)',
                    left: 0,
                    width: 260,
                    backgroundColor: '#18191E',
                    border: '1px solid #36373F',
                    borderRadius: 10,
                    padding: 5,
                    boxShadow: '0 16px 40px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.08)',
                    zIndex: 150,
                  }}
                >
                  <div style={{ padding: '5px 8px 3px 8px', fontSize: 10, color: '#71717A', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Select AI Reasoning Brain
                  </div>
                  {models.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => {
                        setSelectedModel(m.id);
                        setShowModelMenu(false);
                      }}
                      style={{
                        width: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '6px 8px',
                        borderRadius: 6,
                        background: selectedModel === m.id ? '#262830' : 'transparent',
                        border: 'none',
                        color: selectedModel === m.id ? '#FFFFFF' : '#A1A1AA',
                        fontSize: 11.5,
                        textAlign: 'left',
                        cursor: 'pointer',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        {m.icon}
                        <span>{m.name}</span>
                      </div>
                      <span style={{ fontSize: 9, color: '#71717A', fontFamily: 'monospace' }}>{m.badge}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Mode Selector Pill */}
            <div ref={modeMenuRef} style={{ position: 'relative' }}>
              <button
                type="button"
                onClick={() => {
                  setShowModeMenu(!showModeMenu);
                  setShowModelMenu(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  padding: '3px 8px',
                  borderRadius: 6,
                  background: selectedMode === 'educator' ? 'rgba(56, 189, 248, 0.15)' : '#1F2025',
                  border: selectedMode === 'educator' ? '1px solid rgba(56, 189, 248, 0.45)' : '1px solid #303138',
                  color: selectedMode === 'educator' ? '#38BDF8' : '#D4D4D8',
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer',
                  boxShadow: selectedMode === 'educator' ? '0 0 10px rgba(56, 189, 248, 0.2)' : 'none',
                  transition: 'all 0.15s ease',
                }}
              >
                {selectedMode === 'educator' ? (
                  <span style={{ fontSize: 11 }}>🎓</span>
                ) : (
                  <Sliders size={11} color="#D35432" />
                )}
                <span>
                  {selectedMode === 'educator'
                    ? 'Educator'
                    : selectedMode === 'autonomous'
                    ? 'Autonomous'
                    : selectedMode === 'plan'
                    ? 'Plan First'
                    : 'Direct Q&A'}
                </span>
                <ChevronDown size={11} color={selectedMode === 'educator' ? '#38BDF8' : '#71717A'} />
              </button>

              {showModeMenu && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: 'calc(100% + 8px)',
                    left: 0,
                    width: 290,
                    backgroundColor: '#18191E',
                    border: '1px solid #36373F',
                    borderRadius: 12,
                    padding: 6,
                    boxShadow: '0 20px 48px rgba(0, 0, 0, 0.85), 0 0 0 1px rgba(255, 255, 255, 0.08)',
                    zIndex: 150,
                  }}
                >
                  <div
                    style={{
                      padding: '5px 8px 4px 8px',
                      fontSize: 10,
                      color: '#71717A',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      borderBottom: '1px solid #282930',
                      marginBottom: 4,
                    }}
                  >
                    Select Agent Mode
                  </div>
                  {modes.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => {
                        setSelectedMode(m.id);
                        setShowModeMenu(false);
                        if (m.id === 'educator' && onToggleEducatorMode && !isEducatorMode) {
                          onToggleEducatorMode();
                        } else if (m.id !== 'educator' && onToggleEducatorMode && isEducatorMode) {
                          onToggleEducatorMode();
                        }
                      }}
                      style={{
                        width: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2,
                        padding: '7px 9px',
                        borderRadius: 8,
                        background: selectedMode === m.id
                          ? (m.id === 'educator' ? 'rgba(56, 189, 248, 0.16)' : '#262830')
                          : 'transparent',
                        border: selectedMode === m.id && m.id === 'educator'
                          ? '1px solid rgba(56, 189, 248, 0.35)'
                          : '1px solid transparent',
                        color: selectedMode === m.id
                          ? (m.id === 'educator' ? '#38BDF8' : '#FFFFFF')
                          : '#A1A1AA',
                        textAlign: 'left',
                        cursor: 'pointer',
                        transition: 'background 0.15s ease',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: 11.5, fontWeight: 700 }}>{m.label}</span>
                        {m.id === 'educator' && (
                          <span
                            style={{
                              fontSize: 9,
                              fontWeight: 700,
                              color: '#38BDF8',
                              background: 'rgba(56, 189, 248, 0.15)',
                              padding: '1px 5px',
                              borderRadius: 4,
                            }}
                          >
                            BEGINNER
                          </span>
                        )}
                      </div>
                      <span style={{ fontSize: 10, color: '#71717A', lineHeight: 1.3 }}>{m.desc}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Project Context Badge */}
            <div
              style={{
                fontSize: 10.5,
                color: '#71717A',
                padding: '2px 7px',
                borderRadius: 4,
                background: 'rgba(255, 255, 255, 0.04)',
                display: 'none', // Shown on desktop via media queries
              }}
            >
              Context: <span style={{ color: '#A1A1AA' }}>{activeRobotName || projectName}</span>
            </div>
          </div>

          {/* Right Action: Send Button or Stop Button */}
          <div>
            {isLoading ? (
              <button
                type="button"
                onClick={onStopGeneration}
                title="Stop generation"
                style={{
                  width: 30,
                  height: 30,
                  borderRadius: 8,
                  background: '#EF4444',
                  border: 'none',
                  color: '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  boxShadow: '0 2px 8px rgba(239, 68, 68, 0.4)',
                }}
              >
                <Square size={12} fill="#FFFFFF" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={(!text.trim() && !attachment) || disabled}
                title="Send message (Enter)"
                style={{
                  width: 30,
                  height: 30,
                  borderRadius: 8,
                  background: (!text.trim() && !attachment)
                    ? '#26282E'
                    : 'linear-gradient(135deg, #D35432 0%, #A8381C 100%)',
                  border: 'none',
                  color: (!text.trim() && !attachment) ? '#71717A' : '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: (!text.trim() && !attachment) ? 'default' : 'pointer',
                  boxShadow: (!text.trim() && !attachment)
                    ? 'none'
                    : '0 2px 10px rgba(211, 84, 50, 0.45)',
                  transition: 'all 0.15s ease',
                }}
              >
                <ArrowUp size={15} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

