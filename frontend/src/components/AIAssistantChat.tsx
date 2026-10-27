import React, { useState, useRef, useEffect } from 'react';
import {
  Send, Sparkles, CheckCircle2, AlertTriangle, Play, BrainCircuit, Award,
  Maximize2, Minimize2, Trash2, Bot, User, CheckSquare, Target, ShieldCheck,
  RefreshCw, ChevronRight, Layers, ArrowRight
} from 'lucide-react';
import { AIAssistantConfigResponse, ExperimentConfig, RLTrainingConfig } from '../types';
import { interpretAIPrompt, applyAIConfig, runLiveExperiment, startRLTraining, evaluateRLPolicy } from '../services/api';

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: number;
  response?: AIAssistantConfigResponse;
  isExecuting?: boolean;
}

interface AIAssistantChatProps {
  onShowToast: (msg: string) => void;
  selectedScenarioId?: string;
  targetPosition?: [number, number, number] | null;
  onProjectLoaded?: () => void;
  onSelectTarget?: (pos: [number, number, number] | null) => void;
  onSwitchView?: (view: 'metrics' | 'compare' | 'rl_training' | 'benchmark' | 'ai_assistant' | 'hypotheses') => void;
}

export const AIAssistantChat: React.FC<AIAssistantChatProps> = ({
  onShowToast,
  selectedScenarioId,
  targetPosition,
  onProjectLoaded,
  onSelectTarget,
  onSwitchView,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: "👋 Welcome to the **RoboForge AI Robotics Studio**! I can convert natural-language navigation goals into strictly validated physical MuJoCo simulation environments, train Stable-Baselines3 PPO policies, and execute autonomous runs live in your 3D viewport. What challenge would you like to build?",
      timestamp: Date.now(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const presetChips = [
    'Create a tight 3-obstacle slalom with a 2.5m target beacon',
    'Design a narrow corridor navigation challenge with obstacle walls',
    'Create a high-speed direct block navigation scenario',
    'Build a wide arena with 4 scattered cylindrical obstacles',
    'Why did the robot collide and how can we optimize avoidance?',
  ];

  const handleSendMessage = async (textToSend?: string) => {
    const prompt = (textToSend || inputValue).trim();
    if (!prompt || isLoading) return;

    const userMsg: ChatMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      text: prompt,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const res = await interpretAIPrompt(prompt, selectedScenarioId);
      const aiMsg: ChatMessage = {
        id: `ai_${Date.now()}`,
        sender: 'assistant',
        text: `I have synthesized and physically validated the experiment configuration for: **"${res.interpreted_goal}"**.\n\nAll coordinates have been clamped to arena boundaries with guaranteed robot clearance (zero arbitrary code execution). You can review the physical specifications below and launch directly into the 3D viewport.`,
        timestamp: Date.now(),
        response: res,
      };
      setMessages((prev) => [...prev, aiMsg]);
      onShowToast(`AI: Synthesized ${res.suggested_configuration?.environment?.length || 0} obstacles safely.`);
    } catch (e: any) {
      const errorMsg: ChatMessage = {
        id: `ai_err_${Date.now()}`,
        sender: 'assistant',
        text: `⚠️ Synthesis Error: ${e.message || 'Failed to interpret prompt'}. Please ensure parameters describe physical navigation in the arena.`,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMsg]);
      onShowToast(`AI Assistant error: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleApplyTo3D = async (config: any, autoRun: boolean = true) => {
    try {
      onShowToast('Applying validated configuration to MuJoCo physics scene...');
      const res = await applyAIConfig(config);
      if (onProjectLoaded) onProjectLoaded();

      const targetPos = config?.task?.target_position;
      if (targetPos && onSelectTarget) {
        onSelectTarget(targetPos);
      }

      onShowToast('3D Scene updated successfully!');

      if (autoRun && res?.scenario_id) {
        onShowToast('Launching live autonomous execution in 3D viewport...');
        await runLiveExperiment({
          scenario_id: res.scenario_id,
          duration: config?.task?.timeout || 12.0,
        });
        if (onProjectLoaded) onProjectLoaded();
      }
    } catch (e: any) {
      onShowToast(`Failed to apply to 3D scene: ${e.message}`);
    }
  };

  const handleTrainPPOForConfig = async (config: any) => {
    try {
      onShowToast('Launching background PPO training for AI scenario...');
      await startRLTraining({
        model_name: `PPO-${config?.name?.slice(0, 20) || 'AI-Agent'}`,
        total_timesteps: 5000,
        seed: 42,
        randomize_scenarios: true,
        target_position: config?.task?.target_position || [2.5, 0.5, 0.25],
        scenario_id: config?.id,
      });
      if (onSwitchView) onSwitchView('rl_training');
      onShowToast('PPO training worker active! Switched to training view.');
    } catch (e: any) {
      onShowToast(`Failed to start PPO training: ${e.message}`);
    }
  };

  const renderMessageCard = (msg: ChatMessage) => {
    const isUser = msg.sender === 'user';
    const resp = msg.response;
    const config = resp?.suggested_configuration;

    return (
      <div
        key={msg.id}
        style={{
          display: 'flex',
          gap: 10,
          alignItems: 'flex-start',
          flexDirection: isUser ? 'row-reverse' : 'row',
          marginBottom: 12,
        }}
      >
        {/* Avatar */}
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            background: isUser ? 'var(--accent-blue, #2563eb)' : 'linear-gradient(135deg, #7c3aed, #06b6d4)',
            color: '#fff',
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
          }}
        >
          {isUser ? <User size={15} /> : <Bot size={15} />}
        </div>

        {/* Message Bubble / Card */}
        <div
          style={{
            maxWidth: isUser ? '75%' : '85%',
            background: isUser ? 'var(--accent-blue, #2563eb)' : 'var(--bg-card, #1e222b)',
            color: isUser ? '#fff' : 'var(--text-primary, #e2e8f0)',
            padding: '10px 14px',
            borderRadius: isUser ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
            border: isUser ? 'none' : '1px solid var(--border-subtle, #2d3342)',
            fontSize: 12,
            lineHeight: 1.5,
            boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
          }}
        >
          <div style={{ whiteSpace: 'pre-wrap', marginBottom: resp ? 8 : 0 }}>
            {msg.text}
          </div>

          {/* Structured Experiment Response Card */}
          {resp && config && (
            <div
              style={{
                marginTop: 8,
                background: 'var(--bg-canvas, #14171f)',
                padding: 10,
                borderRadius: 8,
                border: '1px solid var(--border-subtle, #2d3342)',
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
              }}
            >
              {/* Validation Status Badge */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5,
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: 4,
                    background: resp.validation_status === 'valid' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(234, 179, 8, 0.2)',
                    color: resp.validation_status === 'valid' ? 'var(--accent-green, #10b981)' : 'var(--accent-yellow, #eab308)',
                  }}
                >
                  <ShieldCheck size={12} />
                  <span>{resp.validation_status === 'valid' ? 'VALIDATED (100% Boundary & Kinematic Safe)' : 'VALIDATION WARNING'}</span>
                </span>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                  Seed: {config.seed || 42} &bull; Timeout: {config.task?.timeout || 12}s
                </span>
              </div>

              {/* Physical Specifications */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, fontSize: 10 }}>
                <div style={{ background: 'var(--bg-card, #1e222b)', padding: '5px 8px', borderRadius: 4 }}>
                  <div style={{ color: 'var(--text-muted)' }}>🎯 Goal Beacon</div>
                  <div style={{ fontWeight: 600, color: 'var(--accent-green, #10b981)' }}>
                    [{config.task?.target_position?.slice(0, 2).map((v: number) => v.toFixed(2)).join(', ')}]
                  </div>
                </div>
                <div style={{ background: 'var(--bg-card, #1e222b)', padding: '5px 8px', borderRadius: 4 }}>
                  <div style={{ color: 'var(--text-muted)' }}>🧱 Obstacles Count</div>
                  <div style={{ fontWeight: 600 }}>{config.environment?.length || 0} objects</div>
                </div>
                <div style={{ background: 'var(--bg-card, #1e222b)', padding: '5px 8px', borderRadius: 4 }}>
                  <div style={{ color: 'var(--text-muted)' }}>🛡️ Safety Clearance</div>
                  <div style={{ fontWeight: 600 }}>{config.task?.policy?.safety_margin?.toFixed(2) || '0.75'}m</div>
                </div>
              </div>

              {/* Safety Summary */}
              <div style={{ fontSize: 10, color: 'var(--text-secondary, #94a3b8)' }}>
                <strong>Safety Verification:</strong> {resp.safety_summary}
              </div>

              {/* Interactive One-Click 3D Action Buttons */}
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
                <button
                  className="rf-btn rf-btn-primary"
                  onClick={() => handleApplyTo3D(config, true)}
                  style={{
                    fontSize: 10,
                    padding: '4px 10px',
                    fontWeight: 600,
                    background: 'linear-gradient(135deg, #0284c7, #2563eb)',
                    border: 'none',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5,
                  }}
                  title="Applies obstacle geometry to MuJoCo physics and immediately begins live autonomous movement in 3D"
                >
                  <Play size={12} fill="#fff" />
                  <span>Execute in 3D Viewport</span>
                </button>

                <button
                  className="rf-btn"
                  onClick={() => handleApplyTo3D(config, false)}
                  style={{ fontSize: 10, padding: '4px 8px', display: 'flex', alignItems: 'center', gap: 4 }}
                  title="Applies obstacle layout to scene without auto-running"
                >
                  <CheckSquare size={12} color="var(--accent-green, #10b981)" />
                  <span>Apply to Scene Only</span>
                </button>

                <button
                  className="rf-btn"
                  onClick={() => handleTrainPPOForConfig(config)}
                  style={{
                    fontSize: 10,
                    padding: '4px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    background: 'linear-gradient(135deg, #7c3aed, #4f46e5)',
                    color: '#fff',
                    border: 'none',
                  }}
                  title="Start background PPO training for this layout"
                >
                  <BrainCircuit size={12} />
                  <span>Train PPO Policy</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: isExpanded ? '85vh' : '100%',
        background: 'var(--bg-panel, #181b22)',
        borderRadius: 8,
        border: '1px solid var(--border-subtle, #2d3342)',
        position: isExpanded ? 'fixed' : 'relative',
        top: isExpanded ? '5vh' : 'auto',
        left: isExpanded ? '5vw' : 'auto',
        width: isExpanded ? '90vw' : '100%',
        zIndex: isExpanded ? 9999 : 1,
        boxShadow: isExpanded ? '0 16px 64px rgba(0,0,0,0.8)' : 'none',
        overflow: 'hidden',
      }}
    >
      {/* Studio Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          background: 'var(--bg-card, #1e222b)',
          borderBottom: '1px solid var(--border-subtle, #2d3342)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div
            style={{
              width: 24,
              height: 24,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #7c3aed, #06b6d4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
            }}
          >
            <Sparkles size={13} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>RoboForge AI Experiment Studio</span>
              <span style={{ fontSize: 9, padding: '1px 5px', borderRadius: 3, background: 'rgba(56, 189, 248, 0.2)', color: '#38bdf8' }}>
                GPT-Style Robotics Assistant
              </span>
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>
              Natural language synthesis &bull; Strict boundary validation &bull; 1-click live 3D execution
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button
            className="rf-btn"
            onClick={() => setMessages([messages[0]])}
            style={{ fontSize: 10, padding: '2px 6px' }}
            title="Clear Chat History"
          >
            <Trash2 size={11} />
            <span>Clear</span>
          </button>
          <button
            className="rf-btn"
            onClick={() => setIsExpanded(!isExpanded)}
            style={{ fontSize: 10, padding: '2px 6px' }}
            title={isExpanded ? 'Collapse Studio' : 'Expand to ChatGPT-style Full Window'}
          >
            {isExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
            <span>{isExpanded ? 'Collapse' : 'Expand Studio'}</span>
          </button>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: 12,
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
        }}
      >
        {messages.map(renderMessageCard)}

        {isLoading && (
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', color: 'var(--text-muted)', fontSize: 11 }}>
            <div
              style={{
                width: 24,
                height: 24,
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #7c3aed, #06b6d4)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                animation: 'pulse 1.5s infinite',
              }}
            >
              <Sparkles size={13} />
            </div>
            <span>Synthesizing physics parameters, obstacles, and validating arena boundaries...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Preset Chips Bar */}
      <div
        style={{
          display: 'flex',
          gap: 4,
          padding: '6px 10px',
          background: 'var(--bg-canvas, #14171f)',
          borderTop: '1px solid var(--border-subtle, #2d3342)',
          overflowX: 'auto',
          whiteSpace: 'nowrap',
        }}
      >
        <span style={{ fontSize: 9, color: 'var(--text-muted)', alignSelf: 'center', flexShrink: 0 }}>
          💡 Try:
        </span>
        {presetChips.map((chip, idx) => (
          <button
            key={idx}
            type="button"
            className="rf-btn"
            onClick={() => handleSendMessage(chip)}
            disabled={isLoading}
            style={{
              fontSize: 9,
              padding: '2px 8px',
              borderRadius: 12,
              background: 'var(--bg-card, #1e222b)',
              color: 'var(--text-secondary, #94a3b8)',
              border: '1px solid var(--border-subtle, #2d3342)',
              flexShrink: 0,
            }}
          >
            {chip}
          </button>
        ))}
      </div>

      {/* Input Composer */}
      <div
        style={{
          padding: '8px 10px',
          background: 'var(--bg-card, #1e222b)',
          borderTop: '1px solid var(--border-subtle, #2d3342)',
          display: 'flex',
          gap: 6,
          alignItems: 'center',
        }}
      >
        <input
          type="text"
          className="rf-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask AI: 'Create a tight slalom challenge', 'Add 2 obstacles ahead', 'Run in 3D'..."
          disabled={isLoading}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSendMessage();
          }}
          style={{
            flex: 1,
            padding: '7px 12px',
            fontSize: 12,
            borderRadius: 6,
            background: 'var(--bg-canvas, #14171f)',
            border: '1px solid var(--border-subtle, #2d3342)',
            color: 'var(--text-primary, #e2e8f0)',
          }}
        />
        <button
          className="rf-btn rf-btn-primary"
          onClick={() => handleSendMessage()}
          disabled={isLoading || !inputValue.trim()}
          style={{
            padding: '7px 14px',
            fontWeight: 600,
            fontSize: 11,
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            background: 'linear-gradient(135deg, #7c3aed, #2563eb)',
            border: 'none',
          }}
        >
          <Send size={13} />
          <span>Send</span>
        </button>
      </div>
    </div>
  );
};
