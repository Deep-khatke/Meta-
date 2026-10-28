import React, { useState, useRef, useEffect } from 'react';
import {
  Send, Sparkles, CheckCircle2, AlertTriangle, Play, BrainCircuit,
  Maximize2, Minimize2, Trash2, Bot, User, CheckSquare, ShieldCheck,
  RefreshCw, ChevronRight, ChevronDown, ArrowRight, Cpu, Wrench, Microscope,
  Check, X, ShieldAlert, Zap, Radio, Sliders, Activity, Info
} from 'lucide-react';
import {
  AIChatRequest,
  AIChatResponse,
  AgentThoughtStep,
  AIApprovalRequest
} from '../types';
import {
  sendAIChatMessage,
  approveAIAction,
  cancelAIMission,
  applyAIConfig,
  runLiveExperiment,
  startRLTraining,
  fetchAIStatus,
  fetchAIChatHistory,
  clearAIChatHistory
} from '../services/api';

export interface PersistentChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: number;
  response?: AIChatResponse;
  isExecuting?: boolean;
  approvalStatus?: 'pending' | 'approved' | 'rejected';
  lastPrompt?: string;
}

interface AIAssistantChatProps {
  onShowToast: (msg: string) => void;
  selectedScenarioId?: string;
  targetPosition?: [number, number, number] | null;
  onProjectLoaded?: () => void;
  onSelectTarget?: (pos: [number, number, number] | null) => void;
  onSwitchView?: (view: 'metrics' | 'compare' | 'rl_training' | 'benchmark' | 'ai_assistant' | 'hypotheses') => void;
}

const STORAGE_KEY = 'roboforge_ai_chat_workspace_history_v3b';
const MODE_STORAGE_KEY = 'roboforge_ai_chat_mode_v3b';

const INITIAL_WELCOME_MSG: PersistentChatMessage = {
  id: 'welcome',
  sender: 'assistant',
  text: "👋 Welcome to the **RoboForge Autonomous AI Robotics Workspace**!\n\nI am connected directly to the live physics engine, research orchestrator, and scientific intelligence memory. I can inspect robot sensors and kinematics, diagnose collisions, propose supervised or autonomous research missions, execute multi-seed simulation trials, and analyze empirical deltas.\n\nChoose your execution mode above and ask any question or command to get started.",
  timestamp: Date.now(),
  response: {
    reply: "Welcome to RoboForge Autonomous AI Workspace.",
    provider: "DeterministicRoboticsHeuristicEngine-v1",
    is_model_inference: false,
    is_fallback: true,
    mode: "supervised",
    thought_steps: [
      {
        step: "initialize",
        title: "Autonomous Workspace Initialized",
        detail: "Connected to MuJoCo physics loop, MissionCompiler, and ResearchMemory backend.",
        status: "success",
      },
    ],
    observed_facts: {
      status: "Ready for telemetry inspection and autonomous experimentation.",
    },
    tools_invoked: ["inspect_simulation_state"],
    awaiting_approval: false,
  },
};

export const AIAssistantChat: React.FC<AIAssistantChatProps> = ({
  onShowToast,
  selectedScenarioId,
  targetPosition,
  onProjectLoaded,
  onSelectTarget,
  onSwitchView,
}) => {
  // Load saved mode or default to 'supervised'
  const [mode, setMode] = useState<'supervised' | 'autonomous'>(() => {
    const saved = localStorage.getItem(MODE_STORAGE_KEY);
    return saved === 'autonomous' ? 'autonomous' : 'supervised';
  });

  // Load saved messages or initial welcome message
  const [messages, setMessages] = useState<PersistentChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {
      // Ignore parse errors
    }
    return [INITIAL_WELCOME_MSG];
  });

  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedThoughts, setExpandedThoughts] = useState<Record<string, boolean>>({});
  const [streamingMsgId, setStreamingMsgId] = useState<string | null>(null);
  const [displayedWordCount, setDisplayedWordCount] = useState<number>(0);
  const [aiStatus, setAiStatus] = useState<{
    active_provider: string;
    has_credentials: boolean;
    model: string;
    tools_available: string[];
    offline_mode: boolean;
    provider_type?: string;
    endpoint?: string;
    available_models?: string[];
  } | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Persist messages whenever updated
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // LocalStorage full or private browsing
    }
    scrollToBottom();
  }, [messages, isLoading]);

  // Persist mode
  useEffect(() => {
    localStorage.setItem(MODE_STORAGE_KEY, mode);
  }, [mode]);

  // Fetch provider status and hydrate multi-turn conversation memory on mount
  useEffect(() => {
    fetchAIStatus().then(setAiStatus).catch(() => {});
    fetchAIChatHistory('default_session', 30).then((res) => {
      if (res?.history && res.history.length > 0) {
        const hydrated: PersistentChatMessage[] = res.history.map((h: any) => ({
          id: h.id || `hist_${h.timestamp}`,
          sender: h.role === 'user' ? 'user' : 'assistant',
          text: h.content,
          timestamp: h.timestamp ? h.timestamp * 1000 : Date.now(),
          response: h.response,
        }));
        setMessages(hydrated);
      }
    }).catch(() => {});
  }, []);

  // Word-by-word gliding typewriter streaming effect for new assistant responses
  useEffect(() => {
    if (!streamingMsgId) return;
    const targetMsg = messages.find((m) => m.id === streamingMsgId);
    if (!targetMsg) {
      setStreamingMsgId(null);
      return;
    }
    const words = targetMsg.text.split(/(\s+)/);
    if (displayedWordCount >= words.length) {
      setStreamingMsgId(null);
      return;
    }
    const timer = setTimeout(() => {
      setDisplayedWordCount((prev) => prev + 2); // Glide 2 tokens per tick for sleek pace
      scrollToBottom();
    }, 18);
    return () => clearTimeout(timer);
  }, [streamingMsgId, displayedWordCount, messages]);

  const toggleThoughts = (msgId: string) => {
    setExpandedThoughts((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  const handleClearHistory = async () => {
    setMessages([INITIAL_WELCOME_MSG]);
    try {
      localStorage.removeItem(STORAGE_KEY);
      await clearAIChatHistory('default_session');
    } catch {}
    onShowToast('Chat memory reset and cleared.');
  };

  const handleSendMessage = async (textToSend?: string) => {
    const prompt = (textToSend || inputValue).trim();
    if (!prompt || isLoading) return;

    const userMsg: PersistentChatMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      text: prompt,
      timestamp: Date.now(),
      lastPrompt: prompt,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Build past turns so model and backend have full conversational context
      const chatHistory = messages.slice(-10).map((m) => ({
        role: m.sender === 'user' ? 'user' : 'assistant',
        content: m.text,
      }));

      const chatReq: AIChatRequest = {
        prompt,
        mode,
        selected_scenario_id: selectedScenarioId,
        max_iterations: 5,
        max_experiments: 8,
        session_id: 'default_session',
        chat_history: chatHistory,
      };

      const res: AIChatResponse = await sendAIChatMessage(chatReq);

      const aiMsg: PersistentChatMessage = {
        id: `ai_${Date.now()}`,
        sender: 'assistant',
        text: res.reply,
        timestamp: Date.now(),
        response: res,
        approvalStatus: res.awaiting_approval ? 'pending' : undefined,
        lastPrompt: prompt,
      };

      setMessages((prev) => [...prev, aiMsg]);
      setStreamingMsgId(aiMsg.id);
      setDisplayedWordCount(4);

      // Automatically expand thought steps for rich transparency
      if (res.thought_steps && res.thought_steps.length > 0) {
        setExpandedThoughts((prev) => ({ ...prev, [aiMsg.id]: true }));
      }

      // If actions modified the project or simulation scene, immediately trigger live scene reload!
      if (res.actions_executed && res.actions_executed.length > 0) {
        if (onProjectLoaded) {
          onProjectLoaded();
        }
        const actionLabels = res.actions_executed.map((a) => a.action).join(', ');
        onShowToast(`RoboForge Autonomy: Executed ${actionLabels}`);
      } else if (res.awaiting_approval) {
        onShowToast('Supervised Action: Plan formulated. Awaiting operator approval.');
      } else if (res.mission) {
        onShowToast(`Research Mission: ${res.mission.status.toUpperCase()} (${res.mission.mission_id})`);
      } else if (res.suggested_configuration?.environment?.length) {
        onShowToast(`AI: Synthesized ${res.suggested_configuration.environment.length} obstacles safely.`);
      } else {
        onShowToast(`AI: Replied using ${res.provider}`);
      }
    } catch (e: any) {
      const errorMsg: PersistentChatMessage = {
        id: `ai_err_${Date.now()}`,
        sender: 'assistant',
        text: `⚠️ **Agent Execution Error:** ${e.message || 'Failed to complete agent reasoning loop'}.\n\nYou can click **Retry** below to re-submit this command or check system logs.`,
        timestamp: Date.now(),
        lastPrompt: prompt,
      };
      setMessages((prev) => [...prev, errorMsg]);
      onShowToast(`AI Chat error: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleApproveAction = async (msgId: string, missionId: string, approved: boolean) => {
    try {
      setIsLoading(true);
      onShowToast(approved ? 'Approving and executing simulation trials...' : 'Rejecting experiment plan...');

      const req: AIApprovalRequest = {
        mission_id: missionId,
        approved,
        reason: approved ? 'Approved by human supervisor in chat' : 'Rejected by supervisor in chat',
      };

      const res = await approveAIAction(req);

      // Update the originating message's approval status
      setMessages((prev) =>
        prev.map((m) =>
          m.id === msgId
            ? { ...m, approvalStatus: approved ? 'approved' : 'rejected' }
            : m
        )
      );

      // Add assistant response with trial execution results
      const resultMsg: PersistentChatMessage = {
        id: `ai_approve_${Date.now()}`,
        sender: 'assistant',
        text: res.reply,
        timestamp: Date.now(),
        response: res,
      };
      setMessages((prev) => [...prev, resultMsg]);
      setExpandedThoughts((prev) => ({ ...prev, [resultMsg.id]: true }));

      onShowToast(approved ? 'Trials executed and evidence registered.' : 'Mission cancelled.');
    } catch (e: any) {
      onShowToast(`Approval action failed: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelMission = async (missionId: string) => {
    try {
      onShowToast('Cancelling active research mission...');
      await cancelAIMission(missionId, 'Cancelled by user via chat');
      onShowToast(`Mission ${missionId} cancelled.`);
      // Add a status notice to chat
      setMessages((prev) => [
        ...prev,
        {
          id: `ai_cancel_${Date.now()}`,
          sender: 'assistant',
          text: `🛑 **Mission Cancelled:** Active research mission \`${missionId}\` was terminated on request.`,
          timestamp: Date.now(),
        },
      ]);
    } catch (e: any) {
      onShowToast(`Failed to cancel mission: ${e.message}`);
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

  const presetChips = [
    { label: '🤖 Robot Capabilities', prompt: 'Inspect this robot and explain its capabilities, kinematics, and sensors.' },
    { label: '🔍 Diagnose Failure', prompt: 'Diagnose why the robot collided or failed in recent runs and explain telemetry.' },
    { label: '🛡️ Supervised Mission', prompt: 'Propose a supervised research mission to optimize navigation safety margin.' },
    { label: '⚡ Autonomous Mission', prompt: 'Run an autonomous research mission to optimize navigation speed and minimize transit time.' },
    { label: '📊 Candidate Deltas', prompt: 'Explain why the latest candidate improved or degraded compared to baseline.' },
    { label: '🧱 Slalom Challenge', prompt: 'Create a tight 3-obstacle slalom with a 2.5m target beacon.' },
  ];

  const formatInlineText = (text: string): React.ReactNode => {
    const codeParts = text.split(/(`[^`]+`)/g);
    return codeParts.map((part, pIdx) => {
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={`code-${pIdx}`}
            style={{
              background: 'rgba(0, 0, 0, 0.4)',
              padding: '2px 6px',
              borderRadius: 4,
              fontSize: 11,
              color: '#38bdf8',
              fontFamily: 'monospace',
            }}
          >
            {part.slice(1, -1)}
          </code>
        );
      }
      const boldParts = part.split(/(\*\*[^*]+\*\*)/g);
      return boldParts.map((bPart, bIdx) => {
        if (bPart.startsWith('**') && bPart.endsWith('**')) {
          return (
            <strong key={`b-${pIdx}-${bIdx}`} style={{ color: '#f8fafc', fontWeight: 700 }}>
              {bPart.slice(2, -2)}
            </strong>
          );
        }
        const italicParts = bPart.split(/(\*[^*]+\*)/g);
        return italicParts.map((iPart, iIdx) => {
          if (iPart.startsWith('*') && iPart.endsWith('*')) {
            return (
              <em key={`i-${pIdx}-${bIdx}-${iIdx}`} style={{ color: '#94a3b8', fontStyle: 'italic' }}>
                {iPart.slice(1, -1)}
              </em>
            );
          }
          return iPart;
        });
      });
    });
  };

  const renderFormattedText = (text: string, isStreaming: boolean = false) => {
    const lines = text.split('\n');
    const renderedLines: React.ReactNode[] = [];

    lines.forEach((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) {
        renderedLines.push(<div key={`blank-${idx}`} style={{ height: 6 }} />);
        return;
      }

      if (trimmed.startsWith('### ')) {
        renderedLines.push(
          <div key={`h3-${idx}`} style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', marginTop: 8, marginBottom: 4 }}>
            {formatInlineText(trimmed.slice(4))}
          </div>
        );
        return;
      }
      if (trimmed.startsWith('## ')) {
        renderedLines.push(
          <div key={`h2-${idx}`} style={{ fontSize: 14, fontWeight: 700, color: '#f8fafc', marginTop: 10, marginBottom: 4 }}>
            {formatInlineText(trimmed.slice(3))}
          </div>
        );
        return;
      }
      if (trimmed.startsWith('# ')) {
        renderedLines.push(
          <div key={`h1-${idx}`} style={{ fontSize: 15, fontWeight: 800, color: '#f8fafc', marginTop: 10, marginBottom: 6 }}>
            {formatInlineText(trimmed.slice(2))}
          </div>
        );
        return;
      }

      // Bullets
      if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
        const content = trimmed.replace(/^[-*•]\s+/, '');
        renderedLines.push(
          <div key={`li-${idx}`} style={{ display: 'flex', gap: 6, alignItems: 'flex-start', margin: '2px 0 2px 4px' }}>
            <span style={{ color: '#38bdf8', lineHeight: 1.5 }}>•</span>
            <span style={{ flex: 1 }}>{formatInlineText(content)}</span>
          </div>
        );
        return;
      }

      // Standard line
      renderedLines.push(
        <div key={`p-${idx}`} style={{ margin: '2px 0', lineHeight: 1.55 }}>
          {formatInlineText(line)}
        </div>
      );
    });

    return (
      <div>
        {renderedLines}
        {isStreaming && (
          <span
            style={{
              display: 'inline-block',
              width: 8,
              height: 14,
              marginLeft: 3,
              verticalAlign: 'middle',
              background: '#38bdf8',
              borderRadius: 2,
              animation: 'pulse 1s infinite',
              boxShadow: '0 0 8px #38bdf8',
            }}
          />
        )}
      </div>
    );
  };

  const renderMessageCard = (msg: PersistentChatMessage) => {
    const isUser = msg.sender === 'user';
    const resp = msg.response;
    const config = resp?.suggested_configuration;
    const mission = resp?.mission;
    const approvalPlan = resp?.approval_plan;

    return (
      <div
        key={msg.id}
        style={{
          display: 'flex',
          gap: 10,
          alignItems: 'flex-start',
          flexDirection: isUser ? 'row-reverse' : 'row',
          marginBottom: 14,
        }}
      >
        {/* Avatar */}
        <div
          style={{
            width: 30,
            height: 30,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            background: isUser
              ? 'var(--accent-blue, #2563eb)'
              : 'linear-gradient(135deg, #7c3aed, #06b6d4)',
            color: '#fff',
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
          }}
        >
          {isUser ? <User size={16} /> : <Bot size={16} />}
        </div>

        {/* Message Bubble Container */}
        <div
          style={{
            maxWidth: isUser ? '75%' : '88%',
            background: isUser ? 'var(--accent-blue, #2563eb)' : 'var(--bg-card, #1e222b)',
            color: isUser ? '#fff' : 'var(--text-primary, #e2e8f0)',
            padding: '12px 14px',
            borderRadius: isUser ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
            border: isUser ? 'none' : '1px solid var(--border-subtle, #2d3342)',
            fontSize: 12,
            lineHeight: 1.5,
            boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
          }}
        >
          {/* Minimal Action Badges if an action was executed */}
          {resp?.actions_executed && resp.actions_executed.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 2 }}>
              {resp.actions_executed.map((act, aIdx) => (
                <div
                  key={aIdx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5,
                    padding: '2px 8px',
                    borderRadius: 4,
                    background: 'rgba(56, 189, 248, 0.15)',
                    border: '1px solid rgba(56, 189, 248, 0.35)',
                    fontSize: 10,
                    fontWeight: 600,
                    color: '#38bdf8',
                  }}
                >
                  <Zap size={10} color="#38bdf8" />
                  <span>
                    {act.action === 'add_drone' && `🚁 Deployed ${act.name || 'Aerial Drone'}`}
                    {act.action === 'add_robot' && `🤖 Deployed ${act.name || 'Differential Robot'}`}
                    {act.action === 'generate_obstacle_course' && `⚡ Synthesized ${act.obstacle_count || 6}-Obstacle Course`}
                    {act.action === 'train_robot' && `🚀 Launched Training Mission`}
                    {act.action === 'clear_scene' && `🧹 Cleared Scene`}
                    {!['add_drone', 'add_robot', 'generate_obstacle_course', 'train_robot', 'clear_scene'].includes(act.action) && act.action}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Main Conversational Content (Supports word-by-word typewriter streaming) */}
          <div style={{ wordBreak: 'break-word' }}>
            {(() => {
              if (streamingMsgId === msg.id) {
                const words = msg.text.split(/(\s+)/);
                const visible = words.slice(0, displayedWordCount).join('');
                return renderFormattedText(visible, true);
              }
              return renderFormattedText(msg.text, false);
            })()}
          </div>

          {/* Supervised Action Approval Gate (Only rendered when human confirmation is required) */}
          {resp?.awaiting_approval && approvalPlan && (
            <div
              style={{
                marginTop: 6,
                background: 'rgba(234, 179, 8, 0.08)',
                padding: '10px 12px',
                borderRadius: 6,
                border: '1px solid rgba(234, 179, 8, 0.3)',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 700, color: '#facc15' }}>
                  <ShieldAlert size={13} />
                  <span>Supervised Approval Required</span>
                </span>
                <span style={{ fontSize: 9, color: '#facc15' }}>{approvalPlan.num_trials || 3} trials</span>
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary, #94a3b8)' }}>
                {approvalPlan.title || 'Controlled Experiment Plan'}
              </div>
              {msg.approvalStatus === 'pending' || !msg.approvalStatus ? (
                <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                  <button
                    className="rf-btn rf-btn-primary"
                    onClick={() => handleApproveAction(msg.id, mission?.mission_id || approvalPlan.mission_id, true)}
                    disabled={isLoading}
                    style={{
                      fontSize: 10,
                      padding: '4px 10px',
                      fontWeight: 600,
                      background: 'linear-gradient(135deg, #10b981, #059669)',
                      border: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                    }}
                  >
                    <Check size={12} />
                    <span>Approve & Execute</span>
                  </button>
                  <button
                    className="rf-btn"
                    onClick={() => handleApproveAction(msg.id, mission?.mission_id || approvalPlan.mission_id, false)}
                    disabled={isLoading}
                    style={{
                      fontSize: 10,
                      padding: '4px 8px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      background: 'rgba(239, 68, 68, 0.15)',
                      color: '#f87171',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                    }}
                  >
                    <X size={12} />
                    <span>Reject</span>
                  </button>
                </div>
              ) : (
                <div style={{ fontSize: 10, fontWeight: 600, color: msg.approvalStatus === 'approved' ? '#4ade80' : '#f87171' }}>
                  {msg.approvalStatus === 'approved' ? '✓ Approved by Operator' : '✕ Rejected by Operator'}
                </div>
              )}
            </div>
          )}

          {/* Clean Action Shortcuts (Sleek 1-line controls, no bulky cards) */}
          {(config || (mission && onSwitchView)) && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
              {config && (
                <>
                  <button
                    className="rf-btn rf-btn-primary"
                    onClick={() => handleApplyTo3D(config, true)}
                    style={{
                      fontSize: 10,
                      padding: '4px 9px',
                      fontWeight: 600,
                      background: 'linear-gradient(135deg, #0284c7, #2563eb)',
                      border: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                    }}
                  >
                    <Play size={11} fill="#fff" />
                    <span>Execute in 3D Viewport</span>
                  </button>
                  <button
                    className="rf-btn"
                    onClick={() => handleApplyTo3D(config, false)}
                    style={{ fontSize: 10, padding: '4px 8px', display: 'flex', alignItems: 'center', gap: 4 }}
                  >
                    <CheckSquare size={11} color="var(--accent-green, #10b981)" />
                    <span>Apply to Scene</span>
                  </button>
                </>
              )}
              {mission && onSwitchView && (
                <button
                  className="rf-btn"
                  onClick={() => onSwitchView('hypotheses')}
                  style={{
                    fontSize: 10,
                    padding: '4px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    background: 'rgba(168, 85, 247, 0.15)',
                    color: '#c084fc',
                    border: '1px solid rgba(168, 85, 247, 0.3)',
                  }}
                >
                  <ArrowRight size={11} />
                  <span>Inspect in Research Studio</span>
                </button>
              )}
              {mission && mission.status !== 'completed' && mission.status !== 'cancelled' && (
                <button
                  className="rf-btn"
                  onClick={() => handleCancelMission(mission.mission_id)}
                  style={{
                    fontSize: 10,
                    padding: '4px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    background: 'rgba(239, 68, 68, 0.12)',
                    color: '#f87171',
                    border: '1px solid rgba(239, 68, 68, 0.25)',
                  }}
                >
                  <X size={11} />
                  <span>Cancel Mission</span>
                </button>
              )}
            </div>
          )}

          {/* Retry Button if this was an error message */}
          {msg.id.includes('err') && msg.lastPrompt && (
            <div style={{ marginTop: 4 }}>
              <button
                className="rf-btn"
                onClick={() => handleSendMessage(msg.lastPrompt)}
                style={{
                  fontSize: 10,
                  padding: '4px 10px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  background: 'rgba(239, 68, 68, 0.15)',
                  color: '#f87171',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                }}
              >
                <RefreshCw size={11} />
                <span>Retry Request</span>
              </button>
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
        height: isExpanded ? '88vh' : '100%',
        background: 'var(--bg-panel, #181b22)',
        borderRadius: 8,
        border: '1px solid var(--border-subtle, #2d3342)',
        position: isExpanded ? 'fixed' : 'relative',
        top: isExpanded ? '6vh' : 'auto',
        left: isExpanded ? '5vw' : 'auto',
        width: isExpanded ? '90vw' : '100%',
        zIndex: isExpanded ? 9999 : 1,
        boxShadow: isExpanded ? '0 16px 64px rgba(0,0,0,0.85)' : 'none',
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
          flexWrap: 'wrap',
          gap: 6,
        }}
      >
        {/* Title & Provider Status */}
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
              <span>RoboForge Autonomous AI Studio</span>
              <span
                style={{
                  fontSize: 9,
                  padding: '1px 6px',
                  borderRadius: 3,
                  background: aiStatus?.provider_type === 'ollama' ? 'rgba(59, 130, 246, 0.2)' : aiStatus?.has_credentials ? 'rgba(34, 197, 94, 0.2)' : 'rgba(234, 179, 8, 0.2)',
                  color: aiStatus?.provider_type === 'ollama' ? '#60a5fa' : aiStatus?.has_credentials ? '#4ade80' : '#facc15',
                  fontWeight: 600,
                }}
                title={aiStatus?.provider_type === 'ollama' ? `Local Ollama Inference (${aiStatus.endpoint || 'http://127.0.0.1:11434'})` : aiStatus?.has_credentials ? `Connected to ${aiStatus.model}` : 'Deterministic physics heuristics'}
              >
                {aiStatus?.provider_type === 'ollama' ? `Local LLM: ${aiStatus.model}` : aiStatus?.has_credentials ? `Model: ${aiStatus.model}` : `Engine: ${aiStatus?.active_provider || 'Deterministic Physics'}`}
              </span>
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>
              Telemetry grounded &bull; Physical constraints &bull; Zero hallucinated telemetry
            </div>
          </div>
        </div>

        {/* Supervised vs Autonomous Mode Selector & Controls */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {/* Mode Selector Toggle */}
          <div
            style={{
              display: 'flex',
              background: 'var(--bg-canvas, #14171f)',
              borderRadius: 6,
              padding: 2,
              border: '1px solid var(--border-subtle, #2d3342)',
            }}
          >
            <button
              type="button"
              onClick={() => {
                setMode('supervised');
                onShowToast('Mode changed: Supervised (Human approval required for trials)');
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                padding: '3px 8px',
                borderRadius: 4,
                border: 'none',
                background: mode === 'supervised' ? 'var(--accent-blue, #2563eb)' : 'transparent',
                color: mode === 'supervised' ? '#fff' : 'var(--text-muted)',
                fontSize: 10,
                fontWeight: 600,
                cursor: 'pointer',
              }}
              title="Supervised Mode: Formulates plans and requires explicit human approval before running simulation trials"
            >
              <ShieldCheck size={11} />
              <span>Supervised</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setMode('autonomous');
                onShowToast('Mode changed: Autonomous (Bounded auto-run within experiment budget)');
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                padding: '3px 8px',
                borderRadius: 4,
                border: 'none',
                background: mode === 'autonomous' ? 'linear-gradient(135deg, #7c3aed, #4f46e5)' : 'transparent',
                color: mode === 'autonomous' ? '#fff' : 'var(--text-muted)',
                fontSize: 10,
                fontWeight: 600,
                cursor: 'pointer',
              }}
              title="Autonomous Mode: Agent plans, validates, and runs bounded multi-seed experiments automatically"
            >
              <Zap size={11} />
              <span>Autonomous</span>
            </button>
          </div>

          <button
            className="rf-btn"
            onClick={handleClearHistory}
            style={{ fontSize: 10, padding: '3px 7px' }}
            title="Clear Chat History"
          >
            <Trash2 size={11} />
            <span>Clear</span>
          </button>
          <button
            className="rf-btn"
            onClick={() => setIsExpanded(!isExpanded)}
            style={{ fontSize: 10, padding: '3px 7px' }}
            title={isExpanded ? 'Collapse Studio' : 'Expand to Full Workspace'}
          >
            {isExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
            <span>{isExpanded ? 'Collapse' : 'Expand'}</span>
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
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', color: 'var(--text-muted)', fontSize: 11, padding: '8px 0' }}>
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
            <span>
              {mode === 'autonomous'
                ? 'Agent reasoning: inspecting context, planning hypothesis, and executing simulation trials...'
                : 'Agent reasoning: inspecting telemetry, formulating plan, and verifying physical bounds...'}
            </span>
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
            onClick={() => handleSendMessage(chip.prompt)}
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
            {chip.label}
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
        <span
          style={{
            fontSize: 9,
            padding: '3px 6px',
            borderRadius: 4,
            background: mode === 'supervised' ? 'rgba(37, 99, 235, 0.2)' : 'rgba(124, 58, 237, 0.2)',
            color: mode === 'supervised' ? '#60a5fa' : '#c084fc',
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          {mode === 'supervised' ? 'Supervised' : 'Autonomous'}
        </span>
        <input
          type="text"
          className="rf-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={
            mode === 'supervised'
              ? "Supervised: 'Inspect robot capabilities', 'Propose research mission', 'Diagnose collision'..."
              : "Autonomous: 'Run autonomous research mission to minimize transit time', 'Synthesize slalom'..."
          }
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
