import React, { useState, useRef, useEffect, useLayoutEffect } from 'react';
import {
  Send, Sparkles, CheckCircle2, AlertTriangle, Play, BrainCircuit,
  Maximize2, Minimize2, Trash2, Bot, User, CheckSquare, ShieldCheck,
  RefreshCw, ChevronRight, ChevronDown, ArrowRight, Cpu, Wrench, Microscope,
  Check, X, ShieldAlert, Zap, Radio, Sliders, Activity, Info, Copy,
  Box, Navigation, Gauge, Compass, Crosshair, Layers, RotateCcw,
  Terminal, Sparkle
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
import { AgentAvatar } from './AgentAvatar';

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
  onMissionWaypoints?: (waypoints: Array<[number, number]>, loop: boolean) => void;
  onSwitchView?: (view: 'metrics' | 'compare' | 'rl_training' | 'benchmark' | 'ai_assistant' | 'hypotheses') => void;
}

const STORAGE_KEY = 'roboforge_ai_chat_workspace_history_v5';
const MODE_STORAGE_KEY = 'roboforge_ai_chat_mode_v5';

// Code Block with Copy Button
const CodeBlock: React.FC<{ language: string; code: string }> = ({ language, code }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      style={{
        margin: '8px 0',
        borderRadius: 8,
        overflow: 'hidden',
        background: '#0d1117',
        border: '1px solid #2f3436',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '5px 12px',
          background: '#161b22',
          borderBottom: '1px solid #21262d',
          fontSize: 11,
          fontFamily: 'var(--font-mono, monospace)',
          color: '#8b949e',
        }}
      >
        <span>{language || 'code'}</span>
        <button
          type="button"
          onClick={handleCopy}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            background: 'transparent',
            border: 'none',
            color: copied ? '#3fb950' : '#8b949e',
            cursor: 'pointer',
            fontSize: 11,
            padding: '2px 6px',
            borderRadius: 4,
          }}
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre
        style={{
          margin: 0,
          padding: '10px 14px',
          overflowX: 'auto',
          fontSize: 12,
          lineHeight: 1.5,
          fontFamily: 'var(--font-mono, monospace)',
          color: '#f4f3ef',
        }}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
};

// Markdown Table Renderer
const MarkdownTable: React.FC<{ rows: string[][] }> = ({ rows }) => {
  if (rows.length === 0) return null;
  const header = rows[0];
  const body = rows.slice(1);

  return (
    <div style={{ overflowX: 'auto', margin: '8px 0' }}>
      <table
        style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: 12,
          textAlign: 'left',
          border: '1px solid #2f3436',
          borderRadius: 6,
          overflow: 'hidden',
        }}
      >
        <thead>
          <tr style={{ background: '#161b22', borderBottom: '2px solid #2f3436' }}>
            {header.map((col, idx) => (
              <th key={idx} style={{ padding: '7px 12px', color: '#f4f3ef', fontWeight: 600 }}>
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, rIdx) => (
            <tr
              key={rIdx}
              style={{
                borderBottom: '1px solid #21262d',
                background: rIdx % 2 === 0 ? 'transparent' : 'rgba(255, 255, 255, 0.02)',
              }}
            >
              {row.map((cell, cIdx) => (
                <td key={cIdx} style={{ padding: '6px 12px', color: '#d6d5cf' }}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export const AIAssistantChat: React.FC<AIAssistantChatProps> = ({
  onShowToast,
  selectedScenarioId,
  targetPosition,
  onProjectLoaded,
  onSelectTarget,
  onMissionWaypoints,
  onSwitchView,
}) => {
  // Mode: autonomous (instant executive actions) vs supervised (asks approval)
  const [mode, setMode] = useState<'supervised' | 'autonomous'>(() => {
    const saved = localStorage.getItem(MODE_STORAGE_KEY);
    return saved === 'supervised' ? 'supervised' : 'autonomous';
  });

  // Chat message history
  const [messages, setMessages] = useState<PersistentChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {}
    return [];
  });

  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedThoughts, setExpandedThoughts] = useState<Record<string, boolean>>({});
  const [streamingMsgId, setStreamingMsgId] = useState<string | null>(null);
  const [displayedWordCount, setDisplayedWordCount] = useState<number>(0);
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);

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
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Auto-resize textarea as content changes
  useLayoutEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(Math.max(scrollHeight, 40), 160)}px`;
    }
  }, [inputValue]);

  // Persist messages whenever updated
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {}
    scrollToBottom();
  }, [messages, isLoading]);

  // Persist mode
  useEffect(() => {
    localStorage.setItem(MODE_STORAGE_KEY, mode);
  }, [mode]);

  // Fetch provider status and chat memory on mount
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

  // Smooth token typewriter glide
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
      setDisplayedWordCount((prev) => prev + 3);
      scrollToBottom();
    }, 16);
    return () => clearTimeout(timer);
  }, [streamingMsgId, displayedWordCount, messages]);

  const toggleThoughts = (msgId: string) => {
    setExpandedThoughts((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  const handleCopyMessage = (msgId: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedMsgId(msgId);
    setTimeout(() => setCopiedMsgId(null), 2000);
  };

  const handleClearHistory = async () => {
    setMessages([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
      await clearAIChatHistory('default_session');
    } catch {}
    onShowToast('Conversation cleared.');
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
    if (textareaRef.current) {
      textareaRef.current.style.height = '40px';
    }
    setIsLoading(true);

    try {
      // Build conversation context
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
      setDisplayedWordCount(6);

      // Expand thinking accordion by default if steps present
      if (res.thought_steps && res.thought_steps.length > 0) {
        setExpandedThoughts((prev) => ({ ...prev, [aiMsg.id]: true }));
      }

      // If actions modified the project or simulation scene, immediately refresh 3D viewport
      if (res.actions_executed && res.actions_executed.length > 0) {
        if (onProjectLoaded) {
          onProjectLoaded();
        }
        // One-shot mission build: push the generated patrol route into the Mission Planner.
        const mission = res.actions_executed.find((a: any) => a.mission_waypoints && a.mission_waypoints.length);
        if (mission && onMissionWaypoints) {
          onMissionWaypoints(
            (mission as any).mission_waypoints.map((p: any) => [p[0], p[1]]),
            (mission as any).mission_loop !== false,
          );
        }
        const actionLabels = res.actions_executed.map((a) => a.action.replace('tool_', '')).join(', ');
        onShowToast(`Executed: ${actionLabels}`);
      } else if (res.awaiting_approval) {
        onShowToast('Action formulated. Waiting for approval.');
      }
    } catch (e: any) {
      const errorMsg: PersistentChatMessage = {
        id: `ai_err_${Date.now()}`,
        sender: 'assistant',
        text: `**Error:** ${e.message || 'Failed to complete agent execution'}.\n\nPlease try again.`,
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
      onShowToast(approved ? 'Executing plan in physics simulation...' : 'Plan cancelled.');

      const req: AIApprovalRequest = {
        mission_id: missionId,
        approved,
        reason: approved ? 'Approved by user' : 'Cancelled by user',
      };

      const res = await approveAIAction(req);

      setMessages((prev) =>
        prev.map((m) =>
          m.id === msgId
            ? { ...m, approvalStatus: approved ? 'approved' : 'rejected' }
            : m
        )
      );

      const resultMsg: PersistentChatMessage = {
        id: `ai_approve_${Date.now()}`,
        sender: 'assistant',
        text: res.reply,
        timestamp: Date.now(),
        response: res,
      };
      setMessages((prev) => [...prev, resultMsg]);
      setExpandedThoughts((prev) => ({ ...prev, [resultMsg.id]: true }));

      if (res.actions_executed && res.actions_executed.length > 0 && onProjectLoaded) {
        onProjectLoaded();
      }
    } catch (e: any) {
      onShowToast(`Action failed: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Quick Action Starter Prompts
  const quickActions = [
    {
      icon: <BrainCircuit size={16} color="#a9aca7" />,
      title: 'Train RL Policy (PPO)',
      desc: 'Train reinforcement learning in scene',
      prompt: 'Train a PPO reinforcement learning policy in the scene until it achieves the goal',
    },
    {
      icon: <RotateCcw size={16} color="#e8622c" />,
      title: 'Clear Scene (Keep Robot)',
      desc: 'Clear obstacles and keep robot safe',
      prompt: 'Clear all obstacles from the scene but keep the active robot untouched',
    },
    {
      icon: <Sliders size={16} color="#5cc596" />,
      title: 'Actuate Motors to Zero',
      desc: 'Reset all joints to home position',
      prompt: 'Reset all robot motor actuators to zero home position',
    },
    {
      icon: <Box size={16} color="#dca23a" />,
      title: 'Spawn Obstacle Course',
      desc: 'Build randomized navigation course',
      prompt: 'Spawn a 4-obstacle course around the active robot with guaranteed clearance',
    },
  ];

  // Markdown Parser
  const parseMarkdown = (content: string) => {
    const blocks: React.ReactNode[] = [];
    const lines = content.split('\n');
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Multi-line code block ```lang ... ```
      if (line.trim().startsWith('```')) {
        const lang = line.trim().slice(3).trim();
        const codeLines: string[] = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }
        i++; // skip closing ```
        blocks.push(<CodeBlock key={`code-${i}`} language={lang} code={codeLines.join('\n')} />);
        continue;
      }

      // Markdown Table
      if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
        const tableLines: string[] = [];
        while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
          tableLines.push(lines[i].trim());
          i++;
        }
        if (tableLines.length >= 2) {
          const parsedRows = tableLines
            .filter((tLine) => !/^\|[\s\-:|]+\|$/.test(tLine)) // ignore divider |---|---|
            .map((tLine) =>
              tLine
                .slice(1, -1)
                .split('|')
                .map((c) => c.trim())
            );
          blocks.push(<MarkdownTable key={`table-${i}`} rows={parsedRows} />);
          continue;
        }
      }

      // Empty line
      if (!line.trim()) {
        blocks.push(<div key={`blank-${i}`} style={{ height: 6 }} />);
        i++;
        continue;
      }

      // Headings
      if (line.startsWith('# ')) {
        blocks.push(
          <h2 key={`h1-${i}`} style={{ fontSize: 16, fontWeight: 700, color: '#f4f3ef', margin: '10px 0 4px' }}>
            {formatInline(line.slice(2))}
          </h2>
        );
        i++;
        continue;
      }
      if (line.startsWith('## ')) {
        blocks.push(
          <h3 key={`h2-${i}`} style={{ fontSize: 14, fontWeight: 700, color: '#f4f3ef', margin: '8px 0 4px' }}>
            {formatInline(line.slice(3))}
          </h3>
        );
        i++;
        continue;
      }
      if (line.startsWith('### ')) {
        blocks.push(
          <h4 key={`h3-${i}`} style={{ fontSize: 13, fontWeight: 600, color: '#e8622c', margin: '6px 0 3px' }}>
            {formatInline(line.slice(4))}
          </h4>
        );
        i++;
        continue;
      }

      // Bullets
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ') || line.trim().startsWith('• ')) {
        const bulletContent = line.trim().replace(/^[-*•]\s+/, '');
        blocks.push(
          <div key={`li-${i}`} style={{ display: 'flex', gap: 8, margin: '3px 0 3px 6px', alignItems: 'flex-start' }}>
            <span style={{ color: '#e8622c', fontSize: 14, lineHeight: 1.4 }}>•</span>
            <div style={{ flex: 1, fontSize: 13, lineHeight: 1.6, color: '#d6d5cf' }}>{formatInline(bulletContent)}</div>
          </div>
        );
        i++;
        continue;
      }

      // Numbered List
      const numMatch = line.trim().match(/^(\d+)\.\s+(.*)/);
      if (numMatch) {
        blocks.push(
          <div key={`num-${i}`} style={{ display: 'flex', gap: 8, margin: '3px 0 3px 6px', alignItems: 'flex-start' }}>
            <span style={{ color: '#e8622c', fontSize: 12, fontWeight: 600, minWidth: 16 }}>{numMatch[1]}.</span>
            <div style={{ flex: 1, fontSize: 13, lineHeight: 1.6, color: '#d6d5cf' }}>{formatInline(numMatch[2])}</div>
          </div>
        );
        i++;
        continue;
      }

      // Normal paragraph line
      blocks.push(
        <p key={`p-${i}`} style={{ margin: '2px 0', fontSize: 13, lineHeight: 1.6, color: '#d1d5db' }}>
          {formatInline(line)}
        </p>
      );
      i++;
    }

    return blocks;
  };

  // Inline formatting: `code`, **bold**, *italic*
  const formatInline = (text: string): React.ReactNode => {
    const codeParts = text.split(/(`[^`]+`)/g);
    return codeParts.map((part, pIdx) => {
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code
            key={`code-${pIdx}`}
            style={{
              background: 'rgba(110, 118, 129, 0.25)',
              padding: '2px 6px',
              borderRadius: 4,
              fontSize: 12,
              color: '#79c0ff',
              fontFamily: 'var(--font-mono, monospace)',
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
            <strong key={`b-${pIdx}-${bIdx}`} style={{ color: '#f4f3ef', fontWeight: 600 }}>
              {bPart.slice(2, -2)}
            </strong>
          );
        }
        const italicParts = bPart.split(/(\*[^*]+\*)/g);
        return italicParts.map((iPart, iIdx) => {
          if (iPart.startsWith('*') && iPart.endsWith('*')) {
            return (
              <em key={`i-${pIdx}-${bIdx}-${iIdx}`} style={{ color: '#9ca3af', fontStyle: 'italic' }}>
                {iPart.slice(1, -1)}
              </em>
            );
          }
          return iPart;
        });
      });
    });
  };

  // Render an Executed Action Card
  const renderActionCard = (act: any, idx: number) => {
    const actionKey = act.action?.replace('tool_', '') || '';
    let title = 'Scene Updated';
    let icon = <Zap size={14} color="#e8622c" />;

    if (actionKey === 'spawn_obstacle') {
      title = `Spawned ${act.shape ? act.shape.toUpperCase() : 'Obstacle'} '${act.name || 'obstacle'}'`;
      icon = <Box size={14} color="#e8622c" />;
    } else if (actionKey === 'remove_obstacle') {
      title = `Removed Obstacle '${act.name || ''}'`;
      icon = <Trash2 size={14} color="#f07378" />;
    } else if (actionKey === 'clear_scene') {
      title = 'Cleared All Obstacles from Scene';
      icon = <RotateCcw size={14} color="#dca23a" />;
    } else if (actionKey === 'generate_obstacle_course') {
      title = `Generated ${act.obstacle_count || 6}-Obstacle Course`;
      icon = <Layers size={14} color="#a9aca7" />;
    } else if (actionKey === 'synthesize_environment') {
      title = `Synthesized ${act.env_type ? act.env_type.toUpperCase().replace('_', ' ') : 'Custom'} Environment (${act.obstacle_count || 'Multi'}-Entities)`;
      icon = <Layers size={14} color="#e8622c" />;
    } else if (actionKey === 'set_robot_speed') {
      title = `Set Max Velocity: ${act.linear_speed ?? 2.0} m/s`;
      icon = <Gauge size={14} color="#5cc596" />;
    } else if (actionKey === 'teleport_robot') {
      title = `Teleported Robot to [${(act.position || [0, 0, 0.05]).join(', ')}]`;
      icon = <Crosshair size={14} color="#a9aca7" />;
    } else if (actionKey === 'set_target_position') {
      title = `Moved Target to [${(act.position || []).join(', ')}]`;
      icon = <Compass size={14} color="#e8622c" />;
    } else if (actionKey === 'set_physics') {
      title = `Physics Configured (gravity: ${act.gravity_z ?? -9.81})`;
      icon = <Sliders size={14} color="#e8622c" />;
    } else if (actionKey === 'simulation_control') {
      title = `Simulation Command: ${act.command?.toUpperCase()}`;
      icon = <Play size={14} color="#5cc596" />;
    } else if (actionKey === 'deploy_swarm') {
      title = `Deployed ${act.count || 4}-Robot Swarm (${act.formation || 'V-Shape'})`;
      icon = <Layers size={14} color="#a9aca7" />;
    } else if (actionKey === 'disband_swarm') {
      title = 'Disbanded Robot Swarm';
      icon = <Trash2 size={14} color="#f07378" />;
    } else if (actionKey === 'plan_and_execute_trajectory') {
      title = `A* Path Executing to [${(act.target_pos || []).join(', ')}]`;
      icon = <Navigation size={14} color="#e8622c" />;
    } else if (actionKey === 'tune_apf_controller') {
      title = `Tuned APF (Safety Margin: ${act.safety_margin ?? 0.85}m)`;
      icon = <Sliders size={14} color="#e8622c" />;
    } else if (actionKey === 'start_autodrive') {
      title = 'Autodrive Engaged';
      icon = <Play size={14} color="#3fb27f" />;
    } else if (actionKey === 'stop_autodrive') {
      title = 'Autodrive Disengaged';
      icon = <RotateCcw size={14} color="#eab308" />;
    } else if (actionKey === 'start_rl_training') {
      title = `RL Training Started (${act.model_name || 'PPO'})`;
      icon = <BrainCircuit size={14} color="#d98a3d" />;
    } else if (actionKey === 'cancel_rl_training') {
      title = 'RL Training Cancelled';
      icon = <RotateCcw size={14} color="#f07378" />;
    } else if (actionKey === 'evaluate_rl_policy') {
      title = `Evaluated Policy (Score: ${typeof act.mean_reward === 'number' ? act.mean_reward.toFixed(1) : 'OK'})`;
      icon = <Activity size={14} color="#a9aca7" />;
    } else if (actionKey === 'deploy_rl_policy') {
      title = `Deployed RL Policy '${act.model_name || 'PPO'}'`;
      icon = <Cpu size={14} color="#5cc596" />;
    } else if (actionKey === 'run_scenario_benchmark' || actionKey === 'run_multitrial_batch') {
      title = `Benchmark Completed (${act.scenario_id || `${act.trials_completed || 5} trials`})`;
      icon = <Microscope size={14} color="#5cc596" />;
    } else if (actionKey === 'start_ros2_bag') {
      title = `ROS 2 Bag Recording Started: ${act.bag_path || 'active'}`;
      icon = <Radio size={14} color="#a9aca7" />;
    } else if (actionKey === 'stop_ros2_bag') {
      title = 'ROS 2 Bag Recording Stopped';
      icon = <Radio size={14} color="#f07378" />;
    } else if (actionKey === 'configure_lidar_sensor' || actionKey === 'capture_camera_snapshot') {
      title = `Sensor Action: ${actionKey.replace(/_/g, ' ')}`;
      icon = <Activity size={14} color="#dca23a" />;
    } else if (actionKey === 'save_scene_snapshot' || actionKey === 'restore_scene_snapshot') {
      title = `Snapshot: ${actionKey.replace(/_/g, ' ')}`;
      icon = <Layers size={14} color="#e8622c" />;
    } else if (actionKey === 'control_drone') {
      title = `Drone Flight: ${act.command?.toUpperCase()} (${act.altitude ?? act.target_altitude ?? 1.5}m)`;
      icon = <Navigation size={14} color="#e8622c" />;
    } else if (actionKey === 'train_drone_policy') {
      title = `Started Drone RL Flight Policy Training (${act.task || 'Takeoff/Landing'})`;
      icon = <BrainCircuit size={14} color="#e8622c" />;
    } else if (actionKey === 'deploy_drone_policy') {
      title = `Deployed 6-DOF Flight Policy to ${act.drone_name || 'Drone'}`;
      icon = <Cpu size={14} color="#3fb27f" />;
    } else if (actionKey === 'add_drone') {
      title = `Deployed 6-DOF Aerial Drone (${act.drone_name || 'Quadcopter'})`;
      icon = <Navigation size={14} color="#e8622c" />;
    } else if (actionKey === 'control_secondary_robot') {
      title = `Secondary Rover Drive (${act.robot || 'ForgeBot-DD2'})`;
      icon = <Sliders size={14} color="#e8622c" />;
    }

    return (
      <div
        key={`act-${idx}`}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          borderRadius: 6,
          background: 'rgba(232, 98, 44, 0.08)',
          border: '1px solid rgba(232, 98, 44, 0.25)',
          margin: '6px 0',
          gap: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div
            style={{
              width: 26,
              height: 26,
              borderRadius: 6,
              background: 'rgba(232, 98, 44, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </div>
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#f4f3ef' }}>{title}</div>
            {act.details && (
              <div style={{ fontSize: 11, color: '#a9aca7' }}>{act.details}</div>
            )}
          </div>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            padding: '2px 8px',
            borderRadius: 12,
            background: 'rgba(34, 197, 94, 0.15)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            fontSize: 10,
            color: '#5cc596',
            fontWeight: 600,
            whiteSpace: 'nowrap',
          }}
        >
          <CheckCircle2 size={11} />
          <span>MuJoCo Active</span>
        </div>
      </div>
    );
  };

  // Render a Single Message
  const renderMessage = (msg: PersistentChatMessage) => {
    const isUser = msg.sender === 'user';
    const resp = msg.response;
    const thoughts = resp?.thought_steps || [];
    const isThoughtsOpen = expandedThoughts[msg.id] ?? (msg.id === messages[messages.length - 1]?.id);
    const actions = resp?.actions_executed || [];
    const isThinking = isLoading && msg.id === messages[messages.length - 1]?.id;

    // Display text with typewriter if streaming
    let displayText = msg.text;
    let isCurrentlyStreaming = false;
    if (streamingMsgId === msg.id) {
      const words = msg.text.split(/(\s+)/);
      displayText = words.slice(0, displayedWordCount).join('');
      isCurrentlyStreaming = displayedWordCount < words.length;
    }

    return (
      <div
        key={msg.id}
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: isUser ? 'flex-end' : 'flex-start',
          marginBottom: 18,
          maxWidth: '100%',
        }}
      >
        {/* User Message */}
        {isUser ? (
          <div
            style={{
              maxWidth: '82%',
              background: '#242938',
              border: '1px solid #3b4252',
              borderRadius: '16px 16px 4px 16px',
              padding: '10px 16px',
              color: '#f4f3ef',
              fontSize: 13,
              lineHeight: 1.5,
              boxShadow: '0 2px 10px rgba(0, 0, 0, 0.25)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {msg.text}
          </div>
        ) : (
          /* Assistant Message (Claude / ChatGPT style) */
          <div
            style={{
              width: '100%',
              maxWidth: '96%',
              display: 'flex',
              gap: 12,
              alignItems: 'flex-start',
            }}
          >
            {/* Bot Avatar */}
            <div style={{ flexShrink: 0, marginTop: 1 }}>
              <AgentAvatar mood={isThinking ? 'thinking' : 'idle'} size={28} showBadge={false} />
            </div>

            {/* Content Column */}
            <div style={{ flex: 1, minWidth: 0 }}>
              {/* Collapsible Thinking Accordion */}
              {thoughts.length > 0 && (
                <div
                  style={{
                    marginBottom: 10,
                    borderRadius: 8,
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid rgba(255, 255, 255, 0.07)',
                    overflow: 'hidden',
                  }}
                >
                  <button
                    type="button"
                    onClick={() => toggleThoughts(msg.id)}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '6px 12px',
                      background: 'transparent',
                      border: 'none',
                      color: '#a9aca7',
                      cursor: 'pointer',
                      fontSize: 12,
                      fontWeight: 500,
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <BrainCircuit size={13} color="#e8622c" />
                      <span>
                        Thought process ({thoughts.length} step{thoughts.length > 1 ? 's' : ''})
                      </span>
                    </div>
                    {isThoughtsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </button>

                  {isThoughtsOpen && (
                    <div
                      style={{
                        padding: '8px 14px 12px',
                        borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                        borderLeft: '2px solid #a9aca7',
                        margin: '0 8px 8px',
                        background: 'rgba(15, 23, 42, 0.3)',
                        borderRadius: 4,
                      }}
                    >
                      {thoughts.map((st, sIdx) => (
                        <div key={sIdx} style={{ marginBottom: sIdx === thoughts.length - 1 ? 0 : 8 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, fontWeight: 600, color: '#ecebe6' }}>
                            <span
                              style={{
                                width: 6,
                                height: 6,
                                borderRadius: '50%',
                                background: st.status === 'success' ? '#5cc596' : st.status === 'running' ? '#e8622c' : '#dca23a',
                              }}
                            />
                            <span>{st.title}</span>
                          </div>
                          {st.detail && (
                            <div style={{ fontSize: 11, color: '#a9aca7', marginLeft: 12, marginTop: 2, lineHeight: 1.4 }}>
                              {st.detail}
                            </div>
                          )}
                        </div>
                      ))}

                      {resp?.thought_process && (
                        <div
                          style={{
                            marginTop: 10,
                            padding: '10px 12px',
                            borderRadius: 6,
                            background: 'rgba(2, 6, 23, 0.55)',
                            border: '1px solid rgba(255, 255, 255, 0.095)',
                            fontSize: 12,
                            color: '#d6d5cf',
                            lineHeight: 1.5,
                          }}
                        >
                          <div
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6,
                              marginBottom: 8,
                              color: '#a9aca7',
                              fontSize: 11,
                              fontWeight: 600,
                            }}
                          >
                            <BrainCircuit size={13} />
                            <span>Reasoning &amp; Actions</span>
                          </div>
                          {parseMarkdown(resp.thought_process)}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Executed Action Cards */}
              {actions.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                  {actions.map((act, aIdx) => renderActionCard(act, aIdx))}
                </div>
              )}

              {/* Main Markdown Text */}
              <div style={{ wordBreak: 'break-word' }}>
                {parseMarkdown(displayText)}
                {isCurrentlyStreaming && (
                  <span
                    style={{
                      display: 'inline-block',
                      width: 6,
                      height: 14,
                      marginLeft: 3,
                      verticalAlign: 'middle',
                      background: '#e8622c',
                      borderRadius: 1,
                      animation: 'pulse 1s infinite',
                    }}
                  />
                )}
              </div>

              {/* Supervised Approval Gate if awaiting approval */}
              {resp?.awaiting_approval && resp.approval_plan && (
                <div
                  style={{
                    marginTop: 10,
                    padding: '12px 14px',
                    borderRadius: 8,
                    background: 'rgba(234, 179, 8, 0.08)',
                    border: '1px solid rgba(234, 179, 8, 0.3)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#facc15', fontSize: 12, fontWeight: 600 }}>
                    <ShieldAlert size={15} />
                    <span>Supervised Action Confirmation Required</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#d6d5cf', marginTop: 4 }}>
                    {resp.approval_plan.title || 'Controlled Experiment Plan'}
                  </div>
                  {msg.approvalStatus === 'pending' || !msg.approvalStatus ? (
                    <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                      <button
                        type="button"
                        className="rf-btn rf-btn-primary"
                        onClick={() => handleApproveAction(msg.id, resp.approval_plan.mission_id || '', true)}
                        disabled={isLoading}
                        style={{ fontSize: 11, padding: '5px 12px', background: '#3fb27f', border: 'none' }}
                      >
                        <Check size={13} />
                        <span>Approve & Execute</span>
                      </button>
                      <button
                        type="button"
                        className="rf-btn"
                        onClick={() => handleApproveAction(msg.id, resp.approval_plan.mission_id || '', false)}
                        disabled={isLoading}
                        style={{ fontSize: 11, padding: '5px 12px', color: '#f07378' }}
                      >
                        <X size={13} />
                        <span>Reject</span>
                      </button>
                    </div>
                  ) : (
                    <div style={{ marginTop: 8, fontSize: 11, fontWeight: 600, color: msg.approvalStatus === 'approved' ? '#5cc596' : '#f07378' }}>
                      {msg.approvalStatus === 'approved' ? 'Approved' : 'Rejected'}
                    </div>
                  )}
                </div>
              )}

              {/* Bottom Actions: Copy / Retry */}
              <div style={{ display: 'flex', gap: 12, marginTop: 8, alignItems: 'center' }}>
                <button
                  type="button"
                  onClick={() => handleCopyMessage(msg.id, msg.text)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    background: 'transparent',
                    border: 'none',
                    color: copiedMsgId === msg.id ? '#e8622c' : '#8a8d89',
                    fontSize: 11,
                    cursor: 'pointer',
                    padding: 0,
                  }}
                  title="Copy message"
                >
                  {copiedMsgId === msg.id ? <Check size={12} /> : <Copy size={12} />}
                  <span>{copiedMsgId === msg.id ? 'Copied' : 'Copy'}</span>
                </button>

                {msg.lastPrompt && (
                  <button
                    type="button"
                    onClick={() => handleSendMessage(msg.lastPrompt)}
                    disabled={isLoading}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      background: 'transparent',
                      border: 'none',
                      color: '#8a8d89',
                      fontSize: 11,
                      cursor: 'pointer',
                      padding: 0,
                    }}
                    title="Retry this prompt"
                  >
                    <RefreshCw size={12} />
                    <span>Retry</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: isExpanded ? '90vh' : '100%',
        background: '#12141a',
        borderRadius: 8,
        border: '1px solid #232729',
        position: isExpanded ? 'fixed' : 'relative',
        top: isExpanded ? '5vh' : 'auto',
        left: isExpanded ? '5vw' : 'auto',
        width: isExpanded ? '90vw' : '100%',
        zIndex: isExpanded ? 9999 : 1,
        boxShadow: isExpanded ? '0 24px 80px rgba(0, 0, 0, 0.85)' : 'none',
        overflow: 'hidden',
      }}
    >
      {/* Top Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 16px',
          background: '#151718',
          borderBottom: '1px solid #232729',
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        {/* Assistant Identity & Model Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <AgentAvatar mood="speaking" size={26} />
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontWeight: 600, fontSize: 13, color: '#f4f3ef' }}>
                RoboForge Assistant
              </span>
              <span
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  padding: '2px 8px',
                  borderRadius: 12,
                  fontSize: 10,
                  fontWeight: 500,
                  background: 'rgba(232, 98, 44, 0.12)',
                  color: '#e8622c',
                  border: '1px solid rgba(232, 98, 44, 0.25)',
                }}
              >
                <span
                  style={{
                    width: 5,
                    height: 5,
                    borderRadius: '50%',
                    background: '#3fb27f',
                    boxShadow: '0 0 6px #3fb27f',
                  }}
                />
                <span>{aiStatus?.model || 'qwen2.5:7b'}</span>
              </span>
            </div>
          </div>
        </div>

        {/* Right Controls: Mode Toggle, Clear, Expand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Mode Segmented Control */}
          <div
            style={{
              display: 'flex',
              background: '#0d0f14',
              borderRadius: 6,
              padding: 2,
              border: '1px solid #232729',
            }}
          >
            <button
              type="button"
              onClick={() => {
                setMode('autonomous');
                onShowToast('Mode: Autonomous (Agent executes actions directly in MuJoCo)');
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 10px',
                borderRadius: 4,
                border: 'none',
                background: mode === 'autonomous' ? '#a9aca7' : 'transparent',
                color: mode === 'autonomous' ? '#ffffff' : '#8a8d89',
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
              title="Autonomous Mode: Executes tool commands directly in the simulation"
            >
              <Zap size={12} />
              <span>Autonomous</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setMode('supervised');
                onShowToast('Mode: Supervised (Requests approval for multi-trial plans)');
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '4px 10px',
                borderRadius: 4,
                border: 'none',
                background: mode === 'supervised' ? 'var(--accent-soft)' : 'transparent',
                color: mode === 'supervised' ? 'var(--accent)' : '#8a8d89',
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
              title="Supervised Mode: Formulates plans and requires operator confirmation"
            >
              <ShieldCheck size={12} />
              <span>Supervised</span>
            </button>
          </div>

          <button
            type="button"
            className="rf-btn"
            onClick={handleClearHistory}
            style={{ fontSize: 11, padding: '4px 8px' }}
            title="Clear Chat History"
          >
            <Trash2 size={13} />
          </button>
          <button
            type="button"
            className="rf-btn"
            onClick={() => setIsExpanded(!isExpanded)}
            style={{ fontSize: 11, padding: '4px 8px' }}
            title={isExpanded ? 'Collapse' : 'Expand full screen'}
          >
            {isExpanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
          </button>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '16px 20px',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Empty State / Welcome Screen */}
        {messages.length === 0 ? (
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '20px 0',
              textAlign: 'center',
            }}
          >
            <div style={{ marginBottom: 14 }}>
              <AgentAvatar mood="idle" size={60} />
            </div>

            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                padding: '3px 10px',
                borderRadius: 12,
                background: 'rgba(232, 98, 44, 0.1)',
                border: '1px solid rgba(232, 98, 44, 0.25)',
                color: '#e8622c',
                fontSize: 11,
                fontWeight: 600,
                marginBottom: 10,
              }}
            >
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#3fb27f', boxShadow: '0 0 6px #3fb27f' }} />
              ROBOAGENT CO-PILOT ONLINE
            </div>

            <h3 style={{ fontSize: 17, fontWeight: 700, color: '#f4f3ef', marginBottom: 6 }}>
              Hey! How can I assist your simulation?
            </h3>
            <p style={{ fontSize: 13, color: '#a9aca7', maxWidth: 460, lineHeight: 1.5, marginBottom: 24 }}>
              I can synthesize physical environments, spawn obstacles, write and train RL policies live in the viewport, tune motor kinematics, and optimize trajectories.
            </p>

            {/* Quick Starter Grid */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: 10,
                width: '100%',
                maxWidth: 620,
              }}
            >
              {quickActions.map((qa, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => handleSendMessage(qa.prompt)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-start',
                    padding: '12px 14px',
                    borderRadius: 8,
                    background: '#151718',
                    border: '1px solid #232729',
                    textAlign: 'left',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = '#e8622c';
                    e.currentTarget.style.background = '#1a1e2a';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = '#232729';
                    e.currentTarget.style.background = '#151718';
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    {qa.icon}
                    <span style={{ fontSize: 12, fontWeight: 600, color: '#f4f3ef' }}>{qa.title}</span>
                  </div>
                  <span style={{ fontSize: 11, color: '#8a8d89', lineHeight: 1.4 }}>{qa.desc}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map(renderMessage)
        )}

        {/* Friendly Animated Agent Thought Card */}
        {isLoading && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 14,
              padding: '12px 16px',
              margin: '10px 0',
              borderRadius: 12,
              background: 'var(--bg-panel)',
              border: '1px solid rgba(232, 98, 44, 0.3)',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4), 0 0 16px rgba(232, 98, 44, 0.1)',
              animation: 'fadeIn 0.3s ease-out',
            }}
          >
            <AgentAvatar mood="thinking" size={42} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: '#f4f3ef' }}>
                  RoboAgent is Reasoning
                </span>
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                    fontSize: 10,
                    fontWeight: 600,
                    padding: '2px 7px',
                    borderRadius: 10,
                    background: 'rgba(232, 98, 44, 0.15)',
                    color: '#e8622c',
                  }}
                >
                  <span
                    style={{
                      width: 5,
                      height: 5,
                      borderRadius: '50%',
                      background: '#e8622c',
                      boxShadow: '0 0 6px #e8622c',
                    }}
                  />
                  SIMULATION ACTIVE
                </span>
              </div>
              <span style={{ fontSize: 11, color: '#a9aca7' }}>
                Synthesizing MuJoCo physical dynamics, verifying kinematic bounds & orchestrating scene...
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Floating Modern Composer (Claude/ChatGPT Style) */}
      <div
        style={{
          padding: '12px 16px',
          background: '#151718',
          borderTop: '1px solid #232729',
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            background: '#0d0f14',
            borderRadius: 10,
            border: '1px solid #2b3140',
            padding: '8px 12px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          }}
        >
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder="Ask a question, or command the scene (e.g. 'spawn a red box at 2, 0', 'set speed to 2.5')..."
            disabled={isLoading}
            rows={1}
            style={{
              width: '100%',
              minHeight: 40,
              maxHeight: 160,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              resize: 'none',
              fontSize: 13,
              lineHeight: 1.5,
              color: '#f4f3ef',
              fontFamily: 'inherit',
            }}
          />

          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginTop: 4,
              paddingTop: 4,
              borderTop: '1px solid rgba(255, 255, 255, 0.04)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: '#8a8d89' }}>
              <span>Shift+Enter for newline</span>
              <span>&bull;</span>
              <span>Mode: {mode === 'autonomous' ? 'Autonomous tool use' : 'Supervised'}</span>
            </div>

            <button
              type="button"
              onClick={() => handleSendMessage()}
              disabled={isLoading || !inputValue.trim()}
              style={{
                width: 32,
                height: 32,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: inputValue.trim() && !isLoading ? '#c4472d' : '#232729',
                color: inputValue.trim() && !isLoading ? '#ffffff' : '#8a8d89',
                border: 'none',
                cursor: inputValue.trim() && !isLoading ? 'pointer' : 'not-allowed',
                transition: 'all 0.15s ease',
              }}
              title="Send message"
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
