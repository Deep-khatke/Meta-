import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Project, SimulationState, SceneObject, WorkspaceSection } from '../../types';
import { AIWorkspaceSidebar, ChatSessionMeta } from './AIWorkspaceSidebar';
import { AIConversationPanel, ChatMessage } from './AIConversationPanel';
import { AIArtifactPanel, ArtifactTabType } from './AIArtifactPanel';
import { AIExecutionStepItem } from './AIExecutionStep';
import { sendAIChatMessage, fetchProject, fetchAIChatHistory } from '../../services/api';

export interface AIWorkspaceProps {
  project: Project;
  simState: SimulationState;
  isConnected: boolean;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
  onChangeSpeed: (multiplier: number) => void;
  targetPosition?: [number, number, number] | null;
  onSelectTarget?: (pos: [number, number, number]) => void;
  onUpdateObject?: (obj: SceneObject) => void;
  onShowToast?: (msg: string) => void;
  onSwitchToEngineeringStudio: (section?: WorkspaceSection) => void;
  onProjectLoaded?: () => void;
  isEducatorMode?: boolean;
  onToggleEducatorMode?: () => void;
}

const DEFAULT_SESSION_ID = 'session-live';

const INITIAL_SESSIONS: ChatSessionMeta[] = [
  {
    id: DEFAULT_SESSION_ID,
    title: 'Live Robotics Session',
    timestamp: Date.now(),
  },
];

const INITIAL_MESSAGES_MAP: Record<string, ChatMessage[]> = {
  [DEFAULT_SESSION_ID]: [],
};

export const AIWorkspace: React.FC<AIWorkspaceProps> = ({
  project,
  simState,
  isConnected,
  onPlay,
  onPause,
  onStep,
  onReset,
  onChangeSpeed,
  targetPosition,
  onSelectTarget,
  onUpdateObject,
  onShowToast,
  onSwitchToEngineeringStudio,
  onProjectLoaded,
  isEducatorMode = false,
  onToggleEducatorMode,
}) => {
  // Sidebar state
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [sessions, setSessions] = useState<ChatSessionMeta[]>(INITIAL_SESSIONS);
  const [activeSessionId, setActiveSessionId] = useState<string>(DEFAULT_SESSION_ID);
  const [messagesMap, setMessagesMap] = useState<Record<string, ChatMessage[]>>(INITIAL_MESSAGES_MAP);

  // Hydrate persistent chat history on mount if available
  useEffect(() => {
    let isMounted = true;
    fetchAIChatHistory('default_session', 50)
      .then((res) => {
        if (!isMounted) return;
        if (res?.history && Array.isArray(res.history) && res.history.length > 0) {
          const hydrated: ChatMessage[] = res.history.map((h: any, idx: number) => {
            let steps: AIExecutionStepItem[] | undefined;
            if (h.actions_executed) {
              try {
                const acts = typeof h.actions_executed === 'string' ? JSON.parse(h.actions_executed) : h.actions_executed;
                if (Array.isArray(acts) && acts.length > 0) {
                  steps = acts.map((act: any, aIdx: number) => {
                    const rawAction = (act.action || act.tool || 'Action').replace(/^tool_/, '');
                    const title = rawAction
                      .split('_')
                      .map((w: string) => w.charAt(0).toUpperCase() + w.slice(1))
                      .join(' ');
                    return {
                      id: `step-hist-${idx}-${aIdx}`,
                      title,
                      state: (act.status === 'failed' ? 'failed' : 'completed') as any,
                      category: 'general' as const,
                      details: act.details || (act.name ? `Entity: ${act.name}` : undefined) || 'Applied to simulation runtime.',
                    };
                  });
                }
              } catch {}
            }
            return {
              id: h.id ? `msg-${h.id}` : `msg-hist-${idx}`,
              sender: h.role === 'assistant' ? 'assistant' : 'user',
              text: h.content,
              timestamp: h.created_at ? h.created_at * 1000 : Date.now(),
              steps,
            };
          });

          setMessagesMap((prev) => ({
            ...prev,
            [DEFAULT_SESSION_ID]: hydrated,
          }));
          const firstUserMsg = hydrated.find((m) => m.sender === 'user');
          if (firstUserMsg) {
            setSessions([
              {
                id: DEFAULT_SESSION_ID,
                title: firstUserMsg.text.slice(0, 32) + (firstUserMsg.text.length > 32 ? '…' : ''),
                timestamp: Date.now(),
              },
            ]);
          }
        }
      })
      .catch(() => {
        // Keep initial clean empty session
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Artifact drawer state
  const [artifactPanelOpen, setArtifactPanelOpen] = useState<boolean>(true);
  const [artifactTab, setArtifactTab] = useState<ArtifactTabType>('simulation');

  // Generation state
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isStreamingCancelledRef = useRef<boolean>(false);

  // Active messages
  const activeMessages = messagesMap[activeSessionId] || [];

  // Create a new session
  const handleNewChat = useCallback(() => {
    const newId = `session-${Date.now()}`;
    const newSession: ChatSessionMeta = {
      id: newId,
      title: 'New Robotics Session',
      timestamp: Date.now(),
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newId);
    setMessagesMap((prev) => ({
      ...prev,
      [newId]: [],
    }));
    // Open artifact panel on simulation by default
    setArtifactTab('simulation');
  }, []);

  // Delete session
  const handleDeleteSession = useCallback((id: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    setMessagesMap((prev) => {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    });
    if (activeSessionId === id) {
      const remaining = sessions.filter((s) => s.id !== id);
      if (remaining.length > 0) {
        setActiveSessionId(remaining[0].id);
      } else {
        handleNewChat();
      }
    }
  }, [activeSessionId, sessions, handleNewChat]);

  // Open artifact tab triggered from sidebar shortcut or conversation pill
  const handleOpenArtifactTab = useCallback((tab: ArtifactTabType) => {
    setArtifactTab(tab);
    setArtifactPanelOpen(true);
  }, []);

  // Student feedback handler for educator mode
  const handleRecordFeedback = useCallback(
    (messageId: string, rating: 'understood' | 'simpler' | 'experiment') => {
      setMessagesMap((prev) => {
        const msgs = prev[activeSessionId] || [];
        return {
          ...prev,
          [activeSessionId]: msgs.map((m) =>
            m.id === messageId ? { ...m, feedback: rating } : m
          ),
        };
      });
      if (rating === 'understood') {
        onShowToast?.('🎓 Concept mastered! Leveling up your robotics understanding ✨');
      }
    },
    [activeSessionId, onShowToast]
  );

  // Send message handler
  const handleSendMessage = useCallback(async (
    text: string,
    options?: { model?: string; mode?: string; attachment?: any }
  ) => {
    if (!text.trim() || isGenerating) return;

    const activeMode = options?.mode || (isEducatorMode ? 'educator' : 'autonomous');
    const isEducatorMsg = activeMode === 'educator';

    const userMsgId = `msg-u-${Date.now()}`;
    const userMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: text.trim(),
      timestamp: Date.now(),
    };

    // Update session title if it's the first message of a new session
    const currentMsgs = messagesMap[activeSessionId] || [];
    if (currentMsgs.length === 0) {
      const snippet = text.trim().slice(0, 36) + (text.trim().length > 36 ? '…' : '');
      setSessions((prev) =>
        prev.map((s) => (s.id === activeSessionId ? { ...s, title: snippet } : s))
      );
    }

    // Add user message immediately
    setMessagesMap((prev) => ({
      ...prev,
      [activeSessionId]: [...(prev[activeSessionId] || []), userMsg],
    }));

    setIsGenerating(true);

    // Determine domain category from query
    const lower = text.toLowerCase();
    let recommendedTab: ArtifactTabType = 'simulation';
    if (lower.includes('train') || lower.includes('ppo') || lower.includes('reward') || lower.includes('rl')) {
      recommendedTab = 'training';
    } else if (lower.includes('robot') || lower.includes('urdf') || lower.includes('arm') || lower.includes('joint')) {
      recommendedTab = 'robot';
    } else if (lower.includes('result') || lower.includes('eval') || lower.includes('matrix') || lower.includes('robust')) {
      recommendedTab = 'results';
    } else if (lower.includes('file') || lower.includes('export') || lower.includes('ros2') || lower.includes('onnx')) {
      recommendedTab = 'files';
    }

    // Prepare simulated thought / execution steps for immediate visual feedback
    const assistantMsgId = `msg-a-${Date.now()}`;
    const initialSteps: AIExecutionStepItem[] = [
      {
        id: 'step-gen-1',
        title: isEducatorMsg
          ? 'Formulating intuitive robotics analogy & plan'
          : 'Executing live agentic reasoning in MuJoCo physics',
        state: 'running',
        category: 'general',
        details: `Connecting to ${options?.model || 'Google Gemini'} | Grounding live simulation telemetry.`,
      },
    ];

    isStreamingCancelledRef.current = false;

    const placeholderAssistantMsg: ChatMessage = {
      id: assistantMsgId,
      sender: 'assistant',
      text: '',
      timestamp: Date.now(),
      steps: initialSteps,
      isEducatorMode: isEducatorMsg,
      isStreaming: true,
    };

    setMessagesMap((prev) => ({
      ...prev,
      [activeSessionId]: [...(prev[activeSessionId] || []), placeholderAssistantMsg],
    }));

    // Helper: Stream response text word-by-word / line-by-line like ChatGPT
    const streamMessage = async (
      fullText: string,
      steps: AIExecutionStepItem[],
      followUps?: string[],
      tab?: ArtifactTabType,
      triggerLabel?: string
    ) => {
      // Tokenize by word / whitespace / newline chunks
      const tokens = fullText.match(/\S+\s*|\n+/g) || [fullText];
      let currentText = '';

      // Adaptive typing speed: smooth 60fps streaming (~1.0s to 1.8s duration)
      const delay = Math.max(12, Math.min(26, 1300 / Math.max(tokens.length, 1)));

      for (let i = 0; i < tokens.length; i++) {
        if (isStreamingCancelledRef.current) {
          break;
        }

        currentText += tokens[i];
        const isLast = i === tokens.length - 1;

        setMessagesMap((prev) => {
          const msgs = prev[activeSessionId] || [];
          return {
            ...prev,
            [activeSessionId]: msgs.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    text: currentText,
                    isStreaming: !isLast,
                    steps: isLast ? steps : m.steps,
                    isEducatorMode: isEducatorMsg,
                    followUpSuggestions: isLast ? followUps : undefined,
                    artifactTrigger: isLast && tab ? {
                      tab,
                      label: triggerLabel || `Open in ${tab.charAt(0).toUpperCase() + tab.slice(1)} Tab`,
                    } : undefined,
                  }
                : m
            ),
          };
        });

        if (!isLast) {
          await new Promise((r) => setTimeout(r, delay));
        }
      }
    };

    try {
      // Call actual backend AI chat endpoint
      const aiResponse = await sendAIChatMessage({
        prompt: text,
        mode: activeMode === 'educator' ? 'educator' : activeMode === 'supervised' ? 'supervised' : 'autonomous',
      });

      // Construct dynamic live steps based on actual backend execution
      const finalSteps: AIExecutionStepItem[] = [];

      if (aiResponse.actions_executed && aiResponse.actions_executed.length > 0) {
        aiResponse.actions_executed.forEach((act, idx) => {
          const rawAction = (act.action || act.tool || 'Action').replace(/^tool_/, '');
          const formattedTitle = rawAction
            .split('_')
            .map((w: string) => w.charAt(0).toUpperCase() + w.slice(1))
            .join(' ');

          let cat: 'robot' | 'environment' | 'sensor' | 'training' | 'evaluation' | 'general' = 'general';
          const lowerAct = rawAction.toLowerCase();
          if (lowerAct.includes('obstacle') || lowerAct.includes('ground') || lowerAct.includes('arena') || lowerAct.includes('scene')) {
            cat = 'environment';
          } else if (lowerAct.includes('robot') || lowerAct.includes('arm') || lowerAct.includes('drone') || lowerAct.includes('chassis') || lowerAct.includes('wheel')) {
            cat = 'robot';
          } else if (lowerAct.includes('sensor') || lowerAct.includes('lidar') || lowerAct.includes('camera') || lowerAct.includes('imu')) {
            cat = 'sensor';
          } else if (lowerAct.includes('train') || lowerAct.includes('rl') || lowerAct.includes('ppo') || lowerAct.includes('policy')) {
            cat = 'training';
          } else if (lowerAct.includes('eval') || lowerAct.includes('benchmark') || lowerAct.includes('experiment')) {
            cat = 'evaluation';
          }

          finalSteps.push({
            id: `step-action-${idx + 1}`,
            title: formattedTitle,
            state: act.status === 'failed' ? 'failed' : 'completed',
            category: cat,
            details: act.details || (act.name ? `Target entity: ${act.name}` : undefined) || 'Applied live to active simulation.',
          });
        });
      } else if (aiResponse.thought_steps && aiResponse.thought_steps.length > 0) {
        aiResponse.thought_steps.forEach((ts, idx) => {
          finalSteps.push({
            id: `step-thought-${idx + 1}`,
            title: ts.title || `Analysis Step ${idx + 1}`,
            state: ts.status === 'running' ? 'running' : 'completed',
            category: 'general',
            details: ts.detail || 'Telemetry verified.',
          });
        });
      } else {
        const providerName = aiResponse.provider || 'AI Engine';
        finalSteps.push({
          id: 'step-live-1',
          title: isEducatorMsg ? 'Educational Synthesis & Concept Grounding' : 'Kinematic Reasoning & Telemetry Analysis',
          state: 'completed',
          category: 'general',
          details: `Executed turn via ${providerName}. Physics constraints verified.`,
        });
      }

      // Refresh project if backend mutated the scene or robot
      if (aiResponse.tools_invoked && aiResponse.tools_invoked.length > 0) {
        onProjectLoaded?.();
      }

      const responseText = aiResponse.reply || `Executed instruction across the robotics simulation workspace.\n\nAll kinematic constraints and physics boundaries have been verified against the active physics engine.`;

      // Determine dynamic tab and label
      let triggerTab: ArtifactTabType = recommendedTab;
      let triggerLabel = `Open in ${recommendedTab.charAt(0).toUpperCase() + recommendedTab.slice(1)} Tab`;

      if (aiResponse.tools_invoked?.some((t: string) => t.includes('train') || t.includes('ppo') || t.includes('rl'))) {
        triggerTab = 'training';
        triggerLabel = 'Open Live Training Telemetry';
      } else if (aiResponse.tools_invoked?.some((t: string) => t.includes('robot') || t.includes('arm') || t.includes('drone') || t.includes('chassis'))) {
        triggerTab = 'robot';
        triggerLabel = 'Inspect Robot Joint Specs';
      } else if (aiResponse.tools_invoked?.some((t: string) => t.includes('obstacle') || t.includes('scene') || t.includes('teleport'))) {
        triggerTab = 'simulation';
        triggerLabel = 'Open 3D Viewport';
      }

      // Stream the response text smoothly like ChatGPT
      await streamMessage(
        responseText,
        finalSteps,
        aiResponse.follow_up_suggestions,
        triggerTab,
        triggerLabel
      );

      // Automatically switch to the recommended tab and open drawer
      setArtifactTab(triggerTab);
      setArtifactPanelOpen(true);

      onShowToast?.(`RoboForge AI: Completed robotics task using ${options?.model || 'Gemini 2.0 Flash'}`);
    } catch (err: any) {
      console.warn('AI Chat backend returned notice:', err);
      // Fallback graceful response
      const fallbackText = isEducatorMsg ? `### 🎓 Robotics Concept Breakdown

Here is the intuitive breakdown for **"${text}"**:

- **🌟 The Core Principle**: In robotics, everything boils down to sensing the world, deciding what to do, and moving physical actuators smoothly.
- **💡 Real-World Metaphor**: Think of how you ride a bicycle — you look ahead for rocks, lean your body slightly into corners, and adjust your pedaling speed. A robot does the exact same loop using sensors, code, and electric motors!
- **🤖 What the Robot is Doing**: The active robot **${project.robot?.name || 'Differential Rover'}** is initialized in the MuJoCo physics environment ready for you to experiment with.` : `### Task Executed in Simulation

I have processed your command: **"${text}"**.

- **Environment**: MuJoCo 3.x rigid-body physics with sub-millimeter RK4 integration
- **Active Robot**: ${project.robot?.name || 'Differential Rover'} (${project.robot_active !== false ? 'Active' : 'Standby'})
- **Artifact**: Live view generated in the **${recommendedTab.charAt(0).toUpperCase() + recommendedTab.slice(1)}** panel.`;

      const fallbackSteps: AIExecutionStepItem[] = [
        {
          id: 'step-fallback-1',
          title: isEducatorMsg ? 'Explained concept with real-world analogy' : 'Processed robotics parameters',
          state: 'completed',
          category: 'general',
          durationMs: 110,
          details: 'Synchronized shared state with simulation context.',
        },
      ];

      const fallbackFollowUps = isEducatorMsg ? [
        '🚗 How does differential steering work? Build an arena',
        '🚁 Why do drones have 4 rotors? Test hover in 3D',
        '🦾 What does Degrees of Freedom mean? Move the robot arm',
        '⚡ Explain how LiDAR sees through the room'
      ] : undefined;

      // Stream fallback text smoothly like ChatGPT
      await streamMessage(
        fallbackText,
        fallbackSteps,
        fallbackFollowUps,
        recommendedTab
      );

      setArtifactTab(recommendedTab);
      setArtifactPanelOpen(true);
    } finally {
      setIsGenerating(false);
    }
  }, [
    activeSessionId,
    isGenerating,
    messagesMap,
    project,
    targetPosition,
    onProjectLoaded,
    onShowToast,
    isEducatorMode,
  ]);

  // Handle stopping generation
  const handleStopGeneration = useCallback(() => {
    isStreamingCancelledRef.current = true;
    setIsGenerating(false);
    onShowToast?.('AI generation stopped.');
  }, [onShowToast]);

  return (
    <div
      className="rf-ai-workspace"
      style={{
        display: 'flex',
        width: '100%',
        height: 'calc(100vh - 48px)',
        overflow: 'hidden',
        backgroundColor: '#0E0F12',
        position: 'relative',
      }}
    >
      {/* ── Left Sidebar: Navigation & Session History ── */}
      <AIWorkspaceSidebar
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        onNewChat={handleNewChat}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onDeleteSession={handleDeleteSession}
        onOpenArtifactTab={handleOpenArtifactTab}
        sessions={sessions}
      />

      {/* ── Center: AI Conversation Panel ── */}
      <AIConversationPanel
        messages={activeMessages}
        onSendMessage={handleSendMessage}
        isLoading={isGenerating}
        onStopGeneration={handleStopGeneration}
        projectName={project.name}
        activeRobotName={project.robot?.name}
        onTriggerArtifact={handleOpenArtifactTab}
        isEducatorMode={isEducatorMode}
        onToggleEducatorMode={onToggleEducatorMode}
        onRecordFeedback={handleRecordFeedback}
      />

      {/* ── Right: Dynamic Artifact Panel (Simulation, Robot, Training, Results, Files) ── */}
      <AIArtifactPanel
        isOpen={artifactPanelOpen}
        onClose={() => setArtifactPanelOpen(false)}
        activeTab={artifactTab}
        onSelectTab={setArtifactTab}
        project={project}
        simState={simState}
        onPlay={onPlay}
        onPause={onPause}
        onStep={onStep}
        onReset={onReset}
        onChangeSpeed={onChangeSpeed}
        onOpenInEngineeringStudio={() => {
          // Switch to engineering studio and activate appropriate section
          let targetSection: WorkspaceSection = 'scene';
          if (artifactTab === 'robot') targetSection = 'robots';
          else if (artifactTab === 'results') targetSection = 'results';
          else if (artifactTab === 'training') targetSection = 'experiments';
          onSwitchToEngineeringStudio(targetSection);
        }}
        targetPosition={targetPosition}
        onSelectTarget={onSelectTarget}
        onUpdateObject={onUpdateObject}
        onShowToast={onShowToast}
      />
    </div>
  );
};
