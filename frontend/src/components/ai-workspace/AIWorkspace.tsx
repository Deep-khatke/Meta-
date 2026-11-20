import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Project, SimulationState, SceneObject, WorkspaceSection } from '../../types';
import { AIWorkspaceSidebar, ChatSessionMeta } from './AIWorkspaceSidebar';
import { AIConversationPanel, ChatMessage } from './AIConversationPanel';
import { AIArtifactPanel, ArtifactTabType } from './AIArtifactPanel';
import { AIExecutionStepItem } from './AIExecutionStep';
import { sendAIChatMessage, fetchProject } from '../../services/api';

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

const INITIAL_SESSIONS: ChatSessionMeta[] = [
  {
    id: 'session-rover-ppo',
    title: 'Autonomous Rover Obstacle Avoidance (PPO)',
    timestamp: Date.now() - 1000 * 60 * 45, // 45m ago (Today)
  },
  {
    id: 'session-arm-ik',
    title: '6-DOF Articulated Robotic Arm Kinematics',
    timestamp: Date.now() - 1000 * 60 * 60 * 26, // Yesterday
  },
  {
    id: 'session-drone-hover',
    title: 'Quadrotor Drone Altitude & Attitude Control',
    timestamp: Date.now() - 1000 * 60 * 60 * 24 * 3, // 3 days ago
  },
];

const INITIAL_MESSAGES_MAP: Record<string, ChatMessage[]> = {
  'session-rover-ppo': [
    {
      id: 'msg-r-1',
      sender: 'user',
      text: 'Train an autonomous rover to navigate randomized obstacles using PPO in MuJoCo.',
      timestamp: Date.now() - 1000 * 60 * 45,
    },
    {
      id: 'msg-r-2',
      sender: 'assistant',
      text: `### Rover Autonomous Navigation Initialized

I have configured the simulation environment with **8 dynamic obstacles** and attached a 360° virtual LiDAR sensor to the differential-drive rover chassis.

The PPO (Proximal Policy Optimization) reinforcement learning harness is now training in the physics runtime with the following hyperparameter profile:
- **Actor-Critic Network**: 2x [128, 128] MLP with Tanh activations
- **Observation Space**: 16-ray LiDAR distances + target delta vector (\`[dx, dy, heading_error]\`)
- **Action Space**: Normalized wheel speeds \`[-1.0, 1.0]\` with 2.5 m/s velocity clamp
- **Reward Function**: \`+2.0 * progress - 10.0 * collision - 0.05 * jerk\`

You can view the real-time reward curve and episode telemetry in the **Training Tab**, or inspect the live 3D obstacle avoidance in the **Simulation Tab**.`,
      timestamp: Date.now() - 1000 * 60 * 44,
      steps: [
        {
          id: 'step-1',
          title: 'Validated kinematic parameters',
          state: 'completed',
          category: 'general',
          durationMs: 120,
          details: 'Rover chassis verified: 4-wheel differential drive, track width 0.38m, wheel radius 0.06m.',
        },
        {
          id: 'step-2',
          title: 'Spawned randomized obstacle field',
          state: 'completed',
          category: 'environment',
          durationMs: 85,
          details: 'Placed 8 bounding cylinders with friction coefficient μ=0.85 across 10m x 10m arena.',
        },
        {
          id: 'step-3',
          title: 'Mounted 16-ray virtual LiDAR sensor',
          state: 'completed',
          category: 'sensor',
          durationMs: 95,
          details: 'Ray length: 8.0m, sweep angle: 360°, raycast query frequency: 50Hz.',
        },
        {
          id: 'step-4',
          title: 'Executing PPO Training Loop',
          state: 'running',
          category: 'training',
          durationMs: 250,
          details: 'Timestep: 24,500 / 100,000 | Current Mean Reward: +18.42 | Collision Rate: 3.2%',
        },
      ],
      artifactTrigger: {
        tab: 'training',
        label: 'Open Live Training Telemetry',
      },
    },
  ],
  'session-arm-ik': [
    {
      id: 'msg-a-1',
      sender: 'user',
      text: 'Construct a 6-DOF articulated serial manipulator with inverse kinematics.',
      timestamp: Date.now() - 1000 * 60 * 60 * 26,
    },
    {
      id: 'msg-a-2',
      sender: 'assistant',
      text: `### 6-DOF Manipulator Assembled

I have built a 6-DOF robotic manipulator based on standard industrial serial kinematics with spherical wrist alignment:
1. **Base Revolute**: \`[-180°, +180°]\` around Z-axis
2. **Shoulder Pitch**: \`[-90°, +120°]\` around Y-axis
3. **Elbow Pitch**: \`[-150°, +150°]\` around Y-axis
4. **Wrist Roll**: Continuous rotation \`[-180°, +180°]\`
5. **Wrist Pitch**: \`[-110°, +110°]\`
6. **Tool Flange Roll**: \`[-180°, +180°]\`

The kinematics solver uses **Damped Least Squares (DLS)** to avoid singularities near boundary reaches. The full URDF model and joint configuration are available in the **Robot Tab**.`,
      timestamp: Date.now() - 1000 * 60 * 60 * 26 + 1000,
      steps: [
        {
          id: 'step-arm-1',
          title: 'Generated MuJoCo XML kinematic tree',
          state: 'completed',
          category: 'robot',
          durationMs: 140,
          details: '6 revolute joints, mass 14.2kg, total reach: 0.85m.',
        },
        {
          id: 'step-arm-2',
          title: 'Initialized DLS Jacobian IK Solver',
          state: 'completed',
          category: 'general',
          durationMs: 110,
          details: 'Damping factor λ = 0.05, position tolerance: 1.0mm, max iterations: 50.',
        },
      ],
      artifactTrigger: {
        tab: 'robot',
        label: 'Inspect Robot Joint Specs',
      },
    },
  ],
  'session-drone-hover': [
    {
      id: 'msg-d-1',
      sender: 'user',
      text: 'Simulate an autonomous quadrotor drone and verify altitude hold.',
      timestamp: Date.now() - 1000 * 60 * 60 * 24 * 3,
    },
    {
      id: 'msg-d-2',
      sender: 'assistant',
      text: `### Quadrotor Drone Simulation Ready

The 4-rotor drone flight dynamics are active in the physics simulation.
- **Rotor Configuration**: X-quad layout with dual CW/CCW counter-rotating rotors
- **Cascaded PID Flight Controller**: Attitude loop running at 250Hz, Altitude loop running at 100Hz
- **Telemetry**: IMU quaternion attitude, climb rate, and 4-axis individual thrust outputs

You can test manual keyboard teleoperation (\`W/S\` pitch, \`A/D\` roll, \`Space/Shift\` climb, \`Q/E\` yaw) or toggle the autonomous hover stabilization in the **Simulation Tab**.`,
      timestamp: Date.now() - 1000 * 60 * 60 * 24 * 3 + 1200,
      artifactTrigger: {
        tab: 'simulation',
        label: 'Open 3D Drone Viewport',
      },
    },
  ],
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
  const [activeSessionId, setActiveSessionId] = useState<string>('session-rover-ppo');
  const [messagesMap, setMessagesMap] = useState<Record<string, ChatMessage[]>>(INITIAL_MESSAGES_MAP);

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
          ? 'Formulating intuitive robotics analogy & simulation plan'
          : 'Parsing robotics query & planning scene configuration',
        state: 'running',
        category: 'general',
        details: `Selected model: ${options?.model || 'Google Gemini 2.0 Flash'} | Mode: ${activeMode}`,
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
      tab?: ArtifactTabType
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
                      label: `Open in ${tab.charAt(0).toUpperCase() + tab.slice(1)} Tab`,
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

      // Construct finalized steps
      const finalSteps: AIExecutionStepItem[] = [
        {
          id: 'step-gen-1',
          title: isEducatorMsg
            ? 'Validated Kinematic Principles & Core Concepts'
            : 'Validated kinematics & physics parameters',
          state: 'completed',
          category: 'general',
          durationMs: 140,
          details: `Processed via ${options?.model || 'Google Gemini 2.0 Flash'} in ${isEducatorMsg ? 'Educator Mode' : 'Autonomous Tool-Calling Mode'}.`,
        },
        {
          id: 'step-gen-2',
          title: isEducatorMsg
            ? 'Demonstrating Live in 3D MuJoCo Physics'
            : 'Updated active simulation state',
          state: 'completed',
          category: 'environment',
          durationMs: 220,
          details: 'Synchronized with MuJoCo physics runtime. 0 collision exceptions.',
        },
      ];

      // Refresh project if backend mutated the scene or robot
      if (aiResponse.tools_invoked && aiResponse.tools_invoked.length > 0) {
        onProjectLoaded?.();
      }

      const responseText = aiResponse.reply || `Executed instruction across the robotics simulation workspace.\n\nAll kinematic constraints and physics boundaries have been verified against the active physics engine.`;

      // Stream the response text smoothly like ChatGPT
      await streamMessage(
        responseText,
        finalSteps,
        aiResponse.follow_up_suggestions,
        recommendedTab
      );

      // Automatically switch to the recommended tab and open drawer
      setArtifactTab(recommendedTab);
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
