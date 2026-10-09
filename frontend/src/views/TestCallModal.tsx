import React, { useState, useEffect, useRef } from 'react'
import type { Agent } from '../lib/types'
import { AudioIndicator } from '../components/ui/AudioIndicator'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Mic, MicOff, PhoneOff, AlertCircle, Headphones, Volume2 } from 'lucide-react'

export interface TestCallModalProps {
  agent: Agent
  onClose: () => void
}

interface TranscriptTurn {
  id: string
  role: 'user' | 'assistant'
  text: string
  interrupted?: boolean
}

// Inline AudioWorklet code: self-contained, no network dependency, eliminates ScriptProcessorNode
const AUDIO_CAPTURE_WORKLET_CODE = `
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.buffer = [];
    this.frameSize = 320; // 20ms at 16kHz
    this.rmsCounter = 0;
    this.rmsAccum = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channelData = input[0];
    if (!channelData || channelData.length === 0) return true;

    // Compute live input RMS for mic volume meter
    let sumSq = 0;
    for (let i = 0; i < channelData.length; i++) {
      const s = channelData[i];
      sumSq += s * s;
    }
    this.rmsAccum += sumSq;
    this.rmsCounter += channelData.length;

    if (this.rmsCounter >= 800) { // ~50ms updates
      const rms = Math.sqrt(this.rmsAccum / this.rmsCounter);
      this.port.postMessage({ type: 'mic_rms', rms });
      this.rmsAccum = 0;
      this.rmsCounter = 0;
    }

    // Downsample to 16,000 Hz if native rate differs
    const nativeRate = sampleRate;
    if (nativeRate === this.targetSampleRate) {
      for (let i = 0; i < channelData.length; i++) {
        this.buffer.push(channelData[i]);
      }
    } else {
      const ratio = nativeRate / this.targetSampleRate;
      const outputLength = Math.floor(channelData.length / ratio);
      for (let i = 0; i < outputLength; i++) {
        const srcIdx = i * ratio;
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(idx0 + 1, channelData.length - 1);
        const frac = srcIdx - idx0;
        const sample = channelData[idx0] * (1 - frac) + channelData[idx1] * frac;
        this.buffer.push(sample);
      }
    }

    // Drain fixed-size 320-sample frames (640 bytes PCM16) with adaptive speech gain boost
    while (this.buffer.length >= this.frameSize) {
      const chunk = this.buffer.splice(0, this.frameSize);
      let chunkSq = 0;
      for (let i = 0; i < this.frameSize; i++) {
        chunkSq += chunk[i] * chunk[i];
      }
      const cRms = Math.sqrt(chunkSq / this.frameSize);
      // Adaptive software gain: boost soft speech smoothly up to 3.5x
      const boost = cRms > 0.0015 && cRms < 0.035 ? Math.min(3.5, 0.035 / cRms) : 1.0;

      const pcm16 = new Int16Array(this.frameSize);
      for (let i = 0; i < this.frameSize; i++) {
        const val = chunk[i] * boost;
        const s = Math.max(-0.98, Math.min(0.98, val));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage({ type: 'pcm16', data: pcm16.buffer }, [pcm16.buffer]);
    }

    return true;
  }
}
registerProcessor('audio-capture-processor', AudioCaptureProcessor);
`

export const TestCallModal: React.FC<TestCallModalProps> = ({ agent, onClose }) => {
  const [callStatus, setCallStatus] = useState<
    'connecting' | 'priming' | 'connected' | 'ended' | 'error'
  >('connecting')
  const [primingElapsedSec, setPrimingElapsedSec] = useState(0)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isMuted, setIsMuted] = useState(false)
  const [muteWhileSpeaking, setMuteWhileSpeaking] = useState(false)
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false)
  const [micRms, setMicRms] = useState(0.0)
  const [speakerRms, setSpeakerRms] = useState(0.0)
  const [elapsedSec, setElapsedSec] = useState(0)
  const [turns, setTurns] = useState<TranscriptTurn[]>([])
  const [endReason, setEndReason] = useState<string | null>(null)
  const [sttSupported, setSttSupported] = useState<boolean>(true)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const nextPlayTimeRef = useRef<number>(0)
  const isAgentSpeakingRef = useRef<boolean>(false)
  const muteWhileSpeakingRef = useRef<boolean>(false)
  const isMutedRef = useRef<boolean>(false)
  const transcriptBoxRef = useRef<HTMLDivElement | null>(null)
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([])
  const recognitionRef = useRef<any>(null)
  const resetUserTurnRef = useRef<(() => void) | null>(null)
  const callStatusRef = useRef<'connecting' | 'priming' | 'connected' | 'ended' | 'error'>('connecting')

  // Sync refs for audio callbacks
  callStatusRef.current = callStatus
  isAgentSpeakingRef.current = isAgentSpeaking
  muteWhileSpeakingRef.current = muteWhileSpeaking
  isMutedRef.current = isMuted

  // 1. Elapsed Call Timer
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null
    if (callStatus === 'connected') {
      interval = setInterval(() => {
        setElapsedSec((s) => s + 1)
      }, 1000)
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [callStatus])

  // Scroll transcript box to bottom on new turns
  useEffect(() => {
    if (transcriptBoxRef.current) {
      transcriptBoxRef.current.scrollTop = transcriptBoxRef.current.scrollHeight
    }
  }, [turns])

  // 2. Initialize WebSocket & Web Audio Streaming with AudioWorklet & Jitter Buffer
  useEffect(() => {
    let isCleanedUp = false
    let pingInterval: ReturnType<typeof setInterval> | null = null

    async function startCall() {
      try {
        // Request Microphone Access with echo cancellation & noise suppression
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        })
        if (isCleanedUp) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        mediaStreamRef.current = stream

        // Initialize AudioContext at 16000 Hz and resume immediately inside user gesture
        const AudioContextClass =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
        const audioCtx = new AudioContextClass({ sampleRate: 16000 })
        audioContextRef.current = audioCtx
        if (audioCtx.state === 'suspended') {
          await audioCtx.resume()
        }

        // Build WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const host = window.location.host
        const voiceParam = agent.voice_id ? `&voice_prompt=${encodeURIComponent(agent.voice_id)}` : ''
        const wsUrl = `${protocol}//${host}/v2/voice?agent_id=${encodeURIComponent(
          agent.id || 'default'
        )}${voiceParam}&sample_rate=16000&codec=pcm16`

        const ws = new WebSocket(wsUrl)
        wsRef.current = ws
        ws.binaryType = 'arraybuffer'

        ws.onopen = () => {
          if (isCleanedUp) return
          ws.send(
            JSON.stringify({
              type: 'client_info',
              audio_context_state: audioCtx.state,
            })
          )
        }

        // Keepalive heartbeat every 3.0s to ensure cloud reverse proxies never timeout the connection
        pingInterval = setInterval(() => {
          if (!isCleanedUp && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }))
          }
        }, 3000)

        ws.onmessage = async (event) => {
          if (isCleanedUp) return

          if (typeof event.data === 'string') {
            try {
              const msg = JSON.parse(event.data)
              if (msg.type === 'status') {
                if (msg.status === 'priming') {
                  setCallStatus('priming')
                  setPrimingElapsedSec(Math.round((msg.elapsed_ms || 0) / 1000))
                } else if (msg.status === 'ready') {
                  setCallStatus('connected')
                }
              } else if (msg.type === 'session_started') {
                setCallStatus('connected')
                if (msg.session_id) {
                  setActiveSessionId(msg.session_id)
                }
              } else if (msg.type === 'transcript') {
                setIsAgentSpeaking(true)
                const role = msg.role || 'assistant'
                const text = msg.text || ''

                // If agent speaks, finish user's preceding turn for clean separation
                if (role !== 'user') {
                  resetUserTurnRef.current?.()
                }

                setTurns((prev) => {
                  if (prev.length > 0 && prev[prev.length - 1].role === role) {
                    const last = prev[prev.length - 1]
                    const updated = [...prev]
                    updated[updated.length - 1] = {
                      ...last,
                      text: last.text + text,
                    }
                    return updated
                  } else {
                    return [
                      ...prev,
                      {
                        id: `${role}-${Date.now()}-${Math.random()}`,
                        role,
                        text,
                      },
                    ]
                  }
                })

                // Reset speaking flag after short token pause
                setTimeout(() => {
                  if (audioCtx.currentTime >= nextPlayTimeRef.current - 0.05) {
                    setIsAgentSpeaking(false)
                  }
                }, 600)
              } else if (msg.type === 'call_ended') {
                setEndReason(msg.reason || 'Call ended')
                setCallStatus('ended')
                cleanupAudio()
              } else if (msg.type === 'error') {
                setErrorMessage(msg.message || 'Call error')
                setCallStatus('error')
                cleanupAudio()
              }
            } catch (err) {
              console.error('Error parsing WS message', err)
            }
          } else if (event.data instanceof ArrayBuffer) {
            // Play received 16 kHz PCM16 audio via jitter buffer
            playPcm16ChunkWithJitterBuffer(event.data)
          }
        }

        ws.onerror = (e) => {
          console.error('WebSocket error', e)
          if (!isCleanedUp) {
            setErrorMessage((prev) => prev || 'WebSocket error: connection failed')
            setCallStatus('error')
          }
        }

        ws.onclose = (e) => {
          if (!isCleanedUp) {
            if (e.code !== 1000 && e.code !== 1001 && e.code !== 1005) {
              setErrorMessage(
                (prev) => prev || (e.reason ? `Server error: ${e.reason} (code ${e.code})` : `Server disconnected (code ${e.code})`)
              )
              setCallStatus('error')
            } else {
              setCallStatus((curr) => {
                if (curr === 'connecting' || curr === 'priming') {
                  setErrorMessage(
                    (prev) => prev || 'Call disconnected before audio session could start. Verify that the PersonaPlex worker is running on the server.'
                  )
                  return 'error'
                }
                return 'ended'
              })
            }
          }
        }

        // Set up Microphone capture via inline AudioWorklet
        const source = audioCtx.createMediaStreamSource(stream)
        const muteGain = audioCtx.createGain()
        muteGain.gain.value = 0 // Mute mic loopback to speaker to prevent self-echo

        const blob = new Blob([AUDIO_CAPTURE_WORKLET_CODE], { type: 'application/javascript' })
        const workletUrl = URL.createObjectURL(blob)
        await audioCtx.audioWorklet.addModule(workletUrl)
        URL.revokeObjectURL(workletUrl)

        const workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor')
        workletNodeRef.current = workletNode

        let userSpeechConsecutive = 0

        workletNode.port.onmessage = (e) => {
          if (isCleanedUp) return
          const data = e.data
          if (data.type === 'mic_rms') {
            setMicRms((prev) => prev * 0.7 + data.rms * 0.3)

            // Interruption & Barge-in detection (responsive threshold)
            if (data.rms > 0.012) {
              userSpeechConsecutive++
              // Re-sync playback cursor to current audio context time to prevent turn delay buildup
              nextPlayTimeRef.current = audioCtx.currentTime
              if (userSpeechConsecutive >= 2 && isAgentSpeakingRef.current) {
                // User is speaking over agent: flush playback jitter buffer immediately
                flushPlaybackBuffer()
                setIsAgentSpeaking(false)
                if (ws.readyState === WebSocket.OPEN) {
                  ws.send(JSON.stringify({ type: 'interrupt' }))
                }
                setTurns((prev) => {
                  if (prev.length > 0 && prev[prev.length - 1].role === 'assistant') {
                    const last = prev[prev.length - 1]
                    if (!last.interrupted) {
                      const updated = [...prev]
                      updated[updated.length - 1] = {
                        ...last,
                        interrupted: true,
                        text: last.text + ' [Interrupted]',
                      }
                      return updated
                    }
                  }
                  return prev
                })
              }
            } else {
              userSpeechConsecutive = 0
            }
          } else if (data.type === 'pcm16' && data.data) {
            if (
              callStatusRef.current !== 'connected' ||
              isMutedRef.current ||
              (muteWhileSpeakingRef.current && isAgentSpeakingRef.current) ||
              ws.readyState !== WebSocket.OPEN
            ) {
              return
            }
            ws.send(data.data)
          }
        }

        source.connect(workletNode)
        workletNode.connect(muteGain)
        muteGain.connect(audioCtx.destination)

        // Live User Speech Recognition (Client-side STT)
        // PersonaPlex is an end-to-end S2S model that only outputs agent text tokens.
        // Web Speech API transcribes the caller's microphone into the transcript box in real-time.
        const SpeechRecognitionClass =
          (window as any).SpeechRecognition ||
          (window as any).webkitSpeechRecognition
        if (SpeechRecognitionClass) {
          try {
            setSttSupported(true)
            const recognition = new SpeechRecognitionClass()
            recognition.continuous = true
            recognition.interimResults = true
            recognition.lang = 'en-US'

            let finalTranscript = ''

            resetUserTurnRef.current = () => {
              finalTranscript = ''
            }

            recognition.onresult = (event: any) => {
              if (isCleanedUp) return
              let interim = ''
              let hasFinalPart = false
              for (let i = event.resultIndex; i < event.results.length; ++i) {
                const textPart = event.results[i][0]?.transcript || ''
                if (event.results[i].isFinal) {
                  finalTranscript += (finalTranscript ? ' ' : '') + textPart.trim()
                  hasFinalPart = true
                } else {
                  interim += textPart
                }
              }
              const spoken = (finalTranscript + (interim ? ' ' + interim : '')).trim()
              if (!spoken) return

              setTurns((prev) => {
                if (prev.length > 0 && prev[prev.length - 1].role === 'user') {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    ...updated[updated.length - 1],
                    text: spoken,
                  }
                  return updated
                } else {
                  return [
                    ...prev,
                    {
                      id: `user-${Date.now()}-${Math.random()}`,
                      role: 'user',
                      text: spoken,
                    },
                  ]
                }
              })

              if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(
                  JSON.stringify({
                    type: 'user_transcript',
                    text: spoken,
                    is_final: hasFinalPart,
                  })
                )
              }
            }

            let restartTimeout: any = null
            const safeRestart = () => {
              if (isCleanedUp || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
              clearTimeout(restartTimeout)
              restartTimeout = setTimeout(() => {
                if (isCleanedUp || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
                try {
                  recognition.start()
                } catch (err) {
                  console.debug('Speech recognition restart retry:', err)
                }
              }, 150)
            }

            recognition.onerror = (e: any) => {
              console.debug('Speech recognition error event:', e?.error)
              if (e?.error === 'not-allowed') {
                return
              }
              safeRestart()
            }

            recognition.onend = () => {
              safeRestart()
            }

            recognition.start()
            recognitionRef.current = recognition
          } catch (e) {
            console.debug('Browser speech recognition initialization notice:', e)
          }
        } else {
          setSttSupported(false)
        }
      } catch (err: unknown) {
        console.error('Failed to start call', err)
        setErrorMessage(
          err instanceof Error
            ? err.message
            : 'Microphone permission denied or audio device unavailable.'
        )
        setCallStatus('error')
      }
    }

    startCall()

    return () => {
      isCleanedUp = true
      if (pingInterval) {
        clearInterval(pingInterval)
      }
      cleanupAudio()
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [agent.id])

  // Jitter-buffered playback for 16 kHz PCM16 chunks
  const playPcm16ChunkWithJitterBuffer = (buffer: ArrayBuffer) => {
    const audioCtx = audioContextRef.current
    if (!audioCtx) return

    const int16 = new Int16Array(buffer)
    const float32 = new Float32Array(int16.length)
    let sum = 0
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0
      sum += float32[i] * float32[i]
    }

    const level = Math.sqrt(sum / int16.length)
    setSpeakerRms((prev) => prev * 0.5 + level * 0.5)

    const audioBuffer = audioCtx.createBuffer(1, float32.length, 16000)
    audioBuffer.copyToChannel(float32, 0)

    const source = audioCtx.createBufferSource()
    source.buffer = audioBuffer
    source.connect(audioCtx.destination)

    const now = audioCtx.currentTime
    // Adaptive ultra-low-latency jitter buffer:
    // Minimum buffer: 25ms (0.025s) for smooth click-free onset (saves ~95ms vs 120ms)
    // Maximum lookahead: 85ms (0.085s) to strictly prevent buffer bloat across turns
    const minBufferDelay = 0.025
    const maxLookahead = 0.085

    let startAt: number
    if (nextPlayTimeRef.current < now) {
      startAt = now + minBufferDelay
    } else if (nextPlayTimeRef.current > now + maxLookahead) {
      // Buffer bloat / packet burst detected: clamp to maxLookahead to prevent latency accumulation
      startAt = now + maxLookahead
    } else {
      startAt = nextPlayTimeRef.current
    }

    source.start(startAt)
    nextPlayTimeRef.current = startAt + audioBuffer.duration
    activeSourcesRef.current.push(source)

    source.onended = () => {
      activeSourcesRef.current = activeSourcesRef.current.filter((s) => s !== source)
    }
  }

  // Flush playback buffer on interruption
  const flushPlaybackBuffer = () => {
    activeSourcesRef.current.forEach((src) => {
      try {
        src.stop()
        src.disconnect()
      } catch {
        // Source may already be stopped
      }
    })
    activeSourcesRef.current = []
    if (audioContextRef.current) {
      nextPlayTimeRef.current = audioContextRef.current.currentTime
    }
  }

  const cleanupAudio = () => {
    flushPlaybackBuffer()
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop()
      } catch {}
      recognitionRef.current = null
    }
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect()
      workletNodeRef.current = null
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop())
      mediaStreamRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(console.error)
      audioContextRef.current = null
    }
  }

  const handleHangup = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'hangup' }))
    }
    setEndReason('Ended by user')
    setCallStatus('ended')
    cleanupAudio()
  }

  const formatTimer = (sec: number) => {
    const m = Math.floor(sec / 60)
    const s = sec % 60
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs select-none">
      <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-3xl max-w-lg w-full p-7 shadow-2xl space-y-5 flex flex-col items-center text-center">
        {/* Status Badge & Header */}
        <div className="space-y-1 w-full">
          <div className="flex items-center justify-center gap-2">
            <Badge
              variant={
                callStatus === 'connected'
                  ? 'connected'
                  : callStatus === 'priming'
                  ? 'warning'
                  : callStatus === 'ended'
                  ? 'neutral'
                  : 'danger'
              }
              dot
            >
              {callStatus === 'connecting'
                ? 'Connecting Line...'
                : callStatus === 'priming'
                ? `Getting ready (${primingElapsedSec}s / ~9s)...`
                : callStatus === 'connected'
                ? isAgentSpeaking
                  ? `${agent.name} Speaking`
                  : 'Listening to You'
                : callStatus === 'ended'
                ? 'Call Ended'
                : 'Connection Failed'}
            </Badge>
            <span className="text-[12px] font-mono text-[#6B6963]">
              {formatTimer(elapsedSec)}
            </span>
          </div>

          <h2 className="text-[20px] font-semibold text-[#1F1E1D] pt-1">
            {agent.name}
          </h2>
          <p className="text-[13px] text-[#6B6963]">
            PersonaPlex 7B Native S2S Voice Agent (24 kHz Mimi)
          </p>
        </div>

        {/* Headphone Advisory Notice */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#F4F3F0] text-[#6B6963] text-[11px] font-medium border border-[rgba(31,30,29,0.06)]">
          <Headphones className="w-3.5 h-3.5 text-[#C2603F]" />
          <span>Please wear headphones for crisp full-duplex audio without feedback</span>
        </div>

        {/* Live Audio Level Meters */}
        <div className="grid grid-cols-2 gap-3 w-full py-2">
          {/* Microphone Meter */}
          <div className="bg-[#FBFBFA] border border-[rgba(31,30,29,0.08)] rounded-xl p-3 flex flex-col items-center gap-1.5">
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#6B6963]">
              <Mic className="w-3.5 h-3.5" />
              <span>Your Mic</span>
            </div>
            <AudioIndicator rms={isMuted ? 0 : micRms} isSpeaking={false} isListening={!isMuted && micRms > 0.015} size={36} />
            <span className="text-[10px] font-mono text-[#8C8980]">
              {isMuted ? 'MUTED' : `${Math.round(micRms * 100)}%`}
            </span>
          </div>

          {/* Speaker / Agent Audio Meter */}
          <div className="bg-[#FBFBFA] border border-[rgba(31,30,29,0.08)] rounded-xl p-3 flex flex-col items-center gap-1.5">
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#6B6963]">
              <Volume2 className="w-3.5 h-3.5 text-[#C2603F]" />
              <span>Agent Voice</span>
            </div>
            <AudioIndicator rms={speakerRms} isSpeaking={speakerRms > 0.015} isListening={false} size={36} />
            <span className="text-[10px] font-mono text-[#8C8980]">
              {isAgentSpeaking || speakerRms > 0.015 ? `${Math.round(speakerRms * 100)}%` : 'SILENT'}
            </span>
          </div>
        </div>

        {/* Live Conversation Transcript Feed */}
        <div
          ref={transcriptBoxRef}
          className="w-full bg-[#FBFBFA] border border-[rgba(31,30,29,0.08)] rounded-2xl h-44 overflow-y-auto p-4 text-left space-y-3 font-sans text-[13px] leading-relaxed"
        >
          {errorMessage ? (
            <div className="flex items-start gap-2 text-[#A63A38] text-[13px] pt-4 justify-center">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{errorMessage}</span>
            </div>
          ) : endReason ? (
            <div className="text-[13px] text-[#6B6963] italic text-center py-8">
              {endReason}
            </div>
          ) : turns.length > 0 ? (
            turns.map((turn) => (
              <div
                key={turn.id}
                className={`flex flex-col ${
                  turn.role === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <span className="text-[10px] text-[#8C8980] mb-0.5 uppercase tracking-wider font-semibold">
                  {turn.role === 'user' ? 'You' : agent.name}
                  {turn.interrupted && ' (Barge-in)'}
                </span>
                <div
                  className={`p-2.5 rounded-xl max-w-[85%] ${
                    turn.role === 'user'
                      ? 'bg-[#1F1E1D] text-white'
                      : 'bg-[#FFFFFF] border border-[rgba(31,30,29,0.1)] text-[#1F1E1D]'
                  }`}
                >
                  {turn.text}
                </div>
              </div>
            ))
          ) : (
            <div className="h-full flex items-center justify-center text-[#9E9B93] text-[13px] italic">
              {callStatus === 'connecting'
                ? 'Connecting to speech gateway...'
                : callStatus === 'priming'
                ? `Loading PersonaPlex model state (${primingElapsedSec}s elapsed)...`
                : isMuted
                ? 'Microphone muted. Unmute to speak.'
                : 'Speak naturally into your microphone...'}
            </div>
          )}
        </div>

        {/* Echo Suppression Option */}
        <label className="flex items-center gap-2 text-[12px] text-[#6B6963] cursor-pointer select-none">
          <input
            type="checkbox"
            checked={muteWhileSpeaking}
            onChange={(e) => setMuteWhileSpeaking(e.target.checked)}
            className="rounded border-[rgba(31,30,29,0.2)] text-[#1F1E1D] focus:ring-0"
          />
          <span>Mute mic while agent speaks (prevents self-talk feedback)</span>
        </label>

        {!sttSupported && (
          <p className="text-[11px] text-[#8C8980]">
            PersonaPlex processes audio natively. Live caller transcript requires Google Chrome or Microsoft Edge.
          </p>
        )}

        {/* Action Controls */}
        <div className="flex items-center gap-4 pt-1">
          {callStatus === 'connected' && (
            <Button
              variant={isMuted ? 'destructive' : 'secondary'}
              size="regular"
              leftIcon={isMuted ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              onClick={() => setIsMuted(!isMuted)}
            >
              {isMuted ? 'Unmute Mic' : 'Mute Mic'}
            </Button>
          )}

          {callStatus === 'connected' || callStatus === 'priming' ? (
            <Button
              variant="destructive"
              size="regular"
              leftIcon={<PhoneOff className="w-4 h-4" />}
              onClick={handleHangup}
            >
              End Call
            </Button>
          ) : (
            <div className="flex items-center gap-2">
              {activeSessionId && (
                <a
                  href={`/v2/conversations/${activeSessionId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-3 py-1.5 rounded-lg border border-[rgba(31,30,29,0.15)] bg-white text-[12px] font-medium text-[#1F1E1D] hover:bg-[#F4F3F0] transition-colors flex items-center gap-1.5 shadow-xs"
                >
                  <span>Download Conversation JSON</span>
                </a>
              )}
              <Button variant="secondary" size="regular" onClick={onClose}>
                Dismiss
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
