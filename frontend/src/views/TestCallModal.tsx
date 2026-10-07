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
}

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

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const nextPlayTimeRef = useRef<number>(0)
  const isAgentSpeakingRef = useRef<boolean>(false)
  const muteWhileSpeakingRef = useRef<boolean>(false)
  const isMutedRef = useRef<boolean>(false)
  const transcriptBoxRef = useRef<HTMLDivElement | null>(null)

  // Sync refs for audio callbacks
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
        const wsUrl = `${protocol}//${host}/v2/voice?agent_id=${encodeURIComponent(
          agent.id
        )}&sample_rate=16000&codec=pcm16`

        const ws = new WebSocket(wsUrl)
        wsRef.current = ws
        ws.binaryType = 'arraybuffer'

        ws.onopen = () => {
          if (isCleanedUp) return
          // Send client diagnostics
          ws.send(
            JSON.stringify({
              type: 'client_info',
              audio_context_state: audioCtx.state,
            })
          )
        }

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
              } else if (msg.type === 'transcript') {
                setIsAgentSpeaking(true)
                const role = msg.role || 'assistant'
                const text = msg.text || ''

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
                (prev) => prev || e.reason || `Server disconnected (code ${e.code})`
              )
              setCallStatus('error')
            } else {
              setCallStatus((curr) =>
                curr === 'connected' || curr === 'priming' ? 'ended' : curr
              )
            }
          }
        }

        // Set up Microphone capture: Try AudioWorklet first, with graceful fallback
        const source = audioCtx.createMediaStreamSource(stream)
        const muteGain = audioCtx.createGain()
        muteGain.gain.value = 0

        let workletLoaded = false
        try {
          await audioCtx.audioWorklet.addModule('/audio-capture-worklet.js')
          const workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor')
          workletNodeRef.current = workletNode

          workletNode.port.onmessage = (e) => {
            if (isCleanedUp) return
            const data = e.data
            if (data.type === 'mic_rms') {
              setMicRms((prev) => prev * 0.7 + data.rms * 0.3)
            } else if (data.type === 'pcm16' && data.data) {
              if (
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
          workletLoaded = true
        } catch (workletErr) {
          console.warn('AudioWorklet unavailable, using fallback processor:', workletErr)
        }

        if (!workletLoaded) {
          // Fallback ScriptProcessorNode
          const processor = audioCtx.createScriptProcessor(1024, 1, 1)
          processorRef.current = processor

          processor.onaudioprocess = (e) => {
            if (isCleanedUp) return
            const inputData = e.inputBuffer.getChannelData(0)
            let sum = 0
            for (let i = 0; i < inputData.length; i++) {
              sum += inputData[i] * inputData[i]
            }
            const level = Math.sqrt(sum / inputData.length)
            setMicRms((prev) => prev * 0.7 + level * 0.3)

            if (
              isMutedRef.current ||
              (muteWhileSpeakingRef.current && isAgentSpeakingRef.current) ||
              ws.readyState !== WebSocket.OPEN
            ) {
              return
            }

            const pcm16 = new Int16Array(inputData.length)
            for (let i = 0; i < inputData.length; i++) {
              const s = Math.max(-1, Math.min(1, inputData[i]))
              pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
            }
            ws.send(pcm16.buffer)
          }

          source.connect(processor)
          processor.connect(muteGain)
          muteGain.connect(audioCtx.destination)
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
    // Target 120 ms jitter buffer: if next play time is in the past, prime buffer ahead
    const targetBufferDelay = 0.12
    const startAt =
      nextPlayTimeRef.current < now
        ? now + targetBufferDelay
        : nextPlayTimeRef.current

    source.start(startAt)
    nextPlayTimeRef.current = startAt + audioBuffer.duration
  }

  const cleanupAudio = () => {
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect()
      workletNodeRef.current = null
    }
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
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
                ? `Getting ready (${primingElapsedSec}s / ~10s)...`
                : callStatus === 'connected'
                ? isAgentSpeaking
                  ? `${agent.name} Speaking`
                  : 'Listening to You'
                : callStatus === 'ended'
                ? 'Call Concluded'
                : 'Connection Failed'}
            </Badge>
          </div>
          <h2 className="text-[22px] font-semibold text-[#1F1E1D] pt-1">{agent.name}</h2>
          <div className="text-[13px] text-[#6B6963] font-mono tabular-nums">
            {callStatus === 'connected'
              ? formatTimer(elapsedSec)
              : callStatus === 'priming'
              ? 'Priming PersonaPlex 7B...'
              : agent.voice_id}
          </div>
        </div>

        {/* Headphones Advisory Banner */}
        <div className="w-full flex items-center justify-center gap-2 px-3 py-1.5 bg-[#FAF9F5] border border-[rgba(31,30,29,0.08)] rounded-xl text-[12px] text-[#6B6963]">
          <Headphones className="w-3.5 h-3.5 text-[#1F1E1D]" />
          <span>Use headphones to prevent microphone echo & feedback loops</span>
        </div>

        {/* Main Audio Pulse Indicator */}
        <div className="py-2">
          <AudioIndicator
            rms={isAgentSpeaking ? speakerRms : micRms}
            isSpeaking={isAgentSpeaking}
            isListening={callStatus === 'connected' && !isMuted}
            size={88}
          />
        </div>

        {/* Dual Live Audio Level Meters (Mic vs Speaker) */}
        <div className="w-full grid grid-cols-2 gap-3 px-2">
          <div className="flex flex-col items-start gap-1 p-2 bg-[#FAF9F5] rounded-xl border border-[rgba(31,30,29,0.06)]">
            <div className="flex items-center justify-between w-full text-[11px] font-medium text-[#6B6963]">
              <span className="flex items-center gap-1">
                <Mic className="w-3 h-3" /> Mic Input
              </span>
              <span className="font-mono">{Math.round(micRms * 100)}%</span>
            </div>
            <div className="w-full h-1.5 bg-black/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-[#1F1E1D] transition-all duration-75 rounded-full"
                style={{ width: `${Math.min(100, Math.round(micRms * 300))}%` }}
              />
            </div>
          </div>

          <div className="flex flex-col items-start gap-1 p-2 bg-[#FAF9F5] rounded-xl border border-[rgba(31,30,29,0.06)]">
            <div className="flex items-center justify-between w-full text-[11px] font-medium text-[#6B6963]">
              <span className="flex items-center gap-1">
                <Volume2 className="w-3 h-3" /> Agent Audio
              </span>
              <span className="font-mono">{Math.round(speakerRms * 100)}%</span>
            </div>
            <div className="w-full h-1.5 bg-black/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-[#C2603F] transition-all duration-75 rounded-full"
                style={{ width: `${Math.min(100, Math.round(speakerRms * 300))}%` }}
              />
            </div>
          </div>
        </div>

        {/* Spoken Dialogue Transcript Box */}
        <div
          ref={transcriptBoxRef}
          aria-live="polite"
          className="w-full bg-[#FAF9F5] border border-[rgba(31,30,29,0.08)] rounded-2xl p-4 h-40 overflow-y-auto text-left text-[13px] text-[#1F1E1D] leading-relaxed shadow-inner space-y-2.5"
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
                  {turn.role === 'user' ? 'Caller' : agent.name}
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
            <Button variant="secondary" size="regular" onClick={onClose}>
              Dismiss
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
