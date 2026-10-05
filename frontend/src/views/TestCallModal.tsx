import React, { useState, useEffect, useRef } from 'react'
import type { Agent } from '../lib/types'
import { AudioIndicator } from '../components/ui/AudioIndicator'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Mic, MicOff, PhoneOff, AlertCircle } from 'lucide-react'

export interface TestCallModalProps {
  agent: Agent
  onClose: () => void
}

export const TestCallModal: React.FC<TestCallModalProps> = ({ agent, onClose }) => {
  const [callStatus, setCallStatus] = useState<'connecting' | 'connected' | 'ended' | 'error'>('connecting')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isMuted, setIsMuted] = useState(false)
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false)
  const [rms, setRms] = useState(0.0)
  const [elapsedSec, setElapsedSec] = useState(0)
  const [transcriptTokens, setTranscriptTokens] = useState<string[]>([])
  const [endReason, setEndReason] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const nextPlayTimeRef = useRef<number>(0)

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

  // 2. Initialize WebSocket & Web Audio Streaming
  useEffect(() => {
    let isCleanedUp = false

    async function startCall() {
      try {
        // Request Microphone Access
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        })
        mediaStreamRef.current = stream

        // Initialize AudioContext at 16000 Hz
        const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
        const audioCtx = new AudioContextClass({ sampleRate: 16000 })
        audioContextRef.current = audioCtx

        // Build WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const host = window.location.host
        const wsUrl = `${protocol}//${host}/v2/voice?agent_id=${encodeURIComponent(agent.id)}&sample_rate=16000&codec=pcm16`

        const ws = new WebSocket(wsUrl)
        wsRef.current = ws
        ws.binaryType = 'arraybuffer'

        ws.onopen = () => {
          if (isCleanedUp) return
          setCallStatus('connected')
        }

        ws.onmessage = async (event) => {
          if (isCleanedUp) return

          if (typeof event.data === 'string') {
            try {
              const msg = JSON.parse(event.data)
              if (msg.type === 'transcript') {
                setIsAgentSpeaking(true)
                setTranscriptTokens((prev) => [...prev, msg.text])
                // Reset speaking flag after token pause
                setTimeout(() => setIsAgentSpeaking(false), 800)
              } else if (msg.type === 'call_ended') {
                setEndReason(msg.reason || 'Call concluded')
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
            // Play received 16 kHz PCM16 audio
            playPcm16Chunk(event.data)
          }
        }

        ws.onerror = (e) => {
          console.error('WebSocket error', e)
          if (!isCleanedUp) {
            setErrorMessage('Could not establish voice connection with worker.')
            setCallStatus('error')
          }
        }

        ws.onclose = () => {
          if (!isCleanedUp) {
            setCallStatus((curr) => (curr === 'connected' ? 'ended' : curr))
          }
        }

        // Set up Microphone capture and PCM16 streaming
        const source = audioCtx.createMediaStreamSource(stream)
        const processor = audioCtx.createScriptProcessor(1024, 1, 1)
        processorRef.current = processor

        processor.onaudioprocess = (e) => {
          if (isMuted || ws.readyState !== WebSocket.OPEN) return

          const inputData = e.inputBuffer.getChannelData(0)
          // Compute RMS for local feedback
          let sum = 0
          for (let i = 0; i < inputData.length; i++) {
            sum += inputData[i] * inputData[i]
          }
          const level = Math.sqrt(sum / inputData.length)
          setRms((prev) => prev * 0.7 + level * 0.3)

          // Convert Float32 to Int16 PCM bytes
          const pcm16 = new Int16Array(inputData.length)
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]))
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
          }
          ws.send(pcm16.buffer)
        }

        source.connect(processor)
        processor.connect(audioCtx.destination)
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

  // Helper to play raw PCM16 chunk smoothly
  const playPcm16Chunk = (buffer: ArrayBuffer) => {
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
    setRms((prev) => prev * 0.5 + level * 0.5)

    const audioBuffer = audioCtx.createBuffer(1, float32.length, 16000)
    audioBuffer.copyToChannel(float32, 0)

    const source = audioCtx.createBufferSource()
    source.buffer = audioBuffer
    source.connect(audioCtx.destination)

    const now = audioCtx.currentTime
    const startAt = Math.max(now, nextPlayTimeRef.current)
    source.start(startAt)
    nextPlayTimeRef.current = startAt + audioBuffer.duration
  }

  const cleanupAudio = () => {
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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs select-none">
      <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-3xl max-w-lg w-full p-8 shadow-xl space-y-6 flex flex-col items-center text-center">
        {/* Call Status & Agent Header */}
        <div className="space-y-1">
          <div className="flex items-center justify-center gap-2">
            <Badge
              variant={callStatus === 'connected' ? 'connected' : callStatus === 'ended' ? 'neutral' : 'warning'}
              dot
            >
              {callStatus === 'connecting'
                ? 'Connecting Line...'
                : callStatus === 'connected'
                ? 'Call in Progress'
                : callStatus === 'ended'
                ? 'Call Concluded'
                : 'Connection Failed'}
            </Badge>
          </div>
          <h2 className="text-[22px] font-semibold text-[#1F1E1D] pt-1">
            {agent.name}
          </h2>
          <div className="text-[13px] text-[#6B6963] font-mono tabular-nums">
            {callStatus === 'connected' ? formatTimer(elapsedSec) : agent.voice_id}
          </div>
        </div>

        {/* Audio Circle Indicator (Apple / Claude Aesthetic) */}
        <div className="py-4">
          <AudioIndicator
            rms={rms}
            isSpeaking={isAgentSpeaking}
            isListening={callStatus === 'connected' && !isMuted}
            size={90}
          />
        </div>

        {/* Spoken Captions Box */}
        <div
          aria-live="polite"
          className="w-full bg-[#FAF9F5] border border-[rgba(31,30,29,0.08)] rounded-xl p-4 min-h-[90px] max-h-36 overflow-y-auto text-left text-[14px] text-[#1F1E1D] leading-relaxed shadow-2xs"
        >
          {errorMessage ? (
            <div className="flex items-start gap-2 text-[#A63A38] text-[13px]">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{errorMessage}</span>
            </div>
          ) : endReason ? (
            <div className="text-[13px] text-[#6B6963] italic text-center py-4">
              {endReason}
            </div>
          ) : transcriptTokens.length > 0 ? (
            <span>{transcriptTokens.join('')}</span>
          ) : (
            <span className="text-[#9E9B93] text-[13px] italic">
              {callStatus === 'connecting'
                ? 'Establishing speech pipeline...'
                : isMuted
                ? 'Microphone muted. Unmute to speak.'
                : 'Speak naturally into your microphone...'}
            </span>
          )}
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-4 pt-2">
          {callStatus === 'connected' && (
            <Button
              variant={isMuted ? 'destructive' : 'secondary'}
              size="regular"
              leftIcon={isMuted ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              onClick={() => setIsMuted(!isMuted)}
            >
              {isMuted ? 'Unmute' : 'Mute'}
            </Button>
          )}

          {callStatus === 'connected' ? (
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
