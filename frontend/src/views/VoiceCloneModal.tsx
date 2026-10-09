import React, { useState, useRef, useEffect } from 'react'
import { api, ApiError } from '../lib/api'
import type { VoicePreset } from '../lib/types'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import {
  Mic,
  Square,
  Upload,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  X,
  AlertCircle,
  CheckCircle2,
  Volume2,
  Check,
} from 'lucide-react'

interface VoiceCloneModalProps {
  isOpen: boolean
  onClose: () => void
  onVoiceCloned: (voice: VoicePreset) => void
}

type UIState = 'idle' | 'recording' | 'uploading' | 'cloning' | 'ready' | 'error'

const ENROLLMENT_SCRIPT =
  'The quick brown fox jumps over the lazy dog. Today I am enrolling my custom voice for real-time conversational AI. I speak with natural cadence, clear articulation, and steady vocal tone.'

/**
 * Creates an uncompressed 24 kHz 16-bit mono RIFF WAV Blob from float32 samples.
 */
function floatTo16BitWav(samples: Float32Array, sampleRate: number = 24000): Blob {
  const numChannels = 1
  const bitsPerSample = 16
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8)
  const blockAlign = numChannels * (bitsPerSample / 8)
  const dataSize = samples.length * (bitsPerSample / 8)
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  // RIFF header
  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i))
    }
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // Audio format 1 = PCM
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitsPerSample, true)
  writeString(36, 'data')
  view.setUint32(40, dataSize, true)

  // Write PCM 16-bit samples
  let offset = 44
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

export const VoiceCloneModal: React.FC<VoiceCloneModalProps> = ({
  isOpen,
  onClose,
  onVoiceCloned,
}) => {
  const [activeTab, setActiveTab] = useState<'record' | 'upload'>('record')
  const [voiceName, setVoiceName] = useState('')
  const [gender, setGender] = useState<'male' | 'female' | 'unspecified'>('unspecified')
  const [hasConsent, setHasConsent] = useState(false)
  const [uiState, setUiState] = useState<UIState>('idle')
  const [recordSeconds, setRecordSeconds] = useState(0)
  const [micRms, setMicRms] = useState(0)
  const [peakRms, setPeakRms] = useState(0)
  const [hasClipping, setHasClipping] = useState(false)
  const [silenceRatio, setSilenceRatio] = useState(0)

  const [recordedWavBlob, setRecordedWavBlob] = useState<Blob | null>(null)
  const [previewAudioUrl, setPreviewAudioUrl] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isPlayingPreview, setIsPlayingPreview] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  // Device selection
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('')

  // Audio recording buffers
  const audioContextRef = useRef<AudioContext | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const collectedSamplesRef = useRef<number[]>([])
  const timerIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const animFrameRef = useRef<number | null>(null)
  const previewAudioRef = useRef<HTMLAudioElement | null>(null)

  // Enumerate input devices
  useEffect(() => {
    if (isOpen) {
      navigator.mediaDevices?.enumerateDevices().then((devices) => {
        const audioInputs = devices.filter((d) => d.kind === 'audioinput')
        setAudioDevices(audioInputs)
        if (audioInputs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(audioInputs[0].deviceId)
        }
      }).catch(() => {})
    }
  }, [isOpen])

  // Reset state on modal close
  useEffect(() => {
    if (!isOpen) {
      cleanupRecording()
      setRecordedWavBlob(null)
      if (previewAudioUrl) URL.revokeObjectURL(previewAudioUrl)
      setPreviewAudioUrl(null)
      setUploadedFile(null)
      setErrorMessage(null)
      setVoiceName('')
      setHasConsent(false)
      setIsPlayingPreview(false)
      setUiState('idle')
      setPeakRms(0)
      setHasClipping(false)
    }
  }, [isOpen])

  const cleanupRecording = () => {
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    if (scriptProcessorRef.current) {
      try {
        scriptProcessorRef.current.disconnect()
      } catch {}
      scriptProcessorRef.current = null
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop())
      micStreamRef.current = null
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try {
        audioContextRef.current.close()
      } catch {}
      audioContextRef.current = null
    }
    setMicRms(0)
  }

  const startRecording = async () => {
    setErrorMessage(null)
    setRecordedWavBlob(null)
    if (previewAudioUrl) {
      URL.revokeObjectURL(previewAudioUrl)
      setPreviewAudioUrl(null)
    }
    collectedSamplesRef.current = []
    setPeakRms(0)
    setHasClipping(false)

    try {
      const constraints: MediaStreamConstraints = {
        audio: {
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      micStreamRef.current = stream

      const AudioContextClass =
        window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const audioCtx = new AudioContextClass({ sampleRate: 24000 })
      audioContextRef.current = audioCtx
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume()
      }

      const source = audioCtx.createMediaStreamSource(stream)
      // Visualizer analyzer
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 512
      source.connect(analyser)

      // ScriptProcessorNode for sample collection & 24kHz downsampling
      const bufferSize = 4096
      const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1)
      scriptProcessorRef.current = processor

      let silentFrames = 0
      let totalFrames = 0
      let maxPeak = 0

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0)
        let sumSq = 0
        for (let i = 0; i < inputData.length; i++) {
          const sample = inputData[i]
          collectedSamplesRef.current.push(sample)
          sumSq += sample * sample
          const absVal = Math.abs(sample)
          if (absVal > maxPeak) maxPeak = absVal
          if (absVal >= 0.98) setHasClipping(true)
        }

        const rms = Math.sqrt(sumSq / inputData.length)
        totalFrames++
        if (rms < 0.01) silentFrames++

        setMicRms(rms)
        setPeakRms(maxPeak)
        if (totalFrames > 0) {
          setSilenceRatio(silentFrames / totalFrames)
        }
      }

      source.connect(processor)
      processor.connect(audioCtx.destination)

      setUiState('recording')
      setRecordSeconds(0)

      timerIntervalRef.current = setInterval(() => {
        setRecordSeconds((s) => {
          if (s >= 14) {
            // Optimal 15s window reached
            stopRecording()
            return 15
          }
          return s + 1
        })
      }, 1000)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Microphone access denied or unavailable.'
      setErrorMessage(msg)
      setUiState('error')
      cleanupRecording()
    }
  }

  const stopRecording = () => {
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)

    const rawSamples = new Float32Array(collectedSamplesRef.current)
    cleanupRecording()

    if (rawSamples.length < 24000 * 3) {
      setErrorMessage('Recording is too short (under 3 seconds). Please record at least 5–15 seconds of clear speech.')
      setUiState('error')
      return
    }

    // Build uncompressed 24 kHz 16-bit mono WAV Blob
    const wavBlob = floatTo16BitWav(rawSamples, 24000)
    setRecordedWavBlob(wavBlob)
    const url = URL.createObjectURL(wavBlob)
    setPreviewAudioUrl(url)
    setUiState('ready')
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setErrorMessage(null)

    const validExts = ['.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac']
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!validExts.includes(ext)) {
      setErrorMessage(`Unsupported file format. Please upload a WAV, MP3, M4A, or WebM audio file.`)
      setUiState('error')
      return
    }

    if (previewAudioUrl) URL.revokeObjectURL(previewAudioUrl)
    setUploadedFile(file)
    const url = URL.createObjectURL(file)
    setPreviewAudioUrl(url)
    if (!voiceName) {
      setVoiceName(file.name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' '))
    }
    setUiState('ready')
  }

  const togglePreviewAudio = () => {
    if (!previewAudioUrl) return

    if (!previewAudioRef.current) {
      previewAudioRef.current = new Audio(previewAudioUrl)
      previewAudioRef.current.onended = () => setIsPlayingPreview(false)
      previewAudioRef.current.onerror = () => setIsPlayingPreview(false)
    } else if (previewAudioRef.current.src !== previewAudioUrl) {
      previewAudioRef.current.src = previewAudioUrl
    }

    if (isPlayingPreview) {
      previewAudioRef.current.pause()
      setIsPlayingPreview(false)
    } else {
      previewAudioRef.current.play().then(() => {
        setIsPlayingPreview(true)
      }).catch((e) => {
        console.error('Preview playback error:', e)
        setIsPlayingPreview(false)
      })
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrorMessage(null)

    if (!voiceName.trim()) {
      setErrorMessage('Please give your cloned voice a name.')
      setUiState('error')
      return
    }

    if (!hasConsent) {
      setErrorMessage('You must verify explicit consent to clone this voice.')
      setUiState('error')
      return
    }

    const audioPayload = activeTab === 'record' ? recordedWavBlob : uploadedFile
    if (!audioPayload) {
      setErrorMessage(
        activeTab === 'record'
          ? 'Please record at least 5 to 15 seconds of speech first.'
          : 'Please select an audio file to upload.'
      )
      setUiState('error')
      return
    }

    setUiState('cloning')

    try {
      const formData = new FormData()
      const filename =
        activeTab === 'record'
          ? `${voiceName.toLowerCase().replace(/\s+/g, '_')}_ref.wav`
          : (uploadedFile?.name || 'custom_voice.wav')

      formData.append('file', audioPayload, filename)
      formData.append('audio', audioPayload, filename)
      formData.append('voice_name', voiceName.trim())
      formData.append('name', voiceName.trim())
      formData.append('consent', 'true')
      formData.append(
        'consent_statement',
        'I confirm that this is my own voice recording for AI conversational speech.'
      )
      if (gender !== 'unspecified') {
        formData.append('gender', gender)
      }

      // 1. Try documented POST /voice/enroll endpoint
      let enrolledVoice: VoicePreset | null = null
      try {
        const resp = await fetch('/voice/enroll', {
          method: 'POST',
          body: formData,
        })
        if (resp.ok) {
          const data = await resp.json()
          enrolledVoice = {
            id: data.voice_id || `${data.id}.wav`,
            name: `${data.name} (Cloned)`,
            gender: data.gender || 'custom',
            speaking_style: 'Cloned Neural Voice',
            accent: 'Custom Cloned Reference',
            recommended_for: 'Custom cloned voice conditioning',
            is_cloned: true,
            preview_url: data.preview_url || `/voice/${data.id}/preview`,
            duration_sec: data.duration_sec,
            qa_passed: data.qa_passed,
            qa_score: data.qa_score,
            recommended_engine: 'personaplex_s2s',
          }
        }
      } catch (enrollErr) {
        console.debug('POST /voice/enroll fallback to api.cloneVoice:', enrollErr)
      }

      // 2. Fallback to /v2/agents/voices/clone
      if (!enrolledVoice) {
        enrolledVoice = await api.cloneVoice(formData)
      }

      onVoiceCloned(enrolledVoice)
      onClose()
    } catch (err: unknown) {
      setUiState('error')
      if (err instanceof ApiError) {
        setErrorMessage(err.problem.detail || err.problem.title)
      } else if (err instanceof Error) {
        setErrorMessage(err.message)
      } else {
        setErrorMessage('Failed to clone voice. Please ensure clean speech and try again.')
      }
    }
  }

  if (!isOpen) return null

  const hasAudioReady = activeTab === 'record' ? !!recordedWavBlob : !!uploadedFile

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs">
      <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-2xl w-full max-w-xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-[rgba(31,30,29,0.08)] bg-[#FDFCFB]">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-[#FBEFEA] border border-[rgba(194,96,63,0.2)] flex items-center justify-center text-[#C2603F]">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-[17px] font-semibold text-[#1F1E1D]">Clone Your Voice</h2>
              <p className="text-[12px] text-[#6B6963]">
                Zero-shot neural conditioning for PersonaPlex S2S (24 kHz PCM)
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-[#6B6963] hover:text-[#1F1E1D] hover:bg-[#FAF9F5] rounded-lg transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-[rgba(31,30,29,0.08)] px-5 bg-white">
          <button
            type="button"
            onClick={() => {
              if (uiState !== 'recording') {
                setActiveTab('record')
                setErrorMessage(null)
              }
            }}
            className={`py-3 px-4 text-[13px] font-medium border-b-2 transition-colors cursor-pointer flex items-center gap-2 ${
              activeTab === 'record'
                ? 'border-[#C2603F] text-[#C2603F]'
                : 'border-transparent text-[#6B6963] hover:text-[#1F1E1D]'
            }`}
          >
            <Mic className="w-4 h-4" />
            Record Microphone (Recommended)
          </button>
          <button
            type="button"
            onClick={() => {
              if (uiState !== 'recording') {
                setActiveTab('upload')
                setErrorMessage(null)
              }
            }}
            className={`py-3 px-4 text-[13px] font-medium border-b-2 transition-colors cursor-pointer flex items-center gap-2 ${
              activeTab === 'upload'
                ? 'border-[#C2603F] text-[#C2603F]'
                : 'border-transparent text-[#6B6963] hover:text-[#1F1E1D]'
            }`}
          >
            <Upload className="w-4 h-4" />
            Upload WAV / Audio File
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-5 space-y-4 max-h-[75vh] overflow-y-auto">
          {errorMessage && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-[12px] flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="flex-1">{errorMessage}</div>
            </div>
          )}

          {/* RECORD TAB */}
          {activeTab === 'record' && (
            <div className="space-y-4">
              {/* Script to Read Aloud */}
              <div className="p-3.5 bg-[#FAF9F5] border border-[rgba(31,30,29,0.08)] rounded-xl space-y-1.5">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-[#C2603F] flex items-center gap-1.5">
                  <Volume2 className="w-3.5 h-3.5" />
                  Script to Read Aloud (10–15 Seconds)
                </span>
                <p className="text-[13px] text-[#1F1E1D] leading-relaxed italic bg-white p-2.5 rounded-lg border border-[rgba(31,30,29,0.06)] font-serif">
                  "{ENROLLMENT_SCRIPT}"
                </p>
                <p className="text-[11px] text-[#6B6963]">
                  Speak at a relaxed, natural pace into your microphone in a quiet room.
                </p>
              </div>

              {/* Device Selector */}
              {audioDevices.length > 1 && (
                <div className="space-y-1">
                  <label className="text-[11px] font-medium text-[#6B6963]">Microphone Device</label>
                  <select
                    value={selectedDeviceId}
                    onChange={(e) => setSelectedDeviceId(e.target.value)}
                    disabled={uiState === 'recording'}
                    className="w-full text-[12px] bg-white border border-[rgba(31,30,29,0.12)] rounded-lg px-2.5 py-1.5 text-[#1F1E1D]"
                  >
                    {audioDevices.map((d) => (
                      <option key={d.deviceId} value={d.deviceId}>
                        {d.label || `Microphone ${d.deviceId.slice(0, 5)}`}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Recording Box */}
              <div className="p-5 border border-dashed border-[rgba(31,30,29,0.18)] rounded-xl bg-[#FAF9F5] flex flex-col items-center justify-center space-y-3">
                {uiState === 'recording' ? (
                  <>
                    <div className="relative">
                      <div className="w-16 h-16 rounded-full bg-red-100 flex items-center justify-center text-red-600 animate-pulse">
                        <Mic className="w-8 h-8" />
                      </div>
                      <span className="absolute -top-1 -right-1 flex h-4 w-4">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-4 w-4 bg-red-500"></span>
                      </span>
                    </div>

                    <div className="text-center">
                      <div className="text-[20px] font-mono font-bold text-[#1F1E1D]">
                        00:{recordSeconds < 10 ? `0${recordSeconds}` : recordSeconds} / 00:15
                      </div>
                      <span className="text-[12px] text-[#6B6963]">Recording 24 kHz mono PCM...</span>
                    </div>

                    {/* Live RMS Meter */}
                    <div className="w-48 space-y-1">
                      <div className="w-full h-2.5 bg-gray-200 rounded-full overflow-hidden flex">
                        <div
                          className={`h-full transition-all duration-75 ${
                            hasClipping ? 'bg-red-500' : 'bg-emerald-500'
                          }`}
                          style={{ width: `${Math.min(100, Math.round(micRms * 250))}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-[10px] text-[#6B6963]">
                        <span>Level: {(micRms * 100).toFixed(0)}%</span>
                        {hasClipping && <span className="text-red-600 font-semibold">Clipping!</span>}
                      </div>
                    </div>

                    <Button
                      type="button"
                      variant="destructive"
                      onClick={stopRecording}
                      className="flex items-center gap-2 mt-2 px-6"
                    >
                      <Square className="w-4 h-4 fill-current" />
                      Stop Recording
                    </Button>
                  </>
                ) : recordedWavBlob ? (
                  <>
                    <div className="w-14 h-14 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600">
                      <CheckCircle2 className="w-8 h-8" />
                    </div>

                    <div className="text-center">
                      <div className="text-[14px] font-semibold text-[#1F1E1D]">
                        Recording Complete ({recordSeconds}s)
                      </div>
                      <div className="text-[11px] text-emerald-600 font-medium">
                        Encoded to 24 kHz 16-bit Mono WAV ({Math.round(recordedWavBlob.size / 1024)} KB)
                      </div>
                    </div>

                    {/* Quality Check Badges */}
                    <div className="flex flex-wrap gap-2 justify-center text-[11px]">
                      <span className="px-2 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 flex items-center gap-1">
                        <Check className="w-3 h-3" />
                        Duration: {recordSeconds}s (Optimal)
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded-full border flex items-center gap-1 ${
                          hasClipping
                            ? 'bg-amber-50 border-amber-200 text-amber-700'
                            : 'bg-emerald-50 border-emerald-200 text-emerald-700'
                        }`}
                      >
                        <Check className="w-3 h-3" />
                        {hasClipping ? 'Minor Peak' : `Peak Level: ${(peakRms * 100).toFixed(0)}%`}
                      </span>
                      <span className="px-2 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 flex items-center gap-1">
                        <Check className="w-3 h-3" />
                        Speech Ratio: {Math.round((1 - silenceRatio) * 100)}%
                      </span>
                    </div>

                    {/* Controls */}
                    <div className="flex items-center gap-2 pt-1">
                      <Button
                        type="button"
                        variant="secondary"
                        size="compact"
                        onClick={togglePreviewAudio}
                        className="flex items-center gap-1.5"
                      >
                        {isPlayingPreview ? (
                          <>
                            <Pause className="w-3.5 h-3.5" />
                            Pause Preview
                          </>
                        ) : (
                          <>
                            <Play className="w-3.5 h-3.5" />
                            Listen Preview
                          </>
                        )}
                      </Button>
                      <Button
                        type="button"
                        variant="quiet"
                        size="compact"
                        onClick={startRecording}
                        className="flex items-center gap-1.5 text-[#6B6963]"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        Re-record
                      </Button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-14 h-14 rounded-full bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] flex items-center justify-center text-[#6B6963]">
                      <Mic className="w-7 h-7" />
                    </div>
                    <div className="text-center">
                      <div className="text-[14px] font-semibold text-[#1F1E1D]">Click to Record</div>
                      <div className="text-[12px] text-[#6B6963]">
                        Read the script aloud for 10 to 15 seconds.
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="primary"
                      onClick={startRecording}
                      className="flex items-center gap-2 px-6"
                    >
                      <Mic className="w-4 h-4" />
                      Start 15s Recording
                    </Button>
                  </>
                )}
              </div>
            </div>
          )}

          {/* UPLOAD TAB */}
          {activeTab === 'upload' && (
            <div className="space-y-3">
              <div className="p-6 border-2 border-dashed border-[rgba(31,30,29,0.18)] rounded-xl bg-[#FAF9F5] flex flex-col items-center justify-center space-y-3 text-center">
                <Upload className="w-8 h-8 text-[#C2603F]" />
                <div>
                  <div className="text-[13px] font-medium text-[#1F1E1D]">
                    {uploadedFile ? uploadedFile.name : 'Select or drag audio file'}
                  </div>
                  <div className="text-[11px] text-[#6B6963]">
                    WAV, MP3, M4A, or WebM (3 to 60 seconds recommended)
                  </div>
                </div>
                <input
                  type="file"
                  id="voice-upload"
                  accept=".wav,.mp3,.m4a,.webm,.ogg,.flac"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <Button
                  type="button"
                  variant="secondary"
                  size="compact"
                  onClick={() => document.getElementById('voice-upload')?.click()}
                >
                  Choose File
                </Button>
              </div>

              {uploadedFile && (
                <div className="flex items-center justify-between p-3 bg-white border border-[rgba(31,30,29,0.08)] rounded-xl">
                  <div className="flex items-center gap-2 truncate">
                    <Volume2 className="w-4 h-4 text-[#C2603F] shrink-0" />
                    <span className="text-[12px] font-medium truncate">{uploadedFile.name}</span>
                  </div>
                  <Button
                    type="button"
                    variant="quiet"
                    size="compact"
                    onClick={togglePreviewAudio}
                    className="flex items-center gap-1"
                  >
                    {isPlayingPreview ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                    {isPlayingPreview ? 'Pause' : 'Play'}
                  </Button>
                </div>
              )}
            </div>
          )}

          {/* Voice Details */}
          <div className="space-y-3 pt-2">
            <div>
              <label className="block text-[12px] font-medium text-[#1F1E1D] mb-1">
                Voice Name <span className="text-red-500">*</span>
              </label>
              <Input
                placeholder="e.g. Kishore Kumar, Alex Sales"
                value={voiceName}
                onChange={(e) => setVoiceName(e.target.value)}
                disabled={uiState === 'cloning'}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-[12px] font-medium text-[#1F1E1D] mb-1">
                Voice Gender Hint (Optional)
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(['unspecified', 'male', 'female'] as const).map((g) => (
                  <button
                    key={g}
                    type="button"
                    onClick={() => setGender(g)}
                    disabled={uiState === 'cloning'}
                    className={`py-2 px-3 rounded-lg border text-[12px] font-medium capitalize transition-all cursor-pointer ${
                      gender === g
                        ? 'border-[#C2603F] bg-[#FBEFEA] text-[#C2603F]'
                        : 'border-[rgba(31,30,29,0.12)] text-[#6B6963] hover:border-[rgba(31,30,29,0.2)]'
                    }`}
                  >
                    {g}
                  </button>
                ))}
              </div>
            </div>

            {/* Consent Checkbox */}
            <div className="p-3 bg-[#FAF9F5] border border-[rgba(31,30,29,0.08)] rounded-xl flex items-start gap-2.5">
              <input
                type="checkbox"
                id="voice-consent"
                checked={hasConsent}
                onChange={(e) => setHasConsent(e.target.checked)}
                disabled={uiState === 'cloning'}
                className="mt-0.5 rounded border-[rgba(31,30,29,0.2)] text-[#C2603F] focus:ring-[#C2603F] cursor-pointer"
              />
              <label htmlFor="voice-consent" className="text-[11px] text-[#6B6963] leading-relaxed cursor-pointer">
                <span className="font-semibold text-[#1F1E1D]">I verify explicit consent: </span>
                I confirm that this audio recording is my own voice, or I have authorization to clone and use this voice reference for AI conversational speech.
              </label>
            </div>
          </div>

          {/* Footer Actions */}
          <div className="flex items-center justify-end gap-2 pt-3 border-t border-[rgba(31,30,29,0.08)]">
            <Button type="button" variant="quiet" onClick={onClose} disabled={uiState === 'cloning'}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              disabled={!hasAudioReady || !voiceName.trim() || !hasConsent || uiState === 'cloning'}
              className="flex items-center gap-2"
            >
              {uiState === 'cloning' ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Conditioning Voice...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Save & Enroll Voice
                </>
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
