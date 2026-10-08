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
} from 'lucide-react'

interface VoiceCloneModalProps {
  isOpen: boolean
  onClose: () => void
  onVoiceCloned: (voice: VoicePreset) => void
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
  const [isRecording, setIsRecording] = useState(false)
  const [recordSeconds, setRecordSeconds] = useState(0)
  const [micRms, setMicRms] = useState(0)
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState<string | null>(null)
  const [isPlayingPreview, setIsPlayingPreview] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const timerIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const animFrameRef = useRef<number | null>(null)
  const previewAudioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    if (!isOpen) {
      cleanupRecording()
      setRecordedBlob(null)
      if (recordedUrl) URL.revokeObjectURL(recordedUrl)
      setRecordedUrl(null)
      setUploadedFile(null)
      if (uploadedPreviewUrl) URL.revokeObjectURL(uploadedPreviewUrl)
      setUploadedPreviewUrl(null)
      setErrorMsg(null)
      setVoiceName('')
      setHasConsent(false)
      setIsPlayingPreview(false)
    }
  }, [isOpen])

  const cleanupRecording = () => {
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop()
      } catch {}
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
    setIsRecording(false)
    setMicRms(0)
  }

  const startRecording = async () => {
    setErrorMsg(null)
    setRecordedBlob(null)
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl)
      setRecordedUrl(null)
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      micStreamRef.current = stream

      // Set up live RMS visualizer
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
      audioContextRef.current = audioCtx
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 512
      source.connect(analyser)

      const pcmData = new Uint8Array(analyser.frequencyBinCount)
      const updateMeter = () => {
        if (!micStreamRef.current) return
        analyser.getByteFrequencyData(pcmData)
        let sum = 0
        for (let i = 0; i < pcmData.length; i++) {
          sum += pcmData[i]
        }
        const avg = sum / pcmData.length / 255.0
        setMicRms(avg)
        animFrameRef.current = requestAnimationFrame(updateMeter)
      }
      updateMeter()

      // Choose supported mimeType
      const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
      ]
      let selectedMime = ''
      for (const m of mimeTypes) {
        if (MediaRecorder.isTypeSupported(m)) {
          selectedMime = m
          break
        }
      }

      const recorder = selectedMime
        ? new MediaRecorder(stream, { mimeType: selectedMime })
        : new MediaRecorder(stream)

      const chunks: Blob[] = []
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunks.push(e.data)
        }
      }

      recorder.onstop = () => {
        const fullBlob = new Blob(chunks, { type: selectedMime || 'audio/webm' })
        setRecordedBlob(fullBlob)
        const url = URL.createObjectURL(fullBlob)
        setRecordedUrl(url)
      }

      mediaRecorderRef.current = recorder
      recorder.start(100) // 100ms slices
      setIsRecording(true)
      setRecordSeconds(0)

      timerIntervalRef.current = setInterval(() => {
        setRecordSeconds((s) => {
          if (s >= 14) {
            // Auto stop at 15s to keep within optimal 5-12s window
            stopRecording()
            return 15
          }
          return s + 1
        })
      }, 1000)
    } catch (err: any) {
      setErrorMsg(err.message || 'Microphone access denied or unavailable.')
      cleanupRecording()
    }
  }

  const stopRecording = () => {
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop())
      micStreamRef.current = null
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {})
      audioContextRef.current = null
    }
    setIsRecording(false)
    setMicRms(0)
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setErrorMsg(null)

    const validExts = ['.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac']
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!validExts.includes(ext)) {
      setErrorMsg(`Unsupported file type. Please upload a WAV, MP3, M4A, or WebM audio file.`)
      return
    }

    if (uploadedPreviewUrl) URL.revokeObjectURL(uploadedPreviewUrl)
    setUploadedFile(file)
    const url = URL.createObjectURL(file)
    setUploadedPreviewUrl(url)
    if (!voiceName) {
      setVoiceName(file.name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' '))
    }
  }

  const togglePreviewAudio = () => {
    const url = activeTab === 'record' ? recordedUrl : uploadedPreviewUrl
    if (!url) return

    if (!previewAudioRef.current) {
      previewAudioRef.current = new Audio(url)
      previewAudioRef.current.onended = () => setIsPlayingPreview(false)
      previewAudioRef.current.onerror = () => setIsPlayingPreview(false)
    } else if (previewAudioRef.current.src !== url) {
      previewAudioRef.current.src = url
    }

    if (isPlayingPreview) {
      previewAudioRef.current.pause()
      setIsPlayingPreview(false)
    } else {
      previewAudioRef.current.play().then(() => {
        setIsPlayingPreview(true)
      }).catch((e) => {
        console.error('Audio preview play error:', e)
        setIsPlayingPreview(false)
      })
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrorMsg(null)

    if (!voiceName.trim()) {
      setErrorMsg('Please give your cloned voice a name.')
      return
    }

    if (!hasConsent) {
      setErrorMsg('You must provide explicit consent to clone this voice.')
      return
    }

    const audioPayload = activeTab === 'record' ? recordedBlob : uploadedFile
    if (!audioPayload) {
      setErrorMsg(
        activeTab === 'record'
          ? 'Please record at least 4–8 seconds of clear speech first.'
          : 'Please select an audio file to upload.'
      )
      return
    }

    if (activeTab === 'record' && recordSeconds < 3) {
      setErrorMsg('Recording is too short. Please speak continuously for at least 4 to 10 seconds.')
      return
    }

    setIsSubmitting(true)

    try {
      const formData = new FormData()
      const filename =
        activeTab === 'record'
          ? `${voiceName.toLowerCase().replace(/\s+/g, '_')}_record.webm`
          : (uploadedFile?.name || 'custom_voice.wav')

      formData.append('file', audioPayload, filename)
      formData.append('name', voiceName.trim())
      formData.append('consent', 'true')
      if (gender !== 'unspecified') {
        formData.append('gender', gender)
      }

      const newVoice = await api.cloneVoice(formData)
      onVoiceCloned(newVoice)
      onClose()
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setErrorMsg(err.problem.detail || err.problem.title)
      } else if (err instanceof Error) {
        setErrorMsg(err.message)
      } else {
        setErrorMsg('Failed to clone voice. Please ensure clear speech and retry.')
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  if (!isOpen) return null

  const hasAudioReady = activeTab === 'record' ? !!recordedBlob : !!uploadedFile

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs">
      <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="p-6 border-b border-[rgba(31,30,29,0.08)] bg-[#FAF9F5] flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-[#FBEFEA] text-[#C2603F]">
                <Sparkles className="w-4 h-4" />
              </span>
              <h2 className="text-[17px] font-semibold text-[#1F1E1D]">Clone Your Voice</h2>
            </div>
            <p className="text-[12px] text-[#6B6963]">
              Record 5–10s of audio or upload a sample. The neural model will normalize loudness to -24 LUFS and condition the voice instantly.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-[#9E9B93] hover:text-[#1F1E1D] p-1 rounded-md transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Error Banner */}
          {errorMsg && (
            <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-xl text-[12px] flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-red-600" />
              <div className="flex-1">{errorMsg}</div>
            </div>
          )}

          {/* Tab Selector */}
          <div className="flex p-1 bg-[#FAF9F5] border border-[rgba(31,30,29,0.08)] rounded-xl">
            <button
              type="button"
              onClick={() => {
                setActiveTab('record')
                setErrorMsg(null)
              }}
              className={`flex-1 py-1.5 text-[12px] font-medium rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                activeTab === 'record'
                  ? 'bg-white text-[#1F1E1D] shadow-xs'
                  : 'text-[#6B6963] hover:text-[#1F1E1D]'
              }`}
            >
              <Mic className="w-3.5 h-3.5" /> Record Microphone
            </button>
            <button
              type="button"
              onClick={() => {
                setActiveTab('upload')
                setErrorMsg(null)
              }}
              className={`flex-1 py-1.5 text-[12px] font-medium rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                activeTab === 'upload'
                  ? 'bg-white text-[#1F1E1D] shadow-xs'
                  : 'text-[#6B6963] hover:text-[#1F1E1D]'
              }`}
            >
              <Upload className="w-3.5 h-3.5" /> Upload Audio File
            </button>
          </div>

          {/* Tab 1: Live Record */}
          {activeTab === 'record' && (
            <div className="p-4 border border-[rgba(31,30,29,0.08)] rounded-xl bg-[#FAF9F5]/60 space-y-4">
              <div className="text-center space-y-1">
                <span className="text-[12px] font-medium text-[#1F1E1D]">
                  {isRecording
                    ? `Recording: ${recordSeconds}s (Speak naturally for 5–10s)`
                    : recordedBlob
                    ? `Recorded sample ready (${recordSeconds}s)`
                    : 'Click Record and speak clearly into your mic'}
                </span>
                <p className="text-[11px] text-[#9E9B93]">
                  Read any sentence, for example: "Hello, I am testing my custom voice for my AI agent."
                </p>
              </div>

              {/* RMS Audio Level Bar */}
              {isRecording && (
                <div className="space-y-1">
                  <div className="h-2 w-full bg-[#E5E3DC] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#C2603F] transition-all duration-75"
                      style={{ width: `${Math.min(100, micRms * 300)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-[#9E9B93]">
                    <span>Quiet</span>
                    <span>Ideal Volume</span>
                    <span>Loud</span>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex items-center justify-center gap-3">
                {!isRecording ? (
                  <Button
                    type="button"
                    variant={recordedBlob ? 'secondary' : 'primary'}
                    size="compact"
                    onClick={startRecording}
                    className="flex items-center gap-1.5"
                  >
                    {recordedBlob ? <RotateCcw className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5" />}
                    {recordedBlob ? 'Re-record' : 'Start Recording'}
                  </Button>
                ) : (
                  <Button
                    type="button"
                    variant="destructive"
                    size="compact"
                    onClick={stopRecording}
                    className="flex items-center gap-1.5"
                  >
                    <Square className="w-3.5 h-3.5 fill-current" /> Stop Recording ({recordSeconds}s)
                  </Button>
                )}

                {recordedBlob && !isRecording && (
                  <Button
                    type="button"
                    variant="secondary"
                    size="compact"
                    onClick={togglePreviewAudio}
                    className="flex items-center gap-1.5"
                  >
                    {isPlayingPreview ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                    {isPlayingPreview ? 'Pause' : 'Listen Preview'}
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* Tab 2: Upload File */}
          {activeTab === 'upload' && (
            <div className="p-4 border-2 border-dashed border-[rgba(31,30,29,0.15)] rounded-xl bg-[#FAF9F5]/40 text-center space-y-3">
              <input
                type="file"
                id="voice-file-input"
                accept=".wav,.mp3,.m4a,.webm,.ogg,.flac"
                onChange={handleFileUpload}
                className="hidden"
              />
              <label
                htmlFor="voice-file-input"
                className="cursor-pointer inline-flex flex-col items-center justify-center space-y-2 p-3 w-full"
              >
                <div className="p-2 rounded-full bg-[#FAF9F5] border border-[rgba(31,30,29,0.08)] text-[#6B6963]">
                  <Upload className="w-5 h-5 text-[#C2603F]" />
                </div>
                <div>
                  <span className="text-[13px] font-medium text-[#1F1E1D] hover:underline">
                    {uploadedFile ? uploadedFile.name : 'Choose audio file or drag here'}
                  </span>
                  <p className="text-[11px] text-[#9E9B93] mt-0.5">
                    WAV, MP3, M4A, or WebM (5 to 15 seconds recommended)
                  </p>
                </div>
              </label>

              {uploadedFile && (
                <div className="flex items-center justify-center gap-2 pt-1 border-t border-[rgba(31,30,29,0.06)]">
                  <Button
                    type="button"
                    variant="secondary"
                    size="compact"
                    onClick={togglePreviewAudio}
                    className="flex items-center gap-1.5 text-[11px]"
                  >
                    {isPlayingPreview ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                    {isPlayingPreview ? 'Pause' : 'Play Upload'}
                  </Button>
                </div>
              )}
            </div>
          )}

          {/* Voice Details */}
          <div className="space-y-4 pt-1 border-t border-[rgba(31,30,29,0.08)]">
            <Input
              label="Voice Name"
              hint="e.g. My Voice, Aarav Real, Elena Custom"
              value={voiceName}
              onChange={(e) => setVoiceName(e.target.value)}
              placeholder="e.g. My Voice"
              maxLength={40}
            />

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-[12px] font-medium text-[#1F1E1D]">Vocal Tone / Gender</label>
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value as any)}
                  className="w-full text-[12px] p-2 rounded-lg border border-[rgba(31,30,29,0.12)] bg-[#FAF9F5] text-[#1F1E1D] focus:outline-none focus:border-[#C2603F]"
                >
                  <option value="unspecified">Auto-Detect</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[12px] font-medium text-[#1F1E1D]">Conditioning Target</label>
                <div className="text-[12px] p-2 rounded-lg border border-[rgba(31,30,29,0.06)] bg-[#FAF9F5] text-[#6B6963] flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  24 kHz Mono • -24 LUFS
                </div>
              </div>
            </div>

            {/* Explicit Consent Checkbox */}
            <div className="p-3 bg-[#FAF9F5] border border-[rgba(31,30,29,0.08)] rounded-xl flex items-start gap-2.5">
              <input
                type="checkbox"
                id="clone-consent"
                checked={hasConsent}
                onChange={(e) => setHasConsent(e.target.checked)}
                className="mt-0.5 accent-[#C2603F] cursor-pointer"
              />
              <label htmlFor="clone-consent" className="text-[11px] text-[#6B6963] leading-relaxed cursor-pointer">
                <strong className="text-[#1F1E1D]">Explicit Consent:</strong> I confirm that this is my own voice recording, or I have received explicit permission to use and clone this voice for AI conversational speech.
              </label>
            </div>
          </div>

          {/* Footer Actions */}
          <div className="flex items-center justify-end gap-2.5 pt-3 border-t border-[rgba(31,30,29,0.08)]">
            <Button type="button" variant="secondary" size="compact" onClick={onClose} disabled={isSubmitting}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="compact"
              disabled={!hasAudioReady || !voiceName.trim() || !hasConsent || isSubmitting}
              className="flex items-center gap-1.5"
            >
              <Sparkles className="w-3.5 h-3.5" />
              {isSubmitting ? 'Normalizing & Cloning...' : 'Clone & Apply Voice'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
