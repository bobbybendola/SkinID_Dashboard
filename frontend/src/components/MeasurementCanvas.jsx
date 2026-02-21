import { useState, useRef, useEffect, useCallback } from 'react'

export default function MeasurementCanvas({ imageSrc, onComplete, onBack }) {
  const canvasRef = useRef(null)
  const imgRef = useRef(null)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [mode, setMode] = useState('width') // 'width' | 'height'
  const [widthLine, setWidthLine] = useState(null)  // { start, end }
  const [heightLine, setHeightLine] = useState(null)
  const [drawing, setDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState(null)
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 })
  const [imageNaturalSize, setImageNaturalSize] = useState({ w: 0, h: 0 })
  const [pxPerMm, setPxPerMm] = useState(null)
  const [showCalibration, setShowCalibration] = useState(true)
  const [calibrationValue, setCalibrationValue] = useState('10')

  const getCanvasCoords = useCallback((e) => {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    }
  }, [])

  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      setImageNaturalSize({ w: img.naturalWidth, h: img.naturalHeight })

      // Fit into container (max 700px wide, maintain aspect ratio)
      const maxW = Math.min(700, window.innerWidth - 64)
      const scale = maxW / img.naturalWidth
      const dispW = Math.round(img.naturalWidth * scale)
      const dispH = Math.round(img.naturalHeight * scale)
      setCanvasSize({ w: dispW, h: dispH })
      setImgLoaded(true)
    }
    img.src = imageSrc
  }, [imageSrc])

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    const img = imgRef.current
    if (!ctx || !img) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    const drawLine = (line, color, label) => {
      if (!line) return
      ctx.beginPath()
      ctx.moveTo(line.start.x, line.start.y)
      ctx.lineTo(line.end.x, line.end.y)
      ctx.strokeStyle = color
      ctx.lineWidth = 2.5
      ctx.setLineDash([])
      ctx.stroke()

      // Draw endpoints
      for (const pt of [line.start, line.end]) {
        ctx.beginPath()
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
        ctx.strokeStyle = 'white'
        ctx.lineWidth = 1.5
        ctx.stroke()
      }

      // Draw label
      const midX = (line.start.x + line.end.x) / 2
      const midY = (line.start.y + line.end.y) / 2
      const dx = line.end.x - line.start.x
      const dy = line.end.y - line.start.y
      const pxLen = Math.sqrt(dx * dx + dy * dy)
      let displayLen = `${Math.round(pxLen)}px`
      if (pxPerMm) {
        displayLen = `${(pxLen / pxPerMm).toFixed(1)} mm`
      }

      ctx.font = 'bold 13px Inter, system-ui, sans-serif'
      const text = `${label}: ${displayLen}`
      const tm = ctx.measureText(text)
      const pad = 6
      ctx.fillStyle = 'rgba(0,0,0,0.7)'
      ctx.roundRect(midX - tm.width / 2 - pad, midY - 20, tm.width + pad * 2, 24, 4)
      ctx.fill()
      ctx.fillStyle = 'white'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(text, midX, midY - 8)
    }

    drawLine(widthLine, '#a78bfa', 'Width')
    drawLine(heightLine, '#34d399', 'Height')

    // Draw current line being drawn
    if (drawing && startPoint) {
      ctx.beginPath()
      ctx.arc(startPoint.x, startPoint.y, 5, 0, Math.PI * 2)
      ctx.fillStyle = mode === 'width' ? '#a78bfa' : '#34d399'
      ctx.fill()
    }
  }, [widthLine, heightLine, drawing, startPoint, mode, pxPerMm])

  useEffect(() => {
    if (imgLoaded) drawCanvas()
  }, [imgLoaded, drawCanvas])

  const handleMouseDown = (e) => {
    const coords = getCanvasCoords(e)
    if (!coords) return
    setDrawing(true)
    setStartPoint(coords)
  }

  const handleMouseMove = (e) => {
    if (!drawing || !startPoint) return
    const coords = getCanvasCoords(e)
    if (!coords) return

    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!ctx) return

    drawCanvas()

    // Draw temporary line
    ctx.beginPath()
    ctx.moveTo(startPoint.x, startPoint.y)
    ctx.lineTo(coords.x, coords.y)
    ctx.strokeStyle = mode === 'width' ? '#a78bfa' : '#34d399'
    ctx.lineWidth = 2
    ctx.setLineDash([6, 4])
    ctx.stroke()
    ctx.setLineDash([])

    // Endpoint
    ctx.beginPath()
    ctx.arc(coords.x, coords.y, 5, 0, Math.PI * 2)
    ctx.fillStyle = mode === 'width' ? '#a78bfa' : '#34d399'
    ctx.fill()
  }

  const handleMouseUp = (e) => {
    if (!drawing || !startPoint) return
    const coords = getCanvasCoords(e)
    if (!coords) return

    const line = { start: startPoint, end: coords }
    if (mode === 'width') {
      setWidthLine(line)
      setMode('height')
    } else {
      setHeightLine(line)
    }

    setDrawing(false)
    setStartPoint(null)
  }

  const getPixelLength = (line) => {
    if (!line) return 0
    const dx = line.end.x - line.start.x
    const dy = line.end.y - line.start.y
    return Math.sqrt(dx * dx + dy * dy)
  }

  const getMmLength = (line) => {
    const px = getPixelLength(line)
    if (!pxPerMm || px === 0) return null
    return (px / pxPerMm).toFixed(1)
  }

  const handleCalibrationSubmit = () => {
    const val = parseFloat(calibrationValue)
    if (!val || val <= 0) return
    // Use the width line for calibration if available
    if (widthLine) {
      const px = getPixelLength(widthLine)
      if (px > 0) {
        setPxPerMm(px / val)
      }
    }
    setShowCalibration(false)
  }

  const handleComplete = () => {
    const scaleX = imageNaturalSize.w / canvasSize.w
    const scaleY = imageNaturalSize.h / canvasSize.h

    const widthPx = getPixelLength(widthLine)
    const heightPx = getPixelLength(heightLine)

    const result = {
      width_px: Math.round(widthPx * scaleX),
      height_px: Math.round(heightPx * scaleY),
      width_mm: getMmLength(widthLine),
      height_mm: getMmLength(heightLine),
      image_width: imageNaturalSize.w,
      image_height: imageNaturalSize.h,
    }

    onComplete(result)
  }

  const canProceed = widthLine && heightLine

  return (
    <div className="space-y-4">
      <div className="glass rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">Measure the Lesion</h3>
            <p className="text-sm text-white/50 mt-1">
              {mode === 'width'
                ? 'Click and drag to draw the WIDTH of the lesion'
                : !canProceed
                  ? 'Now draw the HEIGHT of the lesion'
                  : 'Measurement complete'
              }
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${mode === 'width' && !widthLine ? 'bg-purple-500/30 text-purple-300 border border-purple-400/30' : widthLine ? 'bg-emerald-500/20 text-emerald-300' : 'bg-white/5 text-white/30'}`}>
              Width {widthLine ? '✓' : ''}
            </span>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${mode === 'height' && !heightLine ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-400/30' : heightLine ? 'bg-emerald-500/20 text-emerald-300' : 'bg-white/5 text-white/30'}`}>
              Height {heightLine ? '✓' : ''}
            </span>
          </div>
        </div>

        <div className="flex justify-center">
          <canvas
            ref={canvasRef}
            width={canvasSize.w}
            height={canvasSize.h}
            className="measurement-canvas rounded-xl max-w-full"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { setDrawing(false); setStartPoint(null) }}
          />
        </div>

        {widthLine && !heightLine && showCalibration && (
          <div className="mt-4 glass rounded-xl p-4 max-w-sm mx-auto">
            <p className="text-xs text-white/60 mb-2">
              Optional: If you know the real width, enter it for mm conversion
            </p>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={calibrationValue}
                onChange={(e) => setCalibrationValue(e.target.value)}
                className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white outline-none focus:border-purple-400/50"
                placeholder="Width in mm"
                min="0"
                step="0.1"
              />
              <span className="text-xs text-white/40">mm</span>
              <button
                onClick={handleCalibrationSubmit}
                className="px-3 py-2 bg-purple-500/20 border border-purple-400/30 rounded-lg text-xs text-purple-300 hover:bg-purple-500/30 transition-colors"
              >
                Set
              </button>
              <button
                onClick={() => setShowCalibration(false)}
                className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-xs text-white/40 hover:text-white/60 transition-colors"
              >
                Skip
              </button>
            </div>
          </div>
        )}

        {canProceed && (
          <div className="mt-4 grid grid-cols-2 gap-3 max-w-sm mx-auto">
            <div className="glass rounded-xl p-3 text-center">
              <p className="text-[10px] uppercase tracking-wider text-white/40 mb-1">Width</p>
              <p className="text-lg font-semibold text-purple-300">
                {getMmLength(widthLine) ? `${getMmLength(widthLine)} mm` : `${Math.round(getPixelLength(widthLine))} px`}
              </p>
            </div>
            <div className="glass rounded-xl p-3 text-center">
              <p className="text-[10px] uppercase tracking-wider text-white/40 mb-1">Height</p>
              <p className="text-lg font-semibold text-emerald-300">
                {getMmLength(heightLine) ? `${getMmLength(heightLine)} mm` : `${Math.round(getPixelLength(heightLine))} px`}
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="px-5 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white/60 hover:text-white hover:bg-white/10 transition-all"
        >
          Back
        </button>
        {widthLine && (
          <button
            onClick={() => { setWidthLine(null); setHeightLine(null); setMode('width'); setPxPerMm(null); setShowCalibration(true) }}
            className="px-5 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white/60 hover:text-white hover:bg-white/10 transition-all"
          >
            Redraw
          </button>
        )}
        <div className="flex-1" />
        {canProceed && (
          <button
            onClick={handleComplete}
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 text-sm font-medium hover:from-purple-500 hover:to-indigo-500 transition-all shadow-lg shadow-purple-900/30"
          >
            Continue
          </button>
        )}
      </div>
    </div>
  )
}
