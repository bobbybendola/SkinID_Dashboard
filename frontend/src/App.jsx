import { useState } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import UploadZone from './components/UploadZone'
import MeasurementCanvas from './components/MeasurementCanvas'
import PatientForm from './components/PatientForm'
import ResultsCard from './components/ResultsCard'
import Disclaimer from './components/Disclaimer'
import usePredict from './hooks/usePredict'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [measurements, setMeasurements] = useState(null)
  const [patientInfo, setPatientInfo] = useState({ sex: '', age: '', location: '' })
  const [step, setStep] = useState('upload') // upload | measure | form | results

  const { result, loading, error, predict } = usePredict()

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setStep('measure')
  }

  const handleMeasurementComplete = (dims) => {
    setMeasurements(dims)
    setStep('form')
  }

  const handleSubmit = async (info) => {
    setPatientInfo(info)
    setStep('results')
    await predict(file, measurements, info)
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setMeasurements(null)
    setPatientInfo({ sex: '', age: '', location: '' })
    setStep('upload')
  }

  return (
    <div className="min-h-screen text-white">
      <Navbar />
      <main className="max-w-4xl mx-auto px-4 pb-16">
        <Hero />

        {step === 'upload' && (
          <div className="animate-fade-in-up">
            <UploadZone onFileSelect={handleFileSelect} />
          </div>
        )}

        {step === 'measure' && preview && (
          <div className="animate-fade-in-up">
            <MeasurementCanvas
              imageSrc={preview}
              onComplete={handleMeasurementComplete}
              onBack={() => { setStep('upload'); setFile(null); setPreview(null) }}
            />
          </div>
        )}

        {step === 'form' && (
          <div className="animate-fade-in-up">
            <PatientForm
              preview={preview}
              measurements={measurements}
              onSubmit={handleSubmit}
              onBack={() => setStep('measure')}
            />
          </div>
        )}

        {step === 'results' && (
          <div className="animate-fade-in-up">
            <ResultsCard
              result={result}
              loading={loading}
              error={error}
              preview={preview}
              measurements={measurements}
              patientInfo={patientInfo}
              onReset={handleReset}
            />
          </div>
        )}

        <Disclaimer />
      </main>
    </div>
  )
}

export default App
