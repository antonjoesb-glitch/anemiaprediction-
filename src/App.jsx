import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('manual') // 'manual' or 'upload'
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  // Manual Form State
  const [formData, setFormData] = useState({
    gender: '',
    age: '',
    hemoglobin: '',
    rbc: '',
    mcv: '',
    mch: '',
    mchc: '',
    hematocrit: ''
  })

  // Upload Form State
  const [uploadData, setUploadData] = useState({
    gender: '',
    age: '',
    hemoglobin: '',
    rbc: '',
    mcv: '',
    mch: '',
    mchc: '',
  })

  const [isDragging, setIsDragging] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null) // null, 'reading', 'success', 'error'
  const fileInputRef = useRef(null)

  const handleManualChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value })
  }

  const handleUploadDataChange = (e) => {
    setUploadData({ ...uploadData, [e.target.id]: e.target.value })
  }

  const handleManualSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      const response = await fetch('/api/predict.py', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })

      if (!response.ok) throw new Error('Prediction request failed')

      const data = await response.json()
      setResult(data)
    } catch (err) {
      alert('Error connecting to AI backend: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAnalyzeUpload = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      const response = await fetch('/api/predict.py', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...uploadData,
          hematocrit: formData.hematocrit // not strongly required
        })
      })

      if (!response.ok) throw new Error('Prediction request failed')
      const data = await response.json()
      setResult(data)
    } catch (err) {
      alert('Error connecting to AI backend: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (file) => {
    if (!file) return
    setUploadStatus('reading')

    const formDataObj = new FormData()
    formDataObj.append('file', file)

    try {
      const response = await fetch('/api/extract_report.py', {
        method: 'POST',
        body: formDataObj
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Server error occurred')
      }

      if (data.success) {
        const extracted = data.extracted_values
        setUploadData(prev => ({
          ...prev,
          hemoglobin: extracted.hemoglobin || '',
          rbc: extracted.rbc || '',
          mcv: extracted.mcv || '',
          mch: extracted.mch || '',
          mchc: extracted.mchc || '',
        }))
        setUploadStatus('success')
      } else {
        throw new Error(data.error || 'Unknown error')
      }
    } catch (err) {
      alert('Error reading report: ' + err.message)
      setUploadStatus('error')
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0])
    }
  }

  const getRiskLevel = (probStr) => {
    if (!probStr) return { level: 'Unknown', color: '#64748b', percentage: 0 };
    const prob = parseFloat(probStr.replace('%', ''));
    if (prob <= 30) return { level: 'Low', color: '#10b981', percentage: prob };
    if (prob <= 60) return { level: 'Moderate', color: '#f59e0b', percentage: prob };
    return { level: 'High', color: '#ef4444', percentage: prob };
  }

  return (
    <div className="app-container">
      <div className="background-glow" />

      <header>
        <h1>Anemia Prediction</h1>
        <p>AI Clinical Diagnostic System</p>
      </header>

      <div className="tabs">
        <button
          className={`tab-btn ${activeTab === 'manual' ? 'active' : ''}`}
          onClick={() => { setActiveTab('manual'); setResult(null); }}
        >
          Manual Input
        </button>
        <button
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => { setActiveTab('upload'); setResult(null); }}
        >
          Upload CBC Report
        </button>
      </div>

      <main className="card">
        {activeTab === 'manual' ? (
          <form onSubmit={handleManualSubmit}>
            <div className="form-grid">
              <div className="input-block">
                <label htmlFor="gender">Gender</label>
                <select id="gender" value={formData.gender} onChange={handleManualChange} required>
                  <option value="" disabled>Select gender</option>
                  <option value="1">Male</option>
                  <option value="0">Female</option>
                </select>
              </div>
              <div className="input-block">
                <label htmlFor="age">Age</label>
                <input type="number" id="age" value={formData.age} onChange={handleManualChange} placeholder="Years" required min="0" />
              </div>
              <div className="input-block">
                <label htmlFor="hemoglobin">Hemoglobin (g/dL)</label>
                <input type="number" id="hemoglobin" step="0.1" value={formData.hemoglobin} onChange={handleManualChange} placeholder="9.0 - 18.0" required />
              </div>
              <div className="input-block">
                <label htmlFor="rbc">RBC Count (10¹²/L)</label>
                <input type="number" id="rbc" step="0.1" value={formData.rbc} onChange={handleManualChange} placeholder="3.5 - 6.0" required />
              </div>
              <div className="input-block">
                <label htmlFor="mcv">MCV (fL)</label>
                <input type="number" id="mcv" step="0.1" value={formData.mcv} onChange={handleManualChange} placeholder="70 - 100" required />
              </div>
              <div className="input-block">
                <label htmlFor="mch">MCH (pg)</label>
                <input type="number" id="mch" step="0.1" value={formData.mch} onChange={handleManualChange} placeholder="25 - 35" required />
              </div>
              <div className="input-block">
                <label htmlFor="mchc">MCHC (g/dL)</label>
                <input type="number" id="mchc" step="0.1" value={formData.mchc} onChange={handleManualChange} placeholder="28 - 36" required />
              </div>
              <div className="input-block">
                <label htmlFor="hematocrit">Hematocrit (%)</label>
                <input type="number" id="hematocrit" step="0.1" value={formData.hematocrit} onChange={handleManualChange} placeholder="Optional" />
              </div>
            </div>
            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? <div className="loading-spinner" /> : 'Run AI Diagnostic'}
            </button>
          </form>
        ) : (
          <div className="upload-section">
            <div
              className={`drop-zone ${isDragging ? 'dragging' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => { if (e.target.files[0]) handleFileUpload(e.target.files[0]) }}
                accept=".pdf,image/png,image/jpeg,image/jpg"
                hidden
              />
              {uploadStatus === 'reading' ? (
                <div className="upload-reading">
                  <div className="loading-spinner upload-spinner"></div>
                  <p>Reading your report...</p>
                </div>
              ) : (
                <div className="upload-prompt">
                  <span className="upload-icon">📄</span>
                  <p>Drag & Drop your CBC Report here</p>
                  <span className="upload-subtitle">Supports PDF, JPG, PNG</span>
                </div>
              )}
            </div>

            {uploadStatus === 'success' && (
              <form onSubmit={handleAnalyzeUpload} className="extracted-data-form fade-in">
                <h3 className="section-title">Review Extracted Values</h3>
                <div className="form-grid">
                  <div className="input-block">
                    <label htmlFor="gender">Gender (Required)</label>
                    <select id="gender" value={uploadData.gender} onChange={handleUploadDataChange} required>
                      <option value="" disabled>Select</option>
                      <option value="1">Male</option>
                      <option value="0">Female</option>
                    </select>
                  </div>
                  <div className="input-block">
                    <label htmlFor="age">Age (Required)</label>
                    <input type="number" id="age" value={uploadData.age} onChange={handleUploadDataChange} placeholder="Years" required min="0" />
                  </div>
                  <div className="input-block highlighted-input">
                    <label htmlFor="hemoglobin">Hemoglobin (g/dL)</label>
                    <input type="number" id="hemoglobin" step="0.1" value={uploadData.hemoglobin} onChange={handleUploadDataChange} required />
                  </div>
                  <div className="input-block highlighted-input">
                    <label htmlFor="rbc">RBC Count (10¹²/L)</label>
                    <input type="number" id="rbc" step="0.1" value={uploadData.rbc} onChange={handleUploadDataChange} required />
                  </div>
                  <div className="input-block highlighted-input">
                    <label htmlFor="mcv">MCV (fL)</label>
                    <input type="number" id="mcv" step="0.1" value={uploadData.mcv} onChange={handleUploadDataChange} required />
                  </div>
                  <div className="input-block highlighted-input">
                    <label htmlFor="mch">MCH (pg)</label>
                    <input type="number" id="mch" step="0.1" value={uploadData.mch} onChange={handleUploadDataChange} required />
                  </div>
                  <div className="input-block highlighted-input">
                    <label htmlFor="mchc">MCHC (g/dL)</label>
                    <input type="number" id="mchc" step="0.1" value={uploadData.mchc} onChange={handleUploadDataChange} required />
                  </div>
                </div>
                <button type="submit" className="submit-btn analyze-btn" disabled={loading}>
                  {loading ? <div className="loading-spinner" /> : 'Analyze My Report'}
                </button>
              </form>
            )}
          </div>
        )}
      </main>

      {result && (
        <div className="result-overlay">
          <div className="result-card">
            {activeTab === 'upload' ? (
              <div className="gauge-container">
                <svg viewBox="0 0 100 50" className="gauge-svg">
                  <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#e2e8f0" strokeWidth="10" strokeLinecap="round" />
                  <path
                    d="M 10 50 A 40 40 0 0 1 90 50"
                    fill="none"
                    stroke={getRiskLevel(result.Probability).color}
                    strokeWidth="10"
                    strokeLinecap="round"
                    strokeDasharray="125.6" /* Pi * R */
                    strokeDashoffset={125.6 - (getRiskLevel(result.Probability).percentage / 100 * 125.6)}
                    className="gauge-progress"
                  />
                </svg>
                <div className="gauge-text">
                  <h2 style={{ color: getRiskLevel(result.Probability).color }}>
                    {getRiskLevel(result.Probability).percentage.toFixed(1)}%
                  </h2>
                  <p>Anemia Risk</p>
                </div>
                <div className="risk-level" style={{ backgroundColor: `${getRiskLevel(result.Probability).color}20`, color: getRiskLevel(result.Probability).color }}>
                  {getRiskLevel(result.Probability).level} Risk
                </div>
                <p className="brief-explanation">
                  Based on your supplied CBC parameters and our clinical AI model,
                  you have a {getRiskLevel(result.Probability).level.toLowerCase()} probability of anemia.
                </p>
              </div>
            ) : (
              <>
                <div className={`status-icon ${result.Anemia === 'Yes' ? 'status-positive' : 'status-negative'}`}>
                  {result.Anemia === 'Yes' ? '⚠️' : '✅'}
                </div>
                <h2>{result.Anemia === 'Yes' ? 'Anemia Detected' : 'No Anemia Found'}</h2>
                <p className="probability">Confidence Score: {result.Probability}</p>
              </>
            )}

            <div className="disclaimer-box">
              This result is generated by a machine learning model based on clinical datasets.
              It is for informational purposes only. Please consult a physician for official medical advice.
            </div>

            <button className="close-btn" onClick={() => setResult(null)}>
              Dismiss
            </button>
          </div>
        </div>
      )}

      <footer>
        <p>&copy; 2026 PrecisionHealth AI. Optimized for medical professionals.</p>
      </footer>
    </div>
  )
}

export default App
