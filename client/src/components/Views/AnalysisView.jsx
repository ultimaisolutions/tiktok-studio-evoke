import { useState } from 'react';
import { api } from '../../hooks/useApi';
import { useWebSocket } from '../../hooks/useWebSocket';
import ProgressBar from '../Progress/ProgressBar';
import ProgressLog from '../Progress/ProgressLog';
import './Views.css';

function AnalysisView() {
  const [outputDir, setOutputDir] = useState('videos');
  const [thoroughness, setThoroughness] = useState('balanced');
  const [samplePercent, setSamplePercent] = useState('');
  const [colorClusters, setColorClusters] = useState('');
  const [motionRes, setMotionRes] = useState('');
  const [workers, setWorkers] = useState('');
  const [skipAudio, setSkipAudio] = useState(false);
  const [sceneDetection, setSceneDetection] = useState(false);
  const [fullResolution, setFullResolution] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Job state
  const [jobId, setJobId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // WebSocket for progress
  const { isConnected, progress, logs, clearLogs } = useWebSocket(jobId);

  const [presets] = useState(['quick', 'balanced', 'thorough', 'maximum', 'extreme']);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    clearLogs();

    setLoading(true);
    try {
      const request = {
        output_dir: outputDir,
        analysis_options: {
          thoroughness,
          sample_percent: samplePercent ? parseInt(samplePercent) : null,
          color_clusters: colorClusters ? parseInt(colorClusters) : null,
          motion_res: motionRes ? parseInt(motionRes) : null,
          workers: workers ? parseInt(workers) : null,
          skip_audio: skipAudio,
          scene_detection: sceneDetection,
          full_resolution: fullResolution,
        },
      };

      const result = await api.startAnalysis(request);
      setJobId(result.job_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    if (jobId) {
      try {
        await api.cancelAnalysis(jobId);
        setJobId(null);
      } catch (err) {
        setError(err.message);
      }
    }
  };

  const handleReset = () => {
    setJobId(null);
    clearLogs();
    setError(null);
  };

  const isRunning = progress?.status === 'running';
  const isCompleted = progress?.status === 'completed';
  const isFailed = progress?.status === 'failed';

  return (
    <div className="view-container">
      <div className="view-header">
        <h2>Analyze Videos</h2>
        <p>Analyze existing videos for visual metrics, faces, motion, and more</p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          {/* Basic Options */}
          <div className="form-section">
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="outputDir">Videos Directory</label>
                <input
                  id="outputDir"
                  type="text"
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  disabled={isRunning}
                />
                <div className="form-help">
                  Directory containing downloaded videos to analyze
                </div>
              </div>
              <div className="form-group">
                <label htmlFor="thoroughness">Thoroughness Preset</label>
                <select
                  id="thoroughness"
                  value={thoroughness}
                  onChange={(e) => setThoroughness(e.target.value)}
                  disabled={isRunning}
                >
                  {presets.map((p) => (
                    <option key={p} value={p}>
                      {p.charAt(0).toUpperCase() + p.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Advanced Options Toggle */}
          <div className="collapsible-section">
            <div
              className="collapsible-header"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <span>Advanced Options</span>
              <span>{showAdvanced ? '−' : '+'}</span>
            </div>

            {showAdvanced && (
              <div className="collapsible-content">
                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="samplePercent">Sample Percentage (1-100)</label>
                    <input
                      id="samplePercent"
                      type="number"
                      min="1"
                      max="100"
                      value={samplePercent}
                      onChange={(e) => setSamplePercent(e.target.value)}
                      placeholder="Auto (from preset)"
                      disabled={isRunning}
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="colorClusters">Color Clusters (3-20)</label>
                    <input
                      id="colorClusters"
                      type="number"
                      min="3"
                      max="20"
                      value={colorClusters}
                      onChange={(e) => setColorClusters(e.target.value)}
                      placeholder="Auto (from preset)"
                      disabled={isRunning}
                    />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="motionRes">Motion Resolution (80-1080)</label>
                    <input
                      id="motionRes"
                      type="number"
                      min="80"
                      max="1080"
                      value={motionRes}
                      onChange={(e) => setMotionRes(e.target.value)}
                      placeholder="Auto (from preset)"
                      disabled={isRunning}
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="workers">Parallel Workers</label>
                    <input
                      id="workers"
                      type="number"
                      min="1"
                      max="16"
                      value={workers}
                      onChange={(e) => setWorkers(e.target.value)}
                      placeholder="Auto (CPU - 1)"
                      disabled={isRunning}
                    />
                  </div>
                </div>

                <div className="checkbox-group mt-3">
                  <input
                    type="checkbox"
                    id="sceneDetection"
                    checked={sceneDetection}
                    onChange={(e) => setSceneDetection(e.target.checked)}
                    disabled={isRunning}
                  />
                  <label htmlFor="sceneDetection">
                    Enable scene/cut detection
                  </label>
                </div>

                <div className="checkbox-group mt-2">
                  <input
                    type="checkbox"
                    id="fullResolution"
                    checked={fullResolution}
                    onChange={(e) => setFullResolution(e.target.checked)}
                    disabled={isRunning}
                  />
                  <label htmlFor="fullResolution">
                    Full resolution analysis (no downsampling)
                  </label>
                </div>

                <div className="checkbox-group mt-2">
                  <input
                    type="checkbox"
                    id="skipAudio"
                    checked={skipAudio}
                    onChange={(e) => setSkipAudio(e.target.checked)}
                    disabled={isRunning}
                  />
                  <label htmlFor="skipAudio">
                    Skip audio analysis
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && <div className="error-message">{error}</div>}

          {/* Action Buttons */}
          <div className="action-buttons">
            {!jobId ? (
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? 'Starting...' : 'Start Analysis'}
              </button>
            ) : (
              <>
                {isRunning && (
                  <button
                    type="button"
                    className="btn-danger"
                    onClick={handleCancel}
                  >
                    Cancel
                  </button>
                )}
                {(isCompleted || isFailed) && (
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={handleReset}
                  >
                    New Analysis
                  </button>
                )}
              </>
            )}
          </div>
        </form>

        {/* Progress Display */}
        {jobId && (
          <div className="progress-section mt-4">
            <div className="progress-header">
              <h4>Progress</h4>
              {isConnected ? (
                <span className="connection-status connected">Connected</span>
              ) : (
                <span className="connection-status disconnected">Disconnected</span>
              )}
            </div>

            <ProgressBar
              progress={progress?.progress || 0}
              status={progress?.status || 'pending'}
              currentTask={progress?.current_task}
            />

            <div className="progress-stats">
              <span>Analyzed: {progress?.completed || 0}</span>
              <span>Failed: {progress?.failed || 0}</span>
              <span>Total: {progress?.total || 0}</span>
            </div>

            <h4 className="mt-4 mb-2">Activity Log</h4>
            <ProgressLog logs={logs} maxHeight={250} />
          </div>
        )}
      </div>
    </div>
  );
}

export default AnalysisView;
