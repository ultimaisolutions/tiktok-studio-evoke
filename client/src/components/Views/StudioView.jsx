import { useState, useEffect } from 'react';
import { api } from '../../hooks/useApi';
import { useWebSocket } from '../../hooks/useWebSocket';
import ProgressBar from '../Progress/ProgressBar';
import ProgressLog from '../Progress/ProgressLog';
import './Views.css';

function StudioView() {
  const [outputDir, setOutputDir] = useState('videos');
  const [studioBrowser, setStudioBrowser] = useState('chromium');
  const [username, setUsername] = useState('');
  const [cdpPort, setCdpPort] = useState('');
  const [skipDownload, setSkipDownload] = useState(false);
  const [skipAnalysis, setSkipAnalysis] = useState(false);
  const [thoroughness, setThoroughness] = useState('extreme');

  // Parallelization options
  const [studioWorkers, setStudioWorkers] = useState(2);
  const [downloadWorkers, setDownloadWorkers] = useState(4);
  const [requestDelay, setRequestDelay] = useState(1500);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Session state
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [awaitingLogin, setAwaitingLogin] = useState(false);

  // WebSocket for progress
  const { isConnected, progress, logs, clearLogs } = useWebSocket(sessionId);

  // Config options
  const [studioBrowsers] = useState(['chromium', 'firefox', 'webkit']);
  const [presets] = useState(['quick', 'balanced', 'thorough', 'maximum', 'extreme']);

  // Check for login required event
  useEffect(() => {
    if (progress?.status === 'awaiting_login') {
      setAwaitingLogin(true);
    }
  }, [progress]);

  const handleStart = async (e) => {
    e.preventDefault();
    setError(null);
    setAwaitingLogin(false);
    clearLogs();

    setLoading(true);
    try {
      const request = {
        output_dir: outputDir,
        studio_browser: studioBrowser,
        skip_download: skipDownload,
        skip_analysis: skipAnalysis,
        cdp_port: cdpPort ? parseInt(cdpPort) : null,
        username: username || null,
        analysis_options: !skipAnalysis ? { thoroughness } : null,
        // Parallelization options
        studio_workers: studioWorkers,
        download_workers: skipDownload ? null : downloadWorkers,
        request_delay_ms: requestDelay,
      };

      const result = await api.startStudio(request);
      setSessionId(result.session_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleContinueLogin = async () => {
    if (sessionId) {
      try {
        await api.continueStudio(sessionId);
        setAwaitingLogin(false);
      } catch (err) {
        setError(err.message);
      }
    }
  };

  const handleStop = async () => {
    if (sessionId) {
      try {
        await api.stopStudio(sessionId);
        setSessionId(null);
        setAwaitingLogin(false);
      } catch (err) {
        setError(err.message);
      }
    }
  };

  const handleReset = () => {
    setSessionId(null);
    setAwaitingLogin(false);
    clearLogs();
    setError(null);
  };

  const isRunning = progress?.status === 'running' || progress?.status === 'initializing';
  const isCompleted = progress?.status === 'completed';
  const isFailed = progress?.status === 'failed';

  return (
    <div className="view-container">
      <div className="view-header">
        <h2>TikTok Studio</h2>
        <p>Capture analytics screenshots directly from TikTok Studio</p>
      </div>

      <div className="card">
        <form onSubmit={handleStart}>
          {/* Studio Options */}
          <div className="form-section">
            <div className="form-section-title">Studio Settings</div>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="outputDir">Output Directory</label>
                <input
                  id="outputDir"
                  type="text"
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  disabled={isRunning}
                />
              </div>
              <div className="form-group">
                <label htmlFor="studioBrowser">Browser</label>
                <select
                  id="studioBrowser"
                  value={studioBrowser}
                  onChange={(e) => setStudioBrowser(e.target.value)}
                  disabled={isRunning}
                >
                  {studioBrowsers.map((b) => (
                    <option key={b} value={b}>
                      {b.charAt(0).toUpperCase() + b.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="username">TikTok Username (optional)</label>
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="@username"
                  disabled={isRunning}
                />
                <div className="form-help">
                  Used for constructing video URLs if not detected automatically
                </div>
              </div>
              <div className="form-group">
                <label htmlFor="cdpPort">CDP Port (optional)</label>
                <input
                  id="cdpPort"
                  type="number"
                  value={cdpPort}
                  onChange={(e) => setCdpPort(e.target.value)}
                  placeholder="9222"
                  disabled={isRunning}
                />
                <div className="form-help">
                  Connect to existing browser with remote debugging
                </div>
              </div>
            </div>
          </div>

          {/* Processing Options */}
          <div className="form-section">
            <div className="form-section-title">Processing Options</div>

            <div className="checkbox-group">
              <input
                type="checkbox"
                id="skipDownload"
                checked={skipDownload}
                onChange={(e) => setSkipDownload(e.target.checked)}
                disabled={isRunning}
              />
              <label htmlFor="skipDownload">
                Skip video download (screenshots only)
              </label>
            </div>

            <div className="checkbox-group mt-2">
              <input
                type="checkbox"
                id="skipAnalysis"
                checked={skipAnalysis}
                onChange={(e) => setSkipAnalysis(e.target.checked)}
                disabled={isRunning || skipDownload}
              />
              <label htmlFor="skipAnalysis">
                Skip video analysis
              </label>
            </div>

            {!skipAnalysis && !skipDownload && (
              <div className="form-group mt-3">
                <label htmlFor="thoroughness">Analysis Thoroughness</label>
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
            )}
          </div>

          {/* Advanced Options */}
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
                    <label htmlFor="studioWorkers">Screenshot Workers (1-4)</label>
                    <input
                      id="studioWorkers"
                      type="number"
                      min="1"
                      max="4"
                      value={studioWorkers}
                      onChange={(e) => setStudioWorkers(parseInt(e.target.value) || 2)}
                      disabled={isRunning}
                    />
                    <div className="form-help">
                      Parallel browser pages for capturing screenshots
                    </div>
                  </div>
                  <div className="form-group">
                    <label htmlFor="downloadWorkers">Download Workers (1-8)</label>
                    <input
                      id="downloadWorkers"
                      type="number"
                      min="1"
                      max="8"
                      value={downloadWorkers}
                      onChange={(e) => setDownloadWorkers(parseInt(e.target.value) || 4)}
                      disabled={isRunning || skipDownload}
                    />
                    <div className="form-help">
                      Concurrent video downloads
                    </div>
                  </div>
                </div>

                <div className="form-group">
                  <label htmlFor="requestDelay">Request Delay (ms)</label>
                  <input
                    id="requestDelay"
                    type="number"
                    min="500"
                    max="5000"
                    step="100"
                    value={requestDelay}
                    onChange={(e) => setRequestDelay(parseInt(e.target.value) || 1500)}
                    disabled={isRunning}
                  />
                  <div className="form-help">
                    Delay between requests to avoid TikTok rate limiting (500-5000ms)
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && <div className="error-message">{error}</div>}

          {/* Job Failure Display */}
          {isFailed && progress?.error && (
            <div className="error-message job-error">
              <strong>Job Failed:</strong> {progress.error}
            </div>
          )}

          {/* Login Required Modal */}
          {awaitingLogin && (
            <div className="login-prompt">
              <h4>Manual Login Required</h4>
              <p>
                Please log in to TikTok in the browser window that opened.
                Once you're logged in, click the button below to continue.
              </p>
              <button
                type="button"
                className="btn-primary"
                onClick={handleContinueLogin}
              >
                I've Logged In - Continue
              </button>
            </div>
          )}

          {/* Action Buttons */}
          <div className="action-buttons">
            {!sessionId ? (
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? 'Starting...' : 'New Session'}
              </button>
            ) : (
              <>
                {(isRunning || awaitingLogin) && (
                  <button
                    type="button"
                    className="btn-danger"
                    onClick={handleStop}
                  >
                    Stop Session
                  </button>
                )}
                {isFailed && (
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={handleReset}
                  >
                    Retry
                  </button>
                )}
                {isCompleted && (
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={handleReset}
                  >
                    New Session
                  </button>
                )}
              </>
            )}
          </div>
        </form>

        {/* Progress Display */}
        {sessionId && (
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
              status={awaitingLogin ? 'pending' : (progress?.status || 'pending')}
              currentTask={awaitingLogin ? 'Waiting for login...' : progress?.current_task}
            />

            <div className="progress-stats">
              <span>Processed: {progress?.completed || 0}</span>
              <span>Downloaded: {progress?.downloaded || 0}</span>
              <span>Analyzed: {progress?.analyzed || 0}</span>
              {progress?.workers_active !== undefined && (
                <span>Workers: {progress.workers_active}/{studioWorkers}</span>
              )}
            </div>

            {progress?.current_videos && progress.current_videos.length > 0 && (
              <div className="current-videos-info" style={{ fontSize: '0.85em', color: '#666', marginTop: '0.5em' }}>
                Processing: {progress.current_videos.slice(0, 3).join(', ')}
                {progress.current_videos.length > 3 && ` +${progress.current_videos.length - 3} more`}
              </div>
            )}

            <h4 className="mt-4 mb-2">Activity Log</h4>
            <ProgressLog logs={logs} maxHeight={250} />
          </div>
        )}
      </div>
    </div>
  );
}

export default StudioView;
