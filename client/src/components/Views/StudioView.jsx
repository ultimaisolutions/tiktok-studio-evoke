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

          {/* Error Display */}
          {error && <div className="error-message">{error}</div>}

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
                {loading ? 'Starting...' : 'Start Studio Scraping'}
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
                {(isCompleted || isFailed) && (
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
            </div>

            <h4 className="mt-4 mb-2">Activity Log</h4>
            <ProgressLog logs={logs} maxHeight={250} />
          </div>
        )}
      </div>
    </div>
  );
}

export default StudioView;
