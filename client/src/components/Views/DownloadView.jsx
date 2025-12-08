import { useState, useEffect } from 'react';
import { api } from '../../hooks/useApi';
import { useWebSocket } from '../../hooks/useWebSocket';
import ProgressBar from '../Progress/ProgressBar';
import ProgressLog from '../Progress/ProgressLog';
import './Views.css';

function DownloadView() {
  const [urls, setUrls] = useState('');
  const [outputDir, setOutputDir] = useState('videos');
  const [browser, setBrowser] = useState('chrome');
  const [noBrowser, setNoBrowser] = useState(false);
  const [analyze, setAnalyze] = useState(false);
  const [showAnalysisOptions, setShowAnalysisOptions] = useState(false);

  // Analysis options
  const [thoroughness, setThoroughness] = useState('balanced');

  // Job state
  const [jobId, setJobId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // WebSocket for progress
  const { isConnected, progress, logs, clearLogs } = useWebSocket(jobId);

  // Load config on mount
  const [browsers, setBrowsers] = useState(['chrome', 'firefox', 'edge', 'opera', 'brave', 'chromium']);
  const [presets, setPresets] = useState(['quick', 'balanced', 'thorough', 'maximum', 'extreme']);

  useEffect(() => {
    api.getBrowsers().then((data) => setBrowsers(data.browsers)).catch(() => {});
    api.getPresets().then((data) => setPresets(data.presets)).catch(() => {});
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    clearLogs();

    const urlList = urls
      .split('\n')
      .map((u) => u.trim())
      .filter((u) => u && !u.startsWith('#'));

    if (urlList.length === 0) {
      setError('Please enter at least one URL');
      return;
    }

    setLoading(true);
    try {
      const request = {
        urls: urlList,
        output_dir: outputDir,
        browser,
        no_browser: noBrowser,
        analyze,
        analysis_options: analyze ? { thoroughness } : null,
      };

      const result = await api.startDownload(request);
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
        await api.cancelDownload(jobId);
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
        <h2>Download Videos</h2>
        <p>Download TikTok videos from URLs with optional analysis</p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          {/* URL Input */}
          <div className="form-section">
            <label htmlFor="urls">TikTok URLs (one per line)</label>
            <textarea
              id="urls"
              value={urls}
              onChange={(e) => setUrls(e.target.value)}
              placeholder="https://www.tiktok.com/@user/video/123456789&#10;https://www.tiktok.com/@user/video/987654321"
              rows={6}
              disabled={isRunning}
            />
            <div className="form-help">
              Paste TikTok video URLs, one per line. Lines starting with # are ignored.
            </div>
          </div>

          {/* Basic Options */}
          <div className="form-section">
            <div className="form-section-title">Options</div>
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
                <label htmlFor="browser">Browser (for cookies)</label>
                <select
                  id="browser"
                  value={browser}
                  onChange={(e) => setBrowser(e.target.value)}
                  disabled={isRunning || noBrowser}
                >
                  {browsers.map((b) => (
                    <option key={b} value={b}>
                      {b.charAt(0).toUpperCase() + b.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="checkbox-group">
              <input
                type="checkbox"
                id="noBrowser"
                checked={noBrowser}
                onChange={(e) => setNoBrowser(e.target.checked)}
                disabled={isRunning}
              />
              <label htmlFor="noBrowser">
                Skip browser authentication (public videos only)
              </label>
            </div>

            <div className="checkbox-group mt-2">
              <input
                type="checkbox"
                id="analyze"
                checked={analyze}
                onChange={(e) => {
                  setAnalyze(e.target.checked);
                  setShowAnalysisOptions(e.target.checked);
                }}
                disabled={isRunning}
              />
              <label htmlFor="analyze">Analyze videos after download</label>
            </div>
          </div>

          {/* Analysis Options */}
          {showAnalysisOptions && (
            <div className="form-section">
              <div className="form-section-title">Analysis Options</div>
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
                <div className="form-help">
                  Higher thoroughness = more frames analyzed, better accuracy, longer processing
                </div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && <div className="error-message">{error}</div>}

          {/* Action Buttons */}
          <div className="action-buttons">
            {!jobId ? (
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? 'Starting...' : 'Start Download'}
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
                    New Download
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
              <span>Completed: {progress?.completed || 0}</span>
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

export default DownloadView;
