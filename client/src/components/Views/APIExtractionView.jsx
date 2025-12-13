import { useState, useEffect } from 'react';
import { api } from '../../hooks/useApi';
import { useWebSocket } from '../../hooks/useWebSocket';
import ProgressBar from '../Progress/ProgressBar';
import ProgressLog from '../Progress/ProgressLog';
import './Views.css';

function APIExtractionView() {
  const [studioBrowser, setStudioBrowser] = useState('chromium');
  const [cdpPort, setCdpPort] = useState('');
  const [sampleVideoCount, setSampleVideoCount] = useState(3);

  // Session state
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [awaitingLogin, setAwaitingLogin] = useState(false);

  // Patterns state
  const [patterns, setPatterns] = useState(null);
  const [patternsLoading, setPatternsLoading] = useState(true);
  const [expandedSections, setExpandedSections] = useState({});

  // WebSocket for progress
  const { isConnected, progress, logs, clearLogs } = useWebSocket(sessionId);

  // Config options
  const [studioBrowsers] = useState(['chromium', 'firefox', 'webkit']);

  // Load existing patterns on mount
  useEffect(() => {
    loadPatterns();
  }, []);

  // Check for login required event
  useEffect(() => {
    if (progress?.status === 'awaiting_login') {
      setAwaitingLogin(true);
    }
  }, [progress]);

  const loadPatterns = async () => {
    setPatternsLoading(true);
    try {
      const result = await api.getPatterns();
      if (result.found) {
        setPatterns(result.patterns);
      } else {
        setPatterns(null);
      }
    } catch (err) {
      console.error('Failed to load patterns:', err);
    } finally {
      setPatternsLoading(false);
    }
  };

  const handleStart = async (e) => {
    e.preventDefault();
    setError(null);
    setAwaitingLogin(false);
    clearLogs();

    setLoading(true);
    try {
      const request = {
        studio_browser: studioBrowser,
        cdp_port: cdpPort ? parseInt(cdpPort) : null,
        sample_video_count: sampleVideoCount,
      };

      const result = await api.startExtraction(request);
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
        await api.continueExtraction(sessionId);
        setAwaitingLogin(false);
      } catch (err) {
        setError(err.message);
      }
    }
  };

  const handleStop = async () => {
    if (sessionId) {
      try {
        await api.stopExtraction(sessionId);
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
    loadPatterns();
  };

  const handleApplyPatterns = async () => {
    try {
      const result = await api.applyPatterns();
      if (result.success) {
        setError(null);
        alert('Patterns applied successfully! The scraper will now use these API patterns.');
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDeletePatterns = async () => {
    if (!confirm('Are you sure you want to delete saved patterns?')) return;

    try {
      await api.deletePatterns();
      setPatterns(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleExportPatterns = () => {
    if (!patterns) return;

    const blob = new Blob([JSON.stringify(patterns, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'api_patterns.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const isRunning = progress?.status === 'running' || progress?.status === 'initializing';
  const isCompleted = progress?.status === 'completed';
  const isFailed = progress?.status === 'failed';

  const renderPatternSection = (title, pattern, sectionKey) => {
    if (!pattern) return null;

    const isExpanded = expandedSections[sectionKey];

    return (
      <div className="pattern-section">
        <div
          className="pattern-header"
          onClick={() => toggleSection(sectionKey)}
          style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
        >
          <h4 style={{ margin: 0 }}>{title}</h4>
          <span>{isExpanded ? '-' : '+'}</span>
        </div>

        {isExpanded && (
          <div className="pattern-details" style={{ marginTop: '0.5rem' }}>
            <div style={{ marginBottom: '0.5rem' }}>
              <strong>Endpoint:</strong> <code style={{ wordBreak: 'break-all' }}>{pattern.endpoint}</code>
            </div>
            <div style={{ marginBottom: '0.5rem' }}>
              <strong>Method:</strong> <span className="badge badge-info">{pattern.method}</span>
            </div>
            <div style={{ marginBottom: '0.5rem' }}>
              <strong>Category:</strong> <span className="badge badge-success">{pattern.category}</span>
            </div>

            {Object.keys(pattern.query_params || {}).length > 0 && (
              <div style={{ marginBottom: '0.5rem' }}>
                <strong>Query Parameters:</strong>
                <pre style={{
                  background: 'var(--color-bg-secondary)',
                  padding: '0.5rem',
                  borderRadius: '4px',
                  fontSize: '0.75rem',
                  overflow: 'auto',
                  maxHeight: '150px'
                }}>
                  {JSON.stringify(pattern.query_params, null, 2)}
                </pre>
              </div>
            )}

            {pattern.headers && Object.keys(pattern.headers).length > 0 && (
              <div style={{ marginBottom: '0.5rem' }}>
                <strong>Headers:</strong>
                <pre style={{
                  background: 'var(--color-bg-secondary)',
                  padding: '0.5rem',
                  borderRadius: '4px',
                  fontSize: '0.75rem',
                  overflow: 'auto',
                  maxHeight: '150px'
                }}>
                  {JSON.stringify(pattern.headers, null, 2)}
                </pre>
              </div>
            )}

            {pattern.response_schema && (
              <div style={{ marginBottom: '0.5rem' }}>
                <strong>Response Schema:</strong>
                <pre style={{
                  background: 'var(--color-bg-secondary)',
                  padding: '0.5rem',
                  borderRadius: '4px',
                  fontSize: '0.75rem',
                  overflow: 'auto',
                  maxHeight: '200px'
                }}>
                  {JSON.stringify(pattern.response_schema, null, 2)}
                </pre>
              </div>
            )}

            {pattern.captured_at && (
              <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
                Captured: {new Date(pattern.captured_at).toLocaleString()}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="view-container">
      <div className="view-header">
        <h2>API Extractor</h2>
        <p>Extract TikTok API patterns for resilient scraping</p>
      </div>

      {/* Saved Patterns Section */}
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="form-section">
          <div className="form-section-title">Saved Patterns</div>

          {patternsLoading ? (
            <div className="loading-state">Loading patterns...</div>
          ) : patterns ? (
            <div>
              <div style={{ marginBottom: '1rem', fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                Last updated: {new Date(patterns.last_updated).toLocaleString()}
              </div>

              {renderPatternSection('Video List API', patterns.video_list_api, 'video_list')}
              {renderPatternSection('Analytics API', patterns.analytics_api, 'analytics')}

              {patterns.other_apis?.length > 0 && (
                <div style={{ marginTop: '1rem' }}>
                  <h4>Other APIs ({patterns.other_apis.length})</h4>
                  {patterns.other_apis.map((api, idx) =>
                    renderPatternSection(`API ${idx + 1}: ${api.category}`, api, `other_${idx}`)
                  )}
                </div>
              )}

              <div className="action-buttons" style={{ marginTop: '1rem' }}>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={handleApplyPatterns}
                >
                  Apply to Scraper
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={handleExportPatterns}
                >
                  Export JSON
                </button>
                <button
                  type="button"
                  className="btn-danger"
                  onClick={handleDeletePatterns}
                >
                  Delete Patterns
                </button>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              No patterns extracted yet. Run an extraction to capture TikTok API patterns.
            </div>
          )}
        </div>
      </div>

      {/* Extraction Form */}
      <div className="card">
        <form onSubmit={handleStart}>
          <div className="form-section">
            <div className="form-section-title">Extraction Settings</div>
            <div className="form-row">
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

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="sampleVideoCount">Sample Video Count</label>
                <input
                  id="sampleVideoCount"
                  type="number"
                  min="1"
                  max="10"
                  value={sampleVideoCount}
                  onChange={(e) => setSampleVideoCount(parseInt(e.target.value) || 3)}
                  disabled={isRunning}
                />
                <div className="form-help">
                  Number of video analytics pages to sample (1-10)
                </div>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && <div className="error-message">{error}</div>}

          {/* Job Failure Display */}
          {isFailed && progress?.error && (
            <div className="error-message job-error">
              <strong>Extraction Failed:</strong> {progress.error}
            </div>
          )}

          {/* Login Required Prompt */}
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
                {loading ? 'Starting...' : 'Start Extraction'}
              </button>
            ) : (
              <>
                {(isRunning || awaitingLogin) && (
                  <button
                    type="button"
                    className="btn-danger"
                    onClick={handleStop}
                  >
                    Stop Extraction
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
                    New Extraction
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
              <span>Requests Captured: {progress?.total_requests_captured || 0}</span>
              <span>Patterns Found: {progress?.relevant_patterns || 0}</span>
            </div>

            <h4 className="mt-4 mb-2">Activity Log</h4>
            <ProgressLog logs={logs} maxHeight={250} />
          </div>
        )}
      </div>
    </div>
  );
}

export default APIExtractionView;
