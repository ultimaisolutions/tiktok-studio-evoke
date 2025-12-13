import { useEffect, useRef } from 'react';
import './ProgressLog.css';

function ProgressLog({ logs = [], maxHeight = 300 }) {
  const logEndRef = useRef(null);

  useEffect(() => {
    // Auto-scroll to bottom when new logs appear
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const getEventClass = (event) => {
    if (event.includes('complete') || event.includes('success')) return 'success';
    if (event.includes('fail') || event.includes('error')) return 'error';
    if (event.includes('start') || event.includes('progress')) return 'info';
    if (event.includes('login') || event.includes('warning')) return 'warning';
    return '';
  };

  const formatTime = (timestamp) => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return '';
    }
  };

  const formatEventData = (event, data) => {
    if (!data) return '';

    switch (event) {
      case 'download_start':
        return `Downloading: ${data.url?.slice(0, 50)}...`;
      case 'download_complete':
        return data.success ? `Downloaded: ${data.path || data.url}` : `Failed: ${data.message || 'Unknown error'}`;
      case 'analysis_start':
        return `Analyzing: ${data.index}/${data.total} - ${data.video_path?.split('/').pop()}`;
      case 'analysis_complete':
        return data.success ? `Analysis complete (${data.metrics?.processing_time_ms}ms)` : 'Analysis failed';
      case 'job_progress':
        return `Progress: ${data.completed}/${data.total} (${data.failed} failed)`;
      case 'job_completed':
        return 'Job completed successfully';
      case 'job_failed':
        return `Job failed: ${data.error}`;
      case 'studio_login_required':
        return 'Manual login required - please log in via the browser window';
      case 'studio_video_found':
        return `Found video: ${data.video_id}`;
      case 'studio_screenshot':
        return `Screenshot: ${data.tab} tab for ${data.video_id}`;
      default:
        return JSON.stringify(data).slice(0, 100);
    }
  };

  return (
    <div className="progress-log" style={{ maxHeight }}>
      {logs.length === 0 ? (
        <div className="progress-log-empty">No activity yet</div>
      ) : (
        logs.map((log, index) => (
          <div
            key={index}
            className={`progress-log-entry ${getEventClass(log.event)}`}
          >
            <span className="log-time">{formatTime(log.timestamp)}</span>
            <span className="log-event">{log.event}</span>
            <span className="log-message">{formatEventData(log.event, log.data)}</span>
          </div>
        ))
      )}
      <div ref={logEndRef} />
    </div>
  );
}

export default ProgressLog;
