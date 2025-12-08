import './ProgressBar.css';

function ProgressBar({ progress = 0, status = 'pending', currentTask = null }) {
  const percentage = Math.round(progress * 100);

  return (
    <div className="progress-bar-container">
      <div className="progress-bar-header">
        <span className={`progress-status status-${status}`}>
          {status === 'running' ? 'In Progress' : status}
        </span>
        <span className="progress-percentage">{percentage}%</span>
      </div>
      <div className="progress-bar-track">
        <div
          className={`progress-bar-fill status-${status}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {currentTask && (
        <div className="progress-current-task">{currentTask}</div>
      )}
    </div>
  );
}

export default ProgressBar;
