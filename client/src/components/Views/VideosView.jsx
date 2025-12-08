import { useState, useEffect } from 'react';
import { api } from '../../hooks/useApi';
import './Views.css';
import './VideosView.css';

function VideosView() {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [outputDir, setOutputDir] = useState('videos');

  const loadVideos = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getVideos({ output_dir: outputDir });
      setVideos(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadVideos();
  }, [outputDir]);

  const loadVideoDetails = async (videoId) => {
    try {
      const data = await api.getVideo(videoId, outputDir);
      setSelectedVideo(data);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="view-container">
      <div className="view-header">
        <h2>Downloaded Videos</h2>
        <p>Browse and view downloaded videos with their analysis</p>
      </div>

      {/* Directory selector */}
      <div className="card mb-4">
        <div className="flex items-center gap-3">
          <label htmlFor="outputDir" className="font-medium">Directory:</label>
          <input
            id="outputDir"
            type="text"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            style={{ width: '200px' }}
          />
          <button className="btn-secondary" onClick={loadVideos}>
            Refresh
          </button>
        </div>
      </div>

      {error && <div className="error-message mb-4">{error}</div>}

      <div className="videos-layout">
        {/* Video List */}
        <div className="video-list-container card">
          {loading ? (
            <div className="loading-state">Loading videos...</div>
          ) : videos.length === 0 ? (
            <div className="empty-state">
              <p>No videos found in "{outputDir}"</p>
              <p className="text-sm text-muted mt-2">
                Download some videos first using the Download or Studio pages.
              </p>
            </div>
          ) : (
            <div className="video-list">
              {videos.map((video) => (
                <div
                  key={video.video_id}
                  className={`video-item ${selectedVideo?.video_id === video.video_id ? 'selected' : ''}`}
                  onClick={() => loadVideoDetails(video.video_id)}
                >
                  <div className="video-item-header">
                    <span className="video-id">{video.video_id}</span>
                    <span className="video-date">{video.date}</span>
                  </div>
                  <div className="video-item-meta">
                    <span className="video-username">@{video.username}</span>
                    <div className="video-badges">
                      {video.has_analysis && (
                        <span className="badge badge-success">Analyzed</span>
                      )}
                      {video.screenshots?.length > 0 && (
                        <span className="badge badge-info">Screenshots</span>
                      )}
                    </div>
                  </div>
                  {video.analysis_summary && (
                    <div className="video-item-summary">
                      <span>{video.analysis_summary.video_quality?.duration}s</span>
                      <span>{video.analysis_summary.motion_level} motion</span>
                      {video.analysis_summary.has_face && <span>Face</span>}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Video Details */}
        <div className="video-details-container card">
          {selectedVideo ? (
            <div className="video-details">
              <h3>Video Details</h3>

              <div className="detail-section">
                <h4>Info</h4>
                <dl className="detail-list">
                  <dt>Video ID</dt>
                  <dd>{selectedVideo.video_id}</dd>
                  <dt>Username</dt>
                  <dd>@{selectedVideo.username}</dd>
                  <dt>Date</dt>
                  <dd>{selectedVideo.date}</dd>
                  <dt>Path</dt>
                  <dd className="text-sm">{selectedVideo.path}</dd>
                </dl>
              </div>

              {selectedVideo.metadata && (
                <div className="detail-section">
                  <h4>Metadata</h4>
                  <dl className="detail-list">
                    {selectedVideo.metadata.description && (
                      <>
                        <dt>Description</dt>
                        <dd>{selectedVideo.metadata.description.slice(0, 100)}...</dd>
                      </>
                    )}
                    {selectedVideo.metadata.play_count && (
                      <>
                        <dt>Views</dt>
                        <dd>{selectedVideo.metadata.play_count.toLocaleString()}</dd>
                      </>
                    )}
                    {selectedVideo.metadata.like_count && (
                      <>
                        <dt>Likes</dt>
                        <dd>{selectedVideo.metadata.like_count.toLocaleString()}</dd>
                      </>
                    )}
                    {selectedVideo.metadata.comment_count && (
                      <>
                        <dt>Comments</dt>
                        <dd>{selectedVideo.metadata.comment_count.toLocaleString()}</dd>
                      </>
                    )}
                  </dl>
                </div>
              )}

              {selectedVideo.analysis && (
                <div className="detail-section">
                  <h4>Analysis</h4>
                  <dl className="detail-list">
                    <dt>Resolution</dt>
                    <dd>
                      {selectedVideo.analysis.video_quality?.resolution?.width}x
                      {selectedVideo.analysis.video_quality?.resolution?.height}
                    </dd>
                    <dt>Duration</dt>
                    <dd>{selectedVideo.analysis.video_quality?.duration_seconds}s</dd>
                    <dt>FPS</dt>
                    <dd>{selectedVideo.analysis.video_quality?.fps}</dd>
                    <dt>Motion</dt>
                    <dd>
                      {selectedVideo.analysis.motion_analysis?.motion_level} (
                      {selectedVideo.analysis.motion_analysis?.motion_score})
                    </dd>
                    <dt>Faces Detected</dt>
                    <dd>
                      {selectedVideo.analysis.content_detection?.face_detected
                        ? `Yes (max: ${selectedVideo.analysis.content_detection.max_face_count})`
                        : 'No'}
                    </dd>
                    <dt>Persons Detected</dt>
                    <dd>
                      {selectedVideo.analysis.content_detection?.person_detected
                        ? `Yes (max: ${selectedVideo.analysis.content_detection.max_person_count})`
                        : 'No'}
                    </dd>
                    <dt>Text Overlay</dt>
                    <dd>
                      {selectedVideo.analysis.content_detection?.text_overlay_detected
                        ? 'Yes'
                        : 'No'}
                    </dd>
                    <dt>Color Temperature</dt>
                    <dd>{selectedVideo.analysis.color_analysis?.color_temperature}</dd>
                    <dt>Has Audio</dt>
                    <dd>
                      {selectedVideo.analysis.audio_metrics?.has_audio
                        ? 'Yes'
                        : 'No'}
                    </dd>
                  </dl>
                </div>
              )}

              {Object.keys(selectedVideo.screenshots || {}).length > 0 && (
                <div className="detail-section">
                  <h4>Studio Screenshots</h4>
                  <div className="screenshot-list">
                    {['overview', 'viewers', 'engagement'].map(
                      (tab) =>
                        selectedVideo.screenshots[tab] && (
                          <a
                            key={tab}
                            href={`/api/videos/${selectedVideo.video_id}/screenshot/${tab}?output_dir=${outputDir}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="screenshot-link"
                          >
                            {tab.charAt(0).toUpperCase() + tab.slice(1)}
                          </a>
                        )
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state">
              <p>Select a video to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default VideosView;
