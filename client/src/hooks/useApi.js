import { useState, useCallback } from 'react';

const API_BASE = '/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || error.message || 'Request failed');
  }

  return response.json();
}

/**
 * Hook for making API calls with loading and error states.
 *
 * @returns {object} - { loading, error, get, post, clearError }
 */
export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const get = useCallback(async (endpoint) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch(endpoint);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const post = useCallback(async (endpoint, body) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch(endpoint, {
        method: 'POST',
        body: JSON.stringify(body),
      });
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return { loading, error, get, post, clearError };
}

/**
 * API service functions
 */
export const api = {
  // Config
  getConfig: () => apiFetch('/config/all'),
  getBrowsers: () => apiFetch('/config/browsers'),
  getPresets: () => apiFetch('/config/presets'),

  // Scraper
  startDownload: (request) => apiFetch('/scraper/download', {
    method: 'POST',
    body: JSON.stringify(request),
  }),
  getDownloadStatus: (jobId) => apiFetch(`/scraper/status/${jobId}`),
  cancelDownload: (jobId) => apiFetch(`/scraper/cancel/${jobId}`, { method: 'POST' }),

  // Studio
  startStudio: (request) => apiFetch('/studio/start', {
    method: 'POST',
    body: JSON.stringify(request),
  }),
  getStudioStatus: (sessionId) => apiFetch(`/studio/status/${sessionId}`),
  continueStudio: (sessionId) => apiFetch(`/studio/continue/${sessionId}`, { method: 'POST' }),
  stopStudio: (sessionId) => apiFetch(`/studio/stop/${sessionId}`, { method: 'POST' }),

  // Analysis
  startAnalysis: (request) => apiFetch('/analysis/start', {
    method: 'POST',
    body: JSON.stringify(request),
  }),
  getAnalysisStatus: (jobId) => apiFetch(`/analysis/status/${jobId}`),
  cancelAnalysis: (jobId) => apiFetch(`/analysis/cancel/${jobId}`, { method: 'POST' }),

  // Videos
  getVideos: (params = {}) => {
    const query = new URLSearchParams(params).toString();
    return apiFetch(`/videos${query ? `?${query}` : ''}`);
  },
  getVideo: (videoId, outputDir = 'videos') =>
    apiFetch(`/videos/${videoId}?output_dir=${encodeURIComponent(outputDir)}`),
};

export default useApi;
