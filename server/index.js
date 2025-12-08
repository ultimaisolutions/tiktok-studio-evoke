/**
 * Express proxy server for TikTok Studio UI
 *
 * Proxies API requests from Vite frontend to FastAPI backend
 * and handles WebSocket connections for real-time progress updates.
 */

import express from 'express';
import cors from 'cors';
import { createProxyMiddleware } from 'http-proxy-middleware';

const app = express();
const PORT = process.env.PORT || 3001;
const API_TARGET = process.env.API_TARGET || 'http://localhost:8000';

// CORS configuration for development
app.use(cors({
  origin: [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
  ],
  credentials: true,
}));

// Note: Do NOT use express.json() here - it consumes the request body stream
// before http-proxy-middleware can forward it to FastAPI

// Logging middleware
app.use((req, res, next) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${req.method} ${req.url}`);
  next();
});

// Proxy API requests to FastAPI backend
app.use('/api', createProxyMiddleware({
  target: API_TARGET,
  changeOrigin: true,
  // Express strips /api when mounting, so we need to add it back
  pathRewrite: (path) => `/api${path}`,
  onProxyReq: (proxyReq, req, res) => {
    // Log proxied requests
    console.log(`  → Proxying to ${API_TARGET}/api${req.url}`);
  },
  onProxyRes: (proxyRes, req, res) => {
    console.log(`  ← Response: ${proxyRes.statusCode}`);
  },
  onError: (err, req, res) => {
    console.error('Proxy error:', err.message);
    if (!res.headersSent) {
      res.status(502).json({
        error: 'Backend unavailable',
        message: 'FastAPI server is not responding. Make sure it is running on port 8000.',
        target: API_TARGET,
      });
    }
  },
}));

// WebSocket proxy - store reference to subscribe to upgrade events later
const wsProxy = createProxyMiddleware({
  target: API_TARGET.replace('http', 'ws'),
  changeOrigin: true,
  ws: true,
  // Express strips /ws when mounting, so we need to add it back
  pathRewrite: (path) => `/ws${path}`,
  onError: (err, req, res) => {
    console.error('WebSocket proxy error:', err.message);
  },
});
app.use('/ws', wsProxy);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    proxy_target: API_TARGET,
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'TikTok Studio Proxy Server',
    version: '1.0.0',
    api_target: API_TARGET,
    endpoints: {
      api: '/api/*',
      websocket: '/ws/{job_id}',
      health: '/health',
    },
  });
});

// Start server
const server = app.listen(PORT, () => {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  TikTok Studio Proxy Server');
  console.log('═══════════════════════════════════════════════════════════');
  console.log(`  Server:    http://localhost:${PORT}`);
  console.log(`  Proxying:  ${API_TARGET}`);
  console.log('═══════════════════════════════════════════════════════════');
  console.log('');
});

// Subscribe WebSocket proxy to server upgrade events
// This is required for http-proxy-middleware v3.x to handle WebSocket connections
server.on('upgrade', (req, socket, head) => {
  console.log(`[WS] Upgrade request: ${req.url}`);
  wsProxy.upgrade(req, socket, head);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Shutting down proxy server...');
  server.close(() => {
    process.exit(0);
  });
});
