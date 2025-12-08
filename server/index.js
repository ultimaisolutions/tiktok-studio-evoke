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

// JSON body parsing
app.use(express.json());

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
  pathRewrite: {
    '^/api': '/api',
  },
  onProxyReq: (proxyReq, req, res) => {
    // Log proxied requests
    console.log(`  → Proxying to ${API_TARGET}${req.url}`);
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

// Proxy WebSocket connections
app.use('/ws', createProxyMiddleware({
  target: API_TARGET.replace('http', 'ws'),
  changeOrigin: true,
  ws: true,
  onError: (err, req, res) => {
    console.error('WebSocket proxy error:', err.message);
  },
}));

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

// Handle WebSocket upgrade for proxying
server.on('upgrade', (req, socket, head) => {
  console.log(`WebSocket upgrade request: ${req.url}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Shutting down proxy server...');
  server.close(() => {
    process.exit(0);
  });
});
