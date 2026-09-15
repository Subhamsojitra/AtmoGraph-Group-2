/**
 * WebSocket Transport Service
 * 
 * Provides an isolated, payload-agnostic WebSocket transport boundary
 * connecting to Santanu's confirmed endpoint (/api/v1/ws).
 * 
 * Strictly handles transport lifecycle concerns:
 *   - Connection open/handshake
 *   - Raw JSON message reception & delivery
 *   - Structured transport error handling
 *   - Clean disconnects & reconnect capability
 *   - Ping / Pong transport keepalive
 * 
 * STRICT BOUNDARY RULE:
 * This module does NOT parse, validate, or make assumptions about
 * ML/GNN prediction payload schemas. It delivers raw messages to the
 * prediction service boundary for future payload contract integration.
 */

/**
 * Resolves the WebSocket endpoint URL relative to the current browser host.
 * 
 * @param {string} path - The WebSocket route path (defaults to '/api/v1/ws')
 * @returns {string} The full ws:// or wss:// URL
 */
export function resolveWebSocketUrl(path = '/api/v1/ws') {
  if (typeof window === 'undefined' || !window.location) {
    return `ws://localhost:8000${path}`;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}

/**
 * WebSocketTransport manages the raw WebSocket lifecycle.
 */
export class WebSocketTransport {
  /**
   * @param {Object} options
   * @param {string} [options.url] - WebSocket URL (defaults to resolved /api/v1/ws)
   * @param {boolean} [options.autoReconnect=false] - Whether to attempt reconnect
   * @param {number} [options.reconnectInterval=5000] - Reconnection delay in ms
   */
  constructor(options = {}) {
    this.url = options.url || resolveWebSocketUrl();
    this.autoReconnect = options.autoReconnect ?? false;
    this.reconnectInterval = options.reconnectInterval || 5000;

    this.socket = null;
    this.status = 'idle'; // 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error'
    this.clientId = null;
    this.protocolVersion = null;

    this.listeners = {
      message: new Set(),
      status: new Set(),
      error: new Set(),
      connected: new Set(),
      close: new Set(),
    };

    this._reconnectTimer = null;
    this._isExplicitDisconnect = false;
  }

  /**
   * Subscribes a listener callback to an event.
   * 
   * @param {'message'|'status'|'error'|'connected'|'close'} event
   * @param {Function} callback
   * @returns {Function} Unsubscribe function
   */
  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].add(callback);
    }
    return () => {
      if (this.listeners[event]) {
        this.listeners[event].delete(callback);
      }
    };
  }

  /**
   * Emits an event to registered callbacks.
   * @private
   */
  _emit(event, ...args) {
    if (this.listeners[event]) {
      this.listeners[event].forEach((cb) => {
        try {
          cb(...args);
        } catch (err) {
          console.error(`WebSocket transport listener error [${event}]:`, err);
        }
      });
    }
  }

  /**
   * Updates the current transport status and notifies status listeners.
   * @private
   */
  _setStatus(newStatus) {
    if (this.status !== newStatus) {
      this.status = newStatus;
      this._emit('status', newStatus);
    }
  }

  /**
   * Initiates the WebSocket connection.
   */
  connect() {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this._isExplicitDisconnect = false;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }

    this._setStatus('connecting');

    try {
      this.socket = new WebSocket(this.url);

      this.socket.onopen = () => {
        this._setStatus('connected');
      };

      this.socket.onmessage = (event) => {
        this._handleRawMessage(event.data);
      };

      this.socket.onerror = (event) => {
        console.warn('WebSocket transport encountered a connection error:', event);
        this._setStatus('error');
        this._emit('error', event);
      };

      this.socket.onclose = (event) => {
        const wasClean = event.wasClean;
        this.socket = null;
        this._setStatus('disconnected');
        this._emit('close', { wasClean, code: event.code, reason: event.reason });

        if (!this._isExplicitDisconnect && this.autoReconnect) {
          this._reconnectTimer = setTimeout(() => {
            this.connect();
          }, this.reconnectInterval);
        }
      };
    } catch (err) {
      console.warn('WebSocket transport failed to initialize connection:', err);
      this._setStatus('error');
      this._emit('error', err);
    }
  }

  /**
   * Handles incoming text frames and parses JSON safely without assuming prediction schemas.
   * @private
   */
  _handleRawMessage(rawText) {
    let parsed;
    try {
      parsed = JSON.parse(rawText);
    } catch (err) {
      console.warn('WebSocket transport received non-JSON frame:', rawText, err);
      this._emit('error', { type: 'PARSE_ERROR', raw: rawText, error: err });
      return;
    }

    // Handle Module 15 transport lifecycle messages internally if present
    if (parsed && typeof parsed === 'object') {
      if (parsed.type === 'connected') {
        this.clientId = parsed.data?.client_id || null;
        this.protocolVersion = parsed.data?.protocol || null;
        this._emit('connected', parsed.data);
      }
    }

    // Deliver the raw, unmutated message to prediction boundary listeners
    this._emit('message', parsed);
  }

  /**
   * Sends a structured JSON payload to the server.
   * 
   * @param {Object} message - JSON-serializable message object
   * @returns {boolean} True if sent, false otherwise
   */
  send(message) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket transport cannot send message: socket is not open.', message);
      return false;
    }

    try {
      const payload = typeof message === 'string' ? message : JSON.stringify(message);
      this.socket.send(payload);
      return true;
    } catch (err) {
      console.error('WebSocket transport failed to send message:', err);
      return false;
    }
  }

  /**
   * Sends a transport ping message.
   * 
   * @param {Object} [data] - Optional echo data
   * @returns {boolean}
   */
  ping(data = null) {
    const payload = { type: 'ping' };
    if (data && typeof data === 'object') {
      payload.data = data;
    }
    return this.send(payload);
  }

  /**
   * Cleanly closes the WebSocket connection.
   */
  disconnect() {
    this._isExplicitDisconnect = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }

    if (this.socket) {
      try {
        this.socket.close(1000, 'Client disconnected');
      } catch (err) {
        console.warn('WebSocket transport error during disconnect:', err);
      }
      this.socket = null;
    }

    this._setStatus('disconnected');
  }

  /**
   * Returns current connection status.
   * @returns {'idle'|'connecting'|'connected'|'disconnected'|'error'}
   */
  getStatus() {
    return this.status;
  }
}

/**
 * Factory helper to instantiate a WebSocketTransport instance.
 * 
 * @param {Object} [options]
 * @returns {WebSocketTransport}
 */
export function createWebSocketTransport(options = {}) {
  return new WebSocketTransport(options);
}
