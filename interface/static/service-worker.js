const RUNTIME_REVISION_PATTERN = /^[0-9a-f]{64}$/;
const requestedRevision = new URL(self.location.href).searchParams.get('_aura_runtime') || '';
const SHELL_REVISION = RUNTIME_REVISION_PATTERN.test(requestedRevision)
  ? requestedRevision
  : '';
const CACHE_NAMESPACE = 'aura-runtime-shell-';
const CACHE_NAME = `${CACHE_NAMESPACE}${SHELL_REVISION || 'unverified'}`;
let runtimeTrustActive = Boolean(SHELL_REVISION);
const REVISION_ADDRESSED_ASSETS = [
  '/static/icon.svg',
  '/static/icon-192.png',
  '/static/icon-512.png'
];
const REVISION_ADDRESSED_PATHS = new Set(REVISION_ADDRESSED_ASSETS);
const RUNTIME_CACHE_RETENTION = 4;
const revisionAddressedUrl = (path) => {
  if (!SHELL_REVISION) return path;
  const url = new URL(path, self.location.origin);
  url.searchParams.set('_aura_runtime', SHELL_REVISION);
  return `${url.pathname}${url.search}`;
};
const LIVE_SHELL_PATHS = new Set([
  '/',
  '/static/index.html',
  '/static/design_tokens.css',
  '/static/motion_design.css',
  '/static/error_banner.css',
  '/static/aura.css',
  '/static/voice_mode.css',
  '/static/presence_design.css',
  '/static/vendor/vis-network.min.js',
  '/static/shell_lexicon.js',
  '/static/error_banner.js',
  '/static/sound_design.js',
  '/static/perf_collector.js',
  '/static/aura.js',
  '/static/voice_mode.js',
  '/static/manifest.json',
  '/static/service-worker.js',
  '/static/aura_avatar.svg',
  '/static/vendor/fonts/fredoka-variable-latin.woff2',
  '/static/vendor/fonts/ibm-plex-mono-400-latin.woff2',
  '/static/vendor/fonts/ibm-plex-mono-500-latin.woff2',
  '/static/vendor/fonts/ibm-plex-mono-600-latin.woff2',
  '/static/voice-processor.js'
]);
const ASSETS_TO_CACHE = SHELL_REVISION
  ? [...new Set([
      ...[...LIVE_SHELL_PATHS].filter((path) => (
        path !== '/' && path !== '/static/service-worker.js'
      )),
      ...REVISION_ADDRESSED_ASSETS,
    ])].map(revisionAddressedUrl)
  : [];

// ── Install: Cache core assets ──
self.addEventListener('install', (event) => {
  event.waitUntil(
    SHELL_REVISION
      ? caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS_TO_CACHE))
      : Promise.resolve()
  );
});

// ── Activate: preserve live prior revisions and only claim compatible tabs ──
self.addEventListener('activate', (event) => {
  if (!SHELL_REVISION) {
    event.waitUntil(self.registration.unregister());
    return;
  }
  event.waitUntil((async () => {
    const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    const liveRevisions = new Set();
    let unmarkedClient = false;
    let conflictingClient = false;
    clients.forEach((client) => {
      try {
        const marker = new URL(client.url).searchParams.get('_aura_runtime') || '';
        if (!RUNTIME_REVISION_PATTERN.test(marker)) {
          unmarkedClient = true;
          return;
        }
        liveRevisions.add(marker);
        if (marker !== SHELL_REVISION) conflictingClient = true;
      } catch (_err) {
        unmarkedClient = true;
        conflictingClient = true;
      }
    });
    if (!unmarkedClient) {
      const keys = await caches.keys();
      const revisionCaches = keys.filter((key) => (
        key.startsWith(CACHE_NAMESPACE)
        && RUNTIME_REVISION_PATTERN.test(key.slice(CACHE_NAMESPACE.length))
      ));
      const retained = new Set(revisionCaches.slice(-RUNTIME_CACHE_RETENTION));
      retained.add(CACHE_NAME);
      liveRevisions.forEach((marker) => retained.add(`${CACHE_NAMESPACE}${marker}`));
      await Promise.all(
        revisionCaches
          .filter((key) => !retained.has(key))
          .map((key) => caches.delete(key))
      );
    }
    if (!conflictingClient) await self.clients.claim();
  })());
});

// The shell is served by a runtime on this same machine, so a shell asset
// that has not answered in this long means the runtime is not serving it —
// not that the network is slow. Past that point the cached copy is the better
// answer and waiting only delays the window coming up.
const SHELL_NETWORK_BUDGET_MS = 2000;

const fetchWithinShellBudget = async (requestUrl) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SHELL_NETWORK_BUDGET_MS);
  try {
    return await fetch(requestUrl, {
      cache: 'reload',
      credentials: 'same-origin',
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
};

// ── Fetch: Network-first with cache fallback ──
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET') return;
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/ws/')) return;
  if (url.pathname.startsWith('/api/')) return;
  if (!SHELL_REVISION || !runtimeTrustActive) return;
  const requestRevision = url.searchParams.get('_aura_runtime') || '';
  if (requestRevision && requestRevision !== SHELL_REVISION) return;
  let referrerRevision = '';
  try {
    const referrer = event.request.referrer ? new URL(event.request.referrer) : null;
    referrerRevision = referrer?.origin === self.location.origin
      ? referrer.searchParams.get('_aura_runtime') || ''
      : '';
    if (referrerRevision && referrerRevision !== SHELL_REVISION) return;
  } catch (_err) {
    return;
  }
  const revisionRoot = Boolean(
    url.pathname === '/'
    && event.request.mode === 'navigate'
    && requestRevision === SHELL_REVISION
  );
  const revisionBound = revisionRoot || Boolean(
    url.pathname !== '/'
    &&
    url.pathname !== '/static/service-worker.js'
    && (requestRevision === SHELL_REVISION || referrerRevision === SHELL_REVISION)
    && (LIVE_SHELL_PATHS.has(url.pathname) || REVISION_ADDRESSED_PATHS.has(url.pathname))
  );
  if (revisionBound) {
    const revisionUrl = revisionAddressedUrl(
      revisionRoot ? '/static/index.html' : url.pathname
    );
    event.respondWith((async () => {
      const cache = await caches.open(CACHE_NAME);
      // Network first — what the heading above this listener has always said,
      // and what the code below it did not do.
      //
      // Cache-first pinned a window to one revision permanently. This worker
      // is bound to the revision in its own script URL, and the only thing
      // that could move a tab onto a newer worker was page JS that this same
      // worker was serving out of that frozen cache. Reloading could not
      // escape it: the reload was answered from the same cache. Measured live
      // 2026-08-03 — a desktop window sat on a revision from hours earlier
      // through four runtime restarts and three revision changes, showing
      // "Conversation lane initializing" while /api/health reported
      // conversation_ready: true.
      //
      // The immutable server accepts only snapshots held by this runtime.
      // An unmarked bootstrap document bypasses this worker above, allowing a
      // new runtime to load its current shell before health binds the page to
      // a revision. A marked document remains byte-consistent here.
      try {
        const response = await fetchWithinShellBudget(revisionUrl);
        if (response && response.ok) {
          await cache.put(revisionUrl, response.clone());
          return response;
        }
      } catch (_err) {
        // Offline, aborted, or the runtime is down. Fall through to cache —
        // a stale shell beats no shell.
      }
      const cachedResponse = await cache.match(revisionUrl);
      if (cachedResponse) return cachedResponse;
      return fetch(revisionUrl, { cache: 'reload', credentials: 'same-origin' });
    })());
    return;
  }
});

// ── Push Notifications (from page via postMessage) ──
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'AURA_RETIRE_RUNTIME_SHELL') {
    runtimeTrustActive = false;
    event.waitUntil(
      caches.keys()
        .then((keys) => Promise.all(
          keys.filter((key) => key.startsWith(CACHE_NAMESPACE)).map((key) => caches.delete(key))
        ))
        .then(() => self.registration.unregister())
    );
    return;
  }
  if (event.data && event.data.type === 'SKIP_WAITING') {
    if (SHELL_REVISION && event.data.revision === SHELL_REVISION) {
      self.skipWaiting();
    }
    return;
  }

  if (event.data && event.data.type === 'AURA_REPLY') {
    const { title, body, tag } = event.data;
    // Only notify if no visible client is focused
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clients) => {
      const anyFocused = clients.some(c => c.focused || c.visibilityState === 'visible');
      if (!anyFocused) {
        self.registration.showNotification(title || 'Aura', {
          body: body || 'New message',
          icon: revisionAddressedUrl('/static/icon-192.png'),
          badge: revisionAddressedUrl('/static/icon-192.png'),
          tag: tag || 'aura-reply',
          renotify: true,
          vibrate: [100, 50, 100],
          data: { url: '/' },
          actions: [
            { action: 'open', title: 'Open Aura' },
            { action: 'dismiss', title: 'Dismiss' }
          ]
        });
      }
    });
  }

  // v5.1: Background mode — persistent notification keeps Aura alive
  if (event.data && event.data.type === 'AURA_BACKGROUND_MODE') {
    const enabled = event.data.enabled;
    if (enabled) {
      self.registration.showNotification('Aura is running', {
        body: 'Aura is active in the background. Tap to open.',
        icon: revisionAddressedUrl('/static/icon-192.png'),
        badge: revisionAddressedUrl('/static/icon-192.png'),
        tag: 'aura-background',
        silent: true,
        requireInteraction: true,
        data: { url: '/', background: true },
        actions: [
          { action: 'open', title: 'Open' },
          { action: 'stop', title: 'Stop Background' }
        ]
      });
    } else {
      // Close the persistent notification
      self.registration.getNotifications({ tag: 'aura-background' }).then(notifications => {
        notifications.forEach(n => n.close());
      });
    }
  }
});

// ── Notification click → focus or open app ──
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  if (event.action === 'dismiss') return;

  // v5.1: Stop background mode
  if (event.action === 'stop') {
    self.registration.getNotifications({ tag: 'aura-background' }).then(ns => ns.forEach(n => n.close()));
    return;
  }

  event.waitUntil(
    self.clients.matchAll({ type: 'window' }).then((clients) => {
      // Focus existing window if any
      for (const client of clients) {
        if ('focus' in client) return client.focus();
      }
      // Otherwise open new
      return self.clients.openWindow(event.notification.data?.url || '/');
    })
  );
});

// ── Background Sync: reconnect WebSocket when back online ──
self.addEventListener('sync', (event) => {
  if (event.tag === 'aura-reconnect') {
    event.waitUntil(
      fetch('/api/health/heartbeat').then(r => r.json()).then(data => {
        // Background sync — server alive
      }).catch(() => {
        // Background sync — server unreachable
      })
    );
  }
});

// ── Periodic Background Sync (keeps connection alive) ──
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'aura-heartbeat') {
    event.waitUntil(
      fetch('/api/health/heartbeat').then(r => r.json()).then(data => {
        // Heartbeat received
      })
    );
  }
});
