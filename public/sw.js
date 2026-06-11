const CACHE_NAME = 'potatorag-cache-v3';
const ASSETS = [
  '/',
  '/index.html',
  '/style.css',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  '/js-vector-store.js',
  '/wasm-polar-store-browser.js',
  '/wasm-vector-store-browser.js',
  '/rust_polar.wasm',
  '/vendor/transformers.min.js'
];

// Install Event - Pre-cache assets
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Pre-caching static assets');
      return cache.addAll(ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate Event - Clean up old caches
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[Service Worker] Removing old cache:', key);
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch Event - Cache First for assets, Network Only for APIs
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Bypass API requests to ensure real-time query/ingest operations
  if (url.pathname.startsWith('/api/')) {
    e.respondWith(fetch(e.request));
    return;
  }

  // Cache First Strategy for static assets
  e.respondWith(
    caches.match(e.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(e.request).then((networkResponse) => {
        // Cache newly requested static assets (e.g. fonts)
        if (networkResponse.status === 200 && e.request.method === 'GET') {
          const cacheCopy = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(e.request, cacheCopy);
          });
        }
        return networkResponse;
      });
    }).catch(() => {
      // Offline fallback if network fails
      if (e.request.mode === 'navigate') {
        return caches.match('/index.html');
      }
    })
  );
});
