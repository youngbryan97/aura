const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const staticRoot = path.resolve(__dirname, '..', '..');
const revision = 'a'.repeat(64);

function read(relativePath) {
  return fs.readFileSync(path.join(staticRoot, relativePath), 'utf8');
}

function serviceWorkerHarness({
  scriptRevision = revision,
  clientUrls = [],
  putGate = null,
  cacheKeys = null,
} = {}) {
  const listeners = new Map();
  const opened = [];
  const deleted = [];
  const cacheEntries = new Map();
  const cache = {
    async addAll(urls) {
      for (const url of urls) cacheEntries.set(String(url), { cached: String(url) });
    },
    async match(request) {
      const key = typeof request === 'string' ? request : request.url;
      return cacheEntries.get(key);
    },
    async put(request, response) {
      if (putGate) await putGate.promise;
      const key = typeof request === 'string' ? request : request.url;
      cacheEntries.set(key, response);
    },
  };
  const caches = {
    async open(name) {
      opened.push(name);
      return cache;
    },
    async keys() {
      return cacheKeys || [
        'aura-runtime-shell-old',
        `aura-runtime-shell-${revision}`,
        'unrelated-application-cache',
      ];
    },
    async delete(name) {
      deleted.push(name);
      return true;
    },
  };
  let skipWaitingCalls = 0;
  let claimCalls = 0;
  let unregisterCalls = 0;
  const networkCalls = [];
  const revisionQuery = scriptRevision ? `?_aura_runtime=${scriptRevision}` : '';
  const self = {
    location: {
      href: `http://127.0.0.1:8000/static/service-worker.js${revisionQuery}`,
      origin: 'http://127.0.0.1:8000',
    },
    addEventListener(name, callback) {
      listeners.set(name, callback);
    },
    skipWaiting() {
      skipWaitingCalls += 1;
    },
    clients: {
      async matchAll() {
        return clientUrls.map((url) => ({ url }));
      },
      async claim() {
        claimCalls += 1;
      },
    },
    registration: {
      async unregister() {
        unregisterCalls += 1;
        return true;
      },
    },
  };
  const context = {
    URL,
    Promise,
    AbortController,
    clearTimeout,
    setTimeout,
    caches,
    self,
    fetch: async (request) => {
      networkCalls.push(String(request));
      return { ok: true, clone() { return this; } };
    },
  };
  vm.runInNewContext(read('service-worker.js'), context, {
    filename: 'service-worker.js',
  });
  return {
    cacheEntries,
    deleted,
    listeners,
    opened,
    get claimCalls() { return claimCalls; },
    networkCalls,
    get skipWaitingCalls() { return skipWaitingCalls; },
    get unregisterCalls() { return unregisterCalls; },
  };
}

test('service worker binds cache and assets to the full revision without deleting rollback state', async () => {
  const harness = serviceWorkerHarness();
  let installPromise;
  harness.listeners.get('install')({ waitUntil(value) { installPromise = value; } });
  await installPromise;

  assert.ok(harness.opened.includes(`aura-runtime-shell-${revision}`));
  assert.equal(harness.cacheEntries.size, 24);
  for (const path of [
    '/static/index.html',
    '/static/aura.js',
    '/static/aura.css',
    '/static/icon-192.png',
  ]) {
    assert.ok(harness.cacheEntries.has(`${path}?_aura_runtime=${revision}`), path);
  }

  let activatePromise;
  harness.listeners.get('activate')({ waitUntil(value) { activatePromise = value; } });
  await activatePromise;
  assert.deepEqual(harness.deleted, []);
  assert.equal(harness.claimCalls, 1);

  harness.listeners.get('message')({
    data: { type: 'SKIP_WAITING', revision: 'b'.repeat(64) },
  });
  assert.equal(harness.skipWaitingCalls, 0);
  harness.listeners.get('message')({
    data: { type: 'SKIP_WAITING', revision },
  });
  assert.equal(harness.skipWaitingCalls, 1);

  let shellResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/static/aura.js',
      mode: 'same-origin',
      destination: 'script',
      referrer: `http://127.0.0.1:8000/?_aura_runtime=${revision}`,
    },
    respondWith(value) { shellResponse = value; },
  });
  assert.ok(shellResponse, 'service worker did not revision-bind shell JS');
  assert.equal((await shellResponse).ok, true);
  assert.deepEqual(
    harness.networkCalls,
    [`/static/aura.js?_aura_runtime=${revision}`],
  );
  assert.equal(
    harness.cacheEntries.get(`/static/aura.js?_aura_runtime=${revision}`).ok,
    true,
  );

  let iconResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/static/icon-192.png',
      mode: 'same-origin',
      destination: 'image',
      referrer: `http://127.0.0.1:8000/?_aura_runtime=${revision}`,
    },
    respondWith(value) { iconResponse = value; },
  });
  assert.equal((await iconResponse).ok, true);
  assert.equal(harness.networkCalls.length, 2);

  let futureModuleResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/static/future-shell-module.js',
      mode: 'same-origin',
      destination: 'script',
      referrer: `http://127.0.0.1:8000/?_aura_runtime=${revision}`,
    },
    respondWith(value) { futureModuleResponse = value; },
  });
  assert.equal(futureModuleResponse, undefined);
  assert.equal(harness.networkCalls.length, 2);

  let futureImageResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/static/future-shell-image.png',
      mode: 'same-origin',
      destination: 'image',
      referrer: `http://127.0.0.1:8000/?_aura_runtime=${revision}`,
    },
    respondWith(value) { futureImageResponse = value; },
  });
  assert.equal(futureImageResponse, undefined);
  assert.equal(harness.networkCalls.length, 2);

  let rootNavigationResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/',
      mode: 'navigate',
      destination: 'document',
      referrer: '',
    },
    respondWith(value) { rootNavigationResponse = value; },
  });
  assert.equal(rootNavigationResponse, undefined);

  let revisionNavigationResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: `http://127.0.0.1:8000/?_aura_runtime=${revision}`,
      mode: 'navigate',
      destination: 'document',
      referrer: '',
    },
    respondWith(value) { revisionNavigationResponse = value; },
  });
  assert.equal((await revisionNavigationResponse).ok, true);
  assert.equal(
    harness.networkCalls.at(-1),
    `/static/index.html?_aura_runtime=${revision}`,
  );
});

test('activation preserves revision A tabs when revision B becomes active', async () => {
  const revisionB = 'b'.repeat(64);
  const harness = serviceWorkerHarness({
    scriptRevision: revisionB,
    clientUrls: [
      `http://127.0.0.1:8000/?_aura_runtime=${revision}`,
      `http://127.0.0.1:8000/?_aura_runtime=${revisionB}`,
    ],
  });
  let activatePromise;
  harness.listeners.get('activate')({ waitUntil(value) { activatePromise = value; } });
  await activatePromise;
  assert.equal(harness.claimCalls, 0);
  assert.deepEqual(harness.deleted, []);
});

test('activation bounds rollback caches while preserving every live-tab revision', async () => {
  const revisions = ['1', '2', '3', '4', '5', '6'].map((value) => value.repeat(64));
  const current = revisions[5];
  const harness = serviceWorkerHarness({
    scriptRevision: current,
    clientUrls: [`http://127.0.0.1:8000/?_aura_runtime=${revisions[0]}`],
    cacheKeys: revisions.map((value) => `aura-runtime-shell-${value}`),
  });
  let activation;
  harness.listeners.get('activate')({ waitUntil(value) { activation = value; } });
  await activation;
  assert.deepEqual(harness.deleted, [`aura-runtime-shell-${revisions[1]}`]);
  assert.equal(harness.claimCalls, 0);
});

test('worker ignores cross-origin collisions and all non-shell private resources', () => {
  const harness = serviceWorkerHarness();
  const requests = [
    ['https://attacker.example/static/aura.js', 'script'],
    ['http://127.0.0.1:8000/data/uploads/private.png', 'image'],
    ['http://127.0.0.1:8000/generated/private.json', ''],
    ['http://127.0.0.1:8000/api/health', ''],
  ];
  for (const [url, destination] of requests) {
    let response;
    harness.listeners.get('fetch')({
      request: { method: 'GET', url, destination, mode: 'same-origin', referrer: '' },
      respondWith(value) { response = value; },
    });
    assert.equal(response, undefined, url);
  }
  let crossRevisionResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/static/aura.js',
      destination: 'script',
      mode: 'same-origin',
      referrer: `http://127.0.0.1:8000/?_aura_runtime=${'b'.repeat(64)}`,
    },
    respondWith(value) { crossRevisionResponse = value; },
  });
  assert.equal(crossRevisionResponse, undefined);
  assert.deepEqual(harness.networkCalls, []);
});

test('stale controller cannot bind a fresh unmarked document to its old assets', () => {
  const harness = serviceWorkerHarness();
  let shellResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/static/aura.js',
      destination: 'script',
      mode: 'same-origin',
      referrer: 'http://127.0.0.1:8000/',
    },
    respondWith(value) { shellResponse = value; },
  });

  assert.equal(shellResponse, undefined);
  assert.deepEqual(harness.networkCalls, []);
});

test('revision cache write remains inside the fetch event lifetime', async () => {
  let releasePut;
  const putGate = { promise: new Promise((resolve) => { releasePut = resolve; }) };
  const harness = serviceWorkerHarness({ putGate });
  let shellResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/static/aura.js',
      mode: 'same-origin',
      destination: 'script',
      referrer: `http://127.0.0.1:8000/?_aura_runtime=${revision}`,
    },
    respondWith(value) { shellResponse = value; },
  });
  let settled = false;
  shellResponse.finally(() => { settled = true; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(settled, false);
  releasePut();
  await shellResponse;
  assert.equal(settled, true);
  assert.ok(harness.cacheEntries.has(`/static/aura.js?_aura_runtime=${revision}`));
});

test('trust retirement disables interception, purges Aura caches, and unregisters', async () => {
  const harness = serviceWorkerHarness();
  let retirement;
  harness.listeners.get('message')({
    data: { type: 'AURA_RETIRE_RUNTIME_SHELL', reason: 'test' },
    waitUntil(value) { retirement = value; },
  });
  await retirement;
  assert.deepEqual(
    harness.deleted.sort(),
    [`aura-runtime-shell-${revision}`, 'aura-runtime-shell-old'].sort(),
  );
  assert.equal(harness.unregisterCalls, 1);
  let response;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/static/aura.js',
      mode: 'same-origin',
      destination: 'script',
      referrer: `http://127.0.0.1:8000/?_aura_runtime=${revision}`,
    },
    respondWith(value) { response = value; },
  });
  assert.equal(response, undefined);
});

test('notification icons are revision addressed instead of stable immutable URLs', () => {
  const worker = read('service-worker.js');
  assert.doesNotMatch(worker, /icon:\s*'\/static\/icon-192\.png'/);
  assert.doesNotMatch(worker, /badge:\s*'\/static\/icon-192\.png'/);
  assert.match(worker, /icon: revisionAddressedUrl\('\/static\/icon-192\.png'\)/);
  assert.match(worker, /badge: revisionAddressedUrl\('\/static\/icon-192\.png'\)/);
});

test('unrevisioned worker cannot create or evict trusted caches or claim clients', async () => {
  const harness = serviceWorkerHarness({ scriptRevision: '' });
  let installPromise;
  harness.listeners.get('install')({ waitUntil(value) { installPromise = value; } });
  await installPromise;
  assert.deepEqual(harness.opened, []);

  let activatePromise;
  harness.listeners.get('activate')({ waitUntil(value) { activatePromise = value; } });
  await activatePromise;
  assert.deepEqual(harness.deleted, []);
  assert.equal(harness.claimCalls, 0);
  assert.equal(harness.unregisterCalls, 1);

  let shellResponse;
  harness.listeners.get('fetch')({
    request: {
      method: 'GET',
      url: 'http://127.0.0.1:8000/static/aura.js',
      mode: 'same-origin',
      destination: 'script',
      referrer: '',
    },
    respondWith(value) { shellResponse = value; },
  });
  assert.equal(shellResponse, undefined);
  assert.deepEqual(harness.networkCalls, []);
});

function auraServiceWorkerFunctions(context) {
  const source = read('aura.js');
  const start = source.indexOf('function serviceWorkerRegistrationIsCurrent(revision');
  const end = source.indexOf('function reconcileRuntimeShellRevision(payload)', start);
  assert.ok(start >= 0 && end > start);
  vm.runInNewContext(source.slice(start, end), context, { filename: 'aura.js#sw' });
}

function auraRuntimeRevisionFunctions(context) {
  const source = read('aura.js');
  const start = source.indexOf('function verifiedRuntimeRevision(payload)');
  const end = source.indexOf('async function pollHealth()', start);
  assert.ok(start >= 0 && end > start);
  vm.runInNewContext(source.slice(start, end), context, { filename: 'aura.js#revision' });
}

test('fresh trust loss retires only Aura workers and purges revision state', async () => {
  const values = new Map([
    ['aura.runtime_revision', JSON.stringify({
      schema: 'aura.runtime_revision.client.v2',
      revision,
      generation: 7,
      captured_at_unix: 100,
    })],
  ]);
  const retired = [];
  const messages = [];
  const deletedCaches = [];
  const registrations = [
    {
      name: 'aura',
      scope: 'http://127.0.0.1:8000/',
      active: {
        scriptURL: `http://127.0.0.1:8000/static/service-worker.js?_aura_runtime=${revision}`,
        postMessage(message) { messages.push(message); },
      },
      async unregister() { retired.push(this.name); return true; },
    },
    {
      name: 'unrelated',
      scope: 'http://127.0.0.1:8000/',
      active: { scriptURL: 'http://127.0.0.1:8000/static/other-worker.js' },
      async unregister() { retired.push(this.name); return true; },
    },
  ];
  const context = {
    URL,
    Promise,
    Date,
    Object,
    console: { warn() {}, error() {} },
    RUNTIME_REVISION_STORAGE_KEY: 'aura.runtime_revision',
    RUNTIME_REVISION_RELOAD_STORAGE_KEY: 'aura.runtime_revision_reload',
    RUNTIME_REVISION_RECORD_SCHEMA: 'aura.runtime_revision.client.v2',
    RUNTIME_REVISION_RELOAD_LIMIT: 2,
    SERVICE_WORKER_REGISTRATION_RETRY_MAX_MS: 30000,
    sessionStorage: {
      getItem(key) { return values.get(key) || null; },
      setItem(key, value) { values.set(key, String(value)); },
      removeItem(key) { values.delete(key); },
    },
    window: {
      location: {
        origin: 'http://127.0.0.1:8000',
        href: 'http://127.0.0.1:8000/',
      },
    },
    navigator: {
      serviceWorker: {
        async getRegistrations() { return registrations; },
      },
    },
    caches: {
      async keys() { return ['aura-runtime-shell-old', 'unrelated-cache']; },
      async delete(key) { deletedCaches.push(key); return true; },
    },
    state: {
      runtimeRevision: revision,
      runtimeRevisionGeneration: 7,
      runtimeRevisionCapturedAtUnix: 100,
      runtimeRevisionReloading: false,
      runtimeRevisionTrust: 'trusted',
      runtimeShellRetirementPromise: null,
      runtimeRevisionReloadAttempts: {},
      serviceWorkerRevision: revision,
      serviceWorkerRegistrationTarget: revision,
      serviceWorkerRegistrationPromise: Promise.resolve(),
      serviceWorkerRegistrationEpoch: 1,
      serviceWorkerRegistrationFailures: 0,
      serviceWorkerRegistrationRetryAt: 0,
    },
  };
  auraRuntimeRevisionFunctions(context);

  const directLaunch = {
    runtime_revision: {
      schema: 'aura.runtime_revision.v2',
      required: false,
      verified: false,
      revision_token: '',
    },
    health_read_model: {
      fresh: true,
      expired: false,
      snapshot_generation: 8,
      captured_at_unix: 110,
    },
  };
  assert.equal(context.reconcileRuntimeShellRevision(directLaunch), false);
  const retirement = context.state.runtimeShellRetirementPromise;
  assert.ok(retirement);
  await retirement;

  assert.deepEqual(retired, ['aura']);
  assert.equal(messages[0].type, 'AURA_RETIRE_RUNTIME_SHELL');
  assert.deepEqual(deletedCaches, ['aura-runtime-shell-old']);
  assert.equal(values.has('aura.runtime_revision'), false);
  assert.equal(context.state.runtimeRevisionTrust, 'not_required');
  assert.equal(context.state.serviceWorkerRegistrationTarget, null);

  context.state.runtimeRevision = revision;
  context.state.runtimeRevisionTrust = 'trusted';
  context.state.serviceWorkerRegistrationTarget = revision;
  values.set('aura.runtime_revision', revision);
  const legacyServer = {
    health_read_model: {
      fresh: true,
      expired: false,
      snapshot_generation: 9,
      captured_at_unix: 120,
    },
  };
  assert.equal(context.reconcileRuntimeShellRevision(legacyServer), false);
  await context.state.runtimeShellRetirementPromise;
  assert.deepEqual(retired, ['aura', 'aura']);
});

test('payloads without revision evidence preserve the last authoritative verdict', () => {
  const context = {
    state: { runtimeRevisionTrust: 'trusted' },
  };
  const source = read('aura.js');
  const start = source.indexOf('function runtimeRevisionPolicySatisfied(payload)');
  const end = source.indexOf('function storedRuntimeRevisionRecord()', start);
  vm.runInNewContext(source.slice(start, end), context);
  assert.equal(context.runtimeRevisionPolicyBlocker({ status: 'healthy' }), '');
  context.state.runtimeRevisionTrust = 'untrusted';
  assert.equal(
    context.runtimeRevisionPolicyBlocker({ status: 'healthy' }),
    'runtime_revision_unverified',
  );
});

test('health polling passes the complete revision-bearing payload to splash state', () => {
  const source = read('aura.js');
  const poll = source.slice(
    source.indexOf('async function pollHealth()'),
    source.indexOf('function scheduleHealthPoll', source.indexOf('async function pollHealth()')),
  );
  assert.match(poll, /syncSplashState\(d\)/);
  assert.doesNotMatch(poll, /syncSplashState\(\{\s*telemetry:/);
});

test('page observes existing and new installers before and across update', async () => {
  const activation = [];
  const navigator = { serviceWorker: { controller: {} } };
  const context = {
    URL,
    Promise,
    console: { warn() {}, error() {} },
    navigator,
    requestServiceWorkerActivation(worker, workerRevision) {
      activation.push([worker.name, workerRevision]);
      return true;
    },
    state: {
      serviceWorkerRegistrationTarget: revision,
      serviceWorkerRegistrationEpoch: 1,
    },
  };
  auraServiceWorkerFunctions(context);

  function worker(name, state = 'installing') {
    const callbacks = new Map();
    return {
      name,
      state,
      scriptURL: `http://127.0.0.1:8000/static/service-worker.js?_aura_runtime=${revision}`,
      addEventListener(eventName, callback) { callbacks.set(eventName, callback); },
      callbacks,
    };
  }

  const existing = worker('existing');
  const discovered = worker('discovered');
  const waiting = worker('waiting', 'installed');
  const registrationCallbacks = new Map();
  const registration = {
    installing: existing,
    waiting: null,
    addEventListener(name, callback) { registrationCallbacks.set(name, callback); },
    async update() {
      assert.ok(registrationCallbacks.has('updatefound'));
      assert.ok(existing.callbacks.has('statechange'));
      existing.state = 'installed';
      existing.callbacks.get('statechange')();
      this.installing = discovered;
      registrationCallbacks.get('updatefound')();
      assert.ok(discovered.callbacks.has('statechange'));
      discovered.state = 'installed';
      discovered.callbacks.get('statechange')();
      this.waiting = waiting;
    },
  };

  await context.refreshServiceWorkerRegistration(registration, revision);

  assert.deepEqual(activation, [
    ['existing', revision],
    ['discovered', revision],
    ['waiting', revision],
  ]);
});

test('registration URL carries the complete revision token', async () => {
  const registrations = [];
  const registration = {
    installing: null,
    waiting: null,
    addEventListener() {},
    async update() {},
  };
  const context = {
    URL,
    Promise,
    console: { warn() {}, error() {} },
    navigator: {
      serviceWorker: {
        controller: {},
        async register(url, options) {
          registrations.push([url, options]);
          return registration;
        },
      },
    },
    requestServiceWorkerActivation() { return true; },
    state: {
      serviceWorkerRevision: null,
      serviceWorkerRegistrationTarget: null,
      serviceWorkerRegistrationPromise: null,
      serviceWorkerRegistrationEpoch: 0,
      serviceWorkerRegistrationFailures: 0,
      serviceWorkerRegistrationRetryAt: 0,
    },
  };
  auraServiceWorkerFunctions(context);

  await context.registerRevisionServiceWorker(revision);

  assert.equal(registrations.length, 1);
  assert.equal(
    registrations[0][0],
    `/static/service-worker.js?_aura_runtime=${revision}`,
  );
  assert.equal(registrations[0][1].updateViaCache, 'none');
  assert.equal(registrations[0][1].scope, '/');
  assert.equal(context.state.serviceWorkerRevision, revision);
});

test('overlapping registrations keep the newest revision authoritative', async () => {
  const newerRevision = 'b'.repeat(64);
  const release = new Map();
  const registrations = new Map();
  const makeRegistration = (workerRevision) => ({
    installing: null,
    waiting: null,
    addEventListener() {},
    update() {
      return new Promise((resolve) => release.set(workerRevision, resolve));
    },
  });
  registrations.set(revision, makeRegistration(revision));
  registrations.set(newerRevision, makeRegistration(newerRevision));
  const context = {
    URL,
    Promise,
    console: { warn() {}, error() {} },
    window: { location: { origin: 'http://127.0.0.1:8000' } },
    navigator: {
      serviceWorker: {
        controller: {},
        async getRegistrations() { return []; },
        async register(url) {
          const workerRevision = new URL(url, 'http://127.0.0.1:8000')
            .searchParams.get('_aura_runtime');
          return registrations.get(workerRevision);
        },
      },
    },
    requestServiceWorkerActivation() { return true; },
    state: {
      serviceWorkerRevision: null,
      serviceWorkerRegistrationTarget: null,
      serviceWorkerRegistrationPromise: null,
    },
  };
  auraServiceWorkerFunctions(context);

  const older = context.registerRevisionServiceWorker(revision);
  const newer = context.registerRevisionServiceWorker(newerRevision);
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(release.has(newerRevision));
  assert.equal(release.has(revision), false);

  release.get(newerRevision)();
  await newer;
  assert.equal(context.state.serviceWorkerRevision, newerRevision);

  await older;
  assert.equal(context.state.serviceWorkerRevision, newerRevision);
  assert.equal(context.state.serviceWorkerRegistrationTarget, newerRevision);
  assert.equal(context.state.serviceWorkerRegistrationPromise, newer);
});

test('a transient registration failure clears the memoized promise and later retries', async () => {
  let attempts = 0;
  let now = 1000;
  const registration = {
    installing: null,
    waiting: null,
    addEventListener() {},
    async update() {},
  };
  const context = {
    URL,
    Promise,
    Date: { now: () => now },
    SERVICE_WORKER_REGISTRATION_RETRY_MAX_MS: 30000,
    console: { warn() {}, error() {} },
    window: { location: { origin: 'http://127.0.0.1:8000' } },
    navigator: {
      serviceWorker: {
        controller: {},
        async getRegistrations() { return []; },
        async register() {
          attempts += 1;
          if (attempts === 1) throw new Error('transient');
          return registration;
        },
      },
    },
    requestServiceWorkerActivation() { return true; },
    state: {
      serviceWorkerRevision: null,
      serviceWorkerRegistrationTarget: null,
      serviceWorkerRegistrationPromise: null,
      serviceWorkerRegistrationEpoch: 0,
      serviceWorkerRegistrationFailures: 0,
      serviceWorkerRegistrationRetryAt: 0,
    },
  };
  auraServiceWorkerFunctions(context);

  assert.equal(await context.registerRevisionServiceWorker(revision), null);
  assert.equal(context.state.serviceWorkerRegistrationPromise, null);
  assert.equal(attempts, 1);
  assert.equal(await context.registerRevisionServiceWorker(revision), null);
  assert.equal(attempts, 1);

  now = context.state.serviceWorkerRegistrationRetryAt + 1;
  assert.equal(await context.registerRevisionServiceWorker(revision), registration);
  assert.equal(attempts, 2);
  assert.equal(context.state.serviceWorkerRevision, revision);
});

test('superseded registration cannot activate its stale waiting worker', async () => {
  const newerRevision = 'b'.repeat(64);
  const activations = [];
  const staleWaiting = {
    scriptURL: `http://127.0.0.1:8000/static/service-worker.js?_aura_runtime=${revision}`,
    postMessage(message) { activations.push(message); },
  };
  const context = {
    URL,
    Promise,
    console: { warn() {}, error() {} },
    navigator: { serviceWorker: { controller: {} } },
    requestServiceWorkerActivation(worker, workerRevision) {
      if (
        context.state.serviceWorkerRegistrationTarget
        && context.state.serviceWorkerRegistrationTarget !== workerRevision
      ) return false;
      worker.postMessage({ type: 'SKIP_WAITING', revision: workerRevision });
      return true;
    },
    state: {
      serviceWorkerInstallers: new WeakMap(),
      serviceWorkerRegistrationTarget: newerRevision,
    },
  };
  auraServiceWorkerFunctions(context);
  const registration = {
    installing: null,
    waiting: staleWaiting,
    addEventListener() {},
    async update() {},
  };

  await context.refreshServiceWorkerRegistration(registration, revision);

  assert.deepEqual(activations, []);
});

test('first controlling revision triggers a guarded reload instead of leaving old bytes live', () => {
  const block = source => source.slice(
    source.indexOf("navigator.serviceWorker.addEventListener('controllerchange'"),
    source.indexOf("window.addEventListener('load'", source.indexOf("navigator.serviceWorker.addEventListener('controllerchange'")),
  );
  const handler = block(read('aura.js'));
  assert.match(handler, /if \(!swHadController\) \{\s*swHadController = true;\s*\}/);
  assert.doesNotMatch(handler, /swHadController = true;\s*return;/);
  assert.match(handler, /requestGuardedShellReload/);
});

test('legacy static Aura worker is retired without touching unrelated registrations', async () => {
  const retired = [];
  const registrations = [
    {
      name: 'legacy-aura',
      scope: 'http://127.0.0.1:8000/static/',
      active: { scriptURL: 'http://127.0.0.1:8000/static/service-worker.js' },
      async unregister() { retired.push(this.name); return true; },
    },
    {
      name: 'unrelated',
      scope: 'http://127.0.0.1:8000/static/',
      active: { scriptURL: 'http://127.0.0.1:8000/static/other-worker.js' },
      async unregister() { retired.push(this.name); return true; },
    },
    {
      name: 'current-aura',
      scope: 'http://127.0.0.1:8000/',
      active: {
        scriptURL: `http://127.0.0.1:8000/static/service-worker.js?_aura_runtime=${revision}`,
      },
      async unregister() { retired.push(this.name); return true; },
    },
  ];
  const context = {
    URL,
    Promise,
    console: { warn() {}, error() {} },
    window: { location: { origin: 'http://127.0.0.1:8000' } },
    navigator: {
      serviceWorker: {
        async getRegistrations() { return registrations; },
      },
    },
    requestServiceWorkerActivation() { return true; },
    state: {},
  };
  auraServiceWorkerFunctions(context);

  const count = await context.retireLegacyStaticServiceWorkers();

  assert.equal(count, 1);
  assert.deepEqual(retired, ['legacy-aura']);
});
