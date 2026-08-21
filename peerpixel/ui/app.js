/* The window.

   Two loops. One asks the app what is true, a few times a second. The other
   draws, every frame, and it is the reason the bar looks like motion rather
   than a series of jumps: between two answers it carries on at the speed the
   last answer reported, and eases onto the next one instead of snapping.

   The rules the bar obeys live in peerpixel/progress.py, where they can be
   tested. Nothing here invents progress -- it only refuses to let a real
   number arrive as a jolt. */

const TOKEN = document.body.dataset.token;
const POLL_MS = 400;
const el = (id) => document.getElementById(id);

async function call(path, body) {
  const response = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers: { 'content-type': 'application/json', 'x-peerpixel-token': TOKEN },
    body: body && JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || 'That did not work');
  return payload;
}

/* -- the bar ---------------------------------------------------------- */

const bar = { key: '', shown: 0, target: 0, speed: 0, sampled: 0, done: false, still: false };
let etaBase = null;
let etaAt = 0;
let lastFrame = performance.now();

function feed(progress, key) {
  if (key !== bar.key) {           // a different task: start its bar at its own zero
    bar.key = key;
    bar.shown = 0;
    bar.done = false;
  }
  bar.target = progress.fraction || 0;
  bar.speed = progress.speed || 0;
  bar.sampled = performance.now();
  bar.done = !!progress.finished;
  bar.still = !!progress.failed;
  etaBase = progress.etaSeconds;
  etaAt = performance.now();
}

function draw(now) {
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;

  // Where the app would be by now if it kept the pace it last reported. The
  // ceiling is the same one the server keeps: only finishing shows 100%.
  const coasted = (now - bar.sampled) / 1000 * bar.speed;
  const ahead = bar.done ? 1 : Math.min(0.995, bar.target + coasted);
  if (ahead > bar.shown) bar.shown += (ahead - bar.shown) * (1 - Math.exp(-dt * 6));

  el('stageFill').style.width = (bar.shown * 100).toFixed(2) + '%';
  el('stagePercent').textContent = Math.floor(bar.shown * 100) + '%';
  el('stage').dataset.still = bar.still ? '1' : '0';

  if (etaBase === null || etaBase === undefined) {
    el('stageEta').textContent = bar.still ? 'stopped' : 'working out how long';
  } else {
    const left = Math.max(0, etaBase - (now - etaAt) / 1000);
    el('stageEta').textContent = bar.done ? 'done' : howLong(left);
  }
  requestAnimationFrame(draw);
}

function howLong(seconds) {
  if (seconds < 5) return 'a few seconds left';
  if (seconds < 90) return `${Math.round(seconds)} seconds left`;
  const minutes = seconds / 60;
  if (minutes < 60) return `about ${Math.round(minutes)} minutes left`;
  const hours = Math.floor(minutes / 60);
  const rest = Math.round(minutes % 60);
  return `about ${hours}h ${rest}m left`;
}

/* -- painting the rest ------------------------------------------------ */

let lastImage = 0;
let editingCode = false;

function paint(state) {
  const activity = state.activity || {};
  const worker = state.worker || {};
  const progress = activity.progress || null;
  const busy = !!activity.running && !!progress;

  el('stage').hidden = !busy;
  el('restCard').hidden = busy;

  if (busy) {
    feed(progress, `${activity.task}:${activity.kind}`);
    el('stageTitle').textContent = progress.label || activity.title || 'Working';
    el('stageDetail').textContent = progress.detail || activity.title || ' ';
    el('stageStep').textContent = activity.kind === 'job'
      ? (worker.prompt ? 'Rendering for someone' : 'Rendering')
      : (activity.steps > 1 ? `Step ${activity.step} of ${activity.steps}` : 'Setting up');
    const queued = activity.queued || [];
    const overall = activity.overallEtaSeconds;
    el('stageQueue').textContent = queued.length
      ? `Then: ${queued.join(', ')}${overall ? ` · about ${Math.round(overall / 60)} min in total` : ''}`
      : (activity.kind === 'job' && worker.prompt ? `“${worker.prompt}”` : '');
    // A render still has to be stoppable. It finishes the picture somebody is
    // waiting for and then stops, rather than abandoning work half done.
    el('cancel').hidden = activity.kind === 'job';
    el('stopWorker').hidden = activity.kind !== 'job';
    el('stopWorker').textContent = state.stopping ? 'Stopping after this' : 'Stop after this';
    el('stopWorker').disabled = !!state.stopping;
  } else {
    el('restTitle').textContent = restTitle(state);
    el('restDetail').textContent = restDetail(state);
    el('power').textContent = worker.running ? 'Stop' : 'Start rendering';
    el('power').disabled = !state.ready;
  }

  // status pill
  const [tone, text] = pill(state, activity, worker);
  el('pill').dataset.tone = tone;
  el('pillText').textContent = text;

  // setup checklist
  const steps = state.steps || {};
  const doing = activity.kind === 'setup' ? activity.task : null;
  mark('stepDependencies', steps.dependencies, doing === 'install');
  mark('stepPaired', steps.paired, false);
  mark('stepModel', steps.model, doing === 'model');
  mark('stepApproved', steps.approved, doing === 'bench');
  el('pairedNote').textContent = steps.paired
    ? `Paired as ${state.deviceId || 'this machine'}.`
    : 'Get a code from peerpixel.cc, then paste it here.';
  el('benchNote').textContent = state.benchMs
    ? `${(state.benchMs / 1000).toFixed(1)}s for four steps.`
    : 'One timed render, to be sure nobody is left waiting.';
  el('setupCard').hidden = false;
  el('resume').hidden = state.ready || busy || !steps.paired;

  // session
  el('images').textContent = worker.images || 0;
  el('earned').textContent = `${round(worker.earnedPixels || 0)} px`;
  el('rate').textContent = worker.pixelsPerHour == null
    ? '—' : `${Number(worker.pixelsPerHour).toFixed(1)}/h`;
  if (worker.lastImageAt && worker.lastImageAt !== lastImage) {
    lastImage = worker.lastImageAt;
    el('preview').src = `/api/preview?t=${lastImage}&token=${encodeURIComponent(TOKEN)}`;
    el('shotNote').textContent = worker.lastEarnedPixels > 0
      ? `Delivered · +${round(worker.lastEarnedPixels)} px`
      : 'Delivered · free or self-render, no payout';
  }

  // free work
  if (document.activeElement !== el('allowFree')) el('allowFree').checked = !!state.allowFree;
  el('freeNote').textContent = state.allowFree && !state.allowFreeConfirmed
    ? 'Saved here, but your account has not confirmed it. The switch belongs to your account: flip it on the Contribute page at peerpixel.cc.'
    : 'Free jobs pay nothing and always wait behind paid ones. Nobody’s card renders for strangers without this being on.';

  // update
  const update = state.update || {};
  el('updateBanner').hidden = !update.available;
  el('updateVersions').textContent = update.available
    ? `${update.latest} is out; this is ${update.current}.` : '';

  // details
  el('factMachine').textContent = state.machine || '—';
  el('factAccelerator').textContent = state.accelerator || 'not looked at yet';
  el('factDevice').textContent = state.deviceId || 'not paired';
  el('factVersion').textContent = update.current || '—';
  el('factApi').textContent = state.api || '—';
  el('log').textContent = state.log || 'Nothing yet.';

  const notice = state.message || (progress && progress.failed ? progress.error : '') || worker.error || '';
  el('notice').hidden = !notice;
  el('notice').textContent = notice;
}

function restTitle(state) {
  if (!state.steps.paired) return 'Pair this machine';
  if (!state.ready) return 'Setup is not finished';
  return state.worker.running
    ? (state.worker.connected ? 'Online, waiting for work' : 'Connecting…')
    : 'Ready';
}

function restDetail(state) {
  if (!state.steps.paired) return 'Get a pairing code from peerpixel.cc and paste it below.';
  if (!state.ready) return 'Carry on with setup below and this machine can start earning.';
  if (!state.worker.running) return 'Press start and this machine begins taking work. You can close this window afterwards; it keeps rendering.';
  return state.worker.connected
    ? 'Nothing to render this second. It will start the moment something arrives.'
    : 'Reconnecting to peerpixel.cc.';
}

function pill(state, activity, worker) {
  if (activity.kind === 'job') return ['busy', 'Rendering'];
  if (activity.kind === 'setup') return ['busy', 'Setting up'];
  if (worker.running && worker.connected) return ['live', 'Online'];
  if (worker.running) return ['busy', 'Connecting'];
  if (!state.ready) return ['', 'Not set up'];
  return ['', 'Stopped'];
}

function mark(id, done, doing) {
  el(id).dataset.state = done ? 'done' : (doing ? 'doing' : 'todo');
}

function round(n) {
  return Math.round(Number(n) * 10) / 10;
}

/* -- controls --------------------------------------------------------- */

function wire() {
  el('pair').onclick = async () => {
    const code = el('code').value.trim();
    if (!code) return;
    el('pair').disabled = true;
    try {
      paint(await call('/api/pair', { code }));
      el('code').value = '';
    } catch (error) {
      say(error.message);
    } finally {
      el('pair').disabled = false;
    }
  };
  el('code').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') el('pair').click();
  });
  el('code').addEventListener('focus', () => { editingCode = true; });
  el('code').addEventListener('blur', () => { editingCode = false; });

  el('power').onclick = async () => {
    const starting = el('power').textContent !== 'Stop';
    el('power').disabled = true;
    try { paint(await call('/api/worker', { running: starting })); }
    catch (error) { say(error.message); }
    finally { el('power').disabled = false; }
  };
  el('stopWorker').onclick = async () => {
    el('stopWorker').disabled = true;
    try { await call('/api/worker', { running: false, afterThis: true }); }
    catch (error) { say(error.message); }
    finally { el('stopWorker').disabled = false; }
  };
  el('resume').onclick = () => call('/api/setup', {}).then(paint).catch((e) => say(e.message));
  el('cancel').onclick = () => call('/api/cancel', {}).then(paint).catch((e) => say(e.message));
  el('updateNow').onclick = async () => {
    el('updateNow').disabled = true;
    try { paint(await call('/api/run', { task: 'update' })); }
    catch (error) { say(error.message); el('updateNow').disabled = false; }
  };
  el('allowFree').onchange = async () => {
    try { paint(await call('/api/free', { allowFree: el('allowFree').checked })); }
    catch (error) { el('allowFree').checked = !el('allowFree').checked; say(error.message); }
  };
  el('quit').onclick = async () => {
    await call('/api/quit', {}).catch(() => {});
    el('pillText').textContent = 'Closed';
    document.body.style.opacity = 0.4;
  };
}

function say(message) {
  el('notice').hidden = false;
  el('notice').textContent = message;
}

/* -- loops ------------------------------------------------------------ */

let missed = 0;

async function poll() {
  try {
    paint(await call('/api/state'));
    missed = 0;
  } catch (error) {
    missed += 1;
    // One missed poll is a restart landing; several is a window with nothing
    // behind it, and saying so beats a page that silently stops updating.
    if (missed > 6) {
      el('pill').dataset.tone = 'bad';
      el('pillText').textContent = 'Not running';
    }
  }
  setTimeout(poll, POLL_MS);
}

wire();
requestAnimationFrame(draw);
poll();
