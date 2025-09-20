// face_monitor.js
// Browser-side Step 1: Basic Face Presence Detection using BlazeFace
// Exports startFacePresenceMonitor(opts)
// opts: { videoEl, canvasEl, statusEl, backendEndpoint }

export async function startFacePresenceMonitor(opts = {}) {
  const { videoEl, canvasEl, statusEl, backendEndpoint } = opts;
  if (!videoEl || !canvasEl || !statusEl) throw new Error('videoEl, canvasEl, statusEl required');

  // params (tune for your environment)
  const CHECK_INTERVAL_MS = 250;     // how often we run detection
  const MISSING_THRESHOLD_MS = 2000; // how long of missing face -> raise event
  const SNAPSHOT_QUALITY = 0.6;      // 0..1 for JPEG quality if you include image

  // state
  let model = null;
  let lastFacePresent = true;
  let faceMissingSince = null;

  statusEl.innerText = 'Initializing camera…';
  await setupCamera(videoEl);
  statusEl.innerText = 'Loading model…';

  // load BlazeFace
  model = await blazeface.load();
  statusEl.innerText = 'Model ready — monitoring face presence.';

  // main loop
  setInterval(async () => {
    try {
      const predictions = await model.estimateFaces(videoEl, false);

      

      // basic presence check
      const facePresent = Array.isArray(predictions) && predictions.length > 0;

      // simple confidence: number of faces (1 => confident), also could use box size
      const confidence = Math.min(1, (predictions.length > 0 ? 0.95 : 0.1));

      // detection logic: only send events on transitions (to reduce noise)
      if (!facePresent) {
        if (faceMissingSince === null) faceMissingSince = Date.now();
        const msMissing = Date.now() - faceMissingSince;
        if (msMissing >= MISSING_THRESHOLD_MS && lastFacePresent) {
          // raise event: face missing
          const eventObj = buildEvent('face_missing', 'medium', confidence);
          // attach a small snapshot for evidence (optional)
          eventObj.snapshot = captureSnapshotBase64(videoEl, canvasEl, SNAPSHOT_QUALITY, 160); // 160px width thumbnail
          // send to backend (non-blocking)
          sendEvent(eventObj, backendEndpoint).catch(e => console.warn('sendEvent failed', e));
          // update UI
        //   statusEl.innerText = `Face MISSING (since ${new Date(eventObj.timestamp).toLocaleTimeString()})`;
          statusEl.innerText = "0 face detected";
        }
      } else {
        // face present
        faceMissingSince = null;
        if (!lastFacePresent) {
          const eventObj = buildEvent('face_present', 'low', confidence);
          sendEvent(eventObj, backendEndpoint).catch(e => console.warn('sendEvent failed', e));
        }
        statusEl.innerText = `Face detected — ${predictions.length} face(s).`;
      }

      lastFacePresent = facePresent;
    } catch (err) {
      console.error('Detection loop error', err);
      statusEl.innerText = 'Error in detection loop (check console)';
    }
  }, CHECK_INTERVAL_MS);

  // helpers
  function buildEvent(event, severity = 'low', confidence = 1.0) {
    return {
      user_id: getUserId(), // integrate with your auth
      event,
      severity,
      confidence,
      timestamp: new Date().toISOString(),
      meta: { step: 'face_presence_v1' }
    };
  }

  function captureSnapshotBase64(video, canvas, quality = 0.6, thumbWidth = 160) {
    // draw mirrored (so it matches user perspective)
    const ctx = canvas.getContext('2d');
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return null;
    // scale down thumbnail
    const scale = thumbWidth / w;
    const tw = Math.round(w * scale);
    const th = Math.round(h * scale);
    canvas.width = tw;
    canvas.height = th;
    // mirror horizontally so face is not reversed relative to user view
    ctx.save();
    ctx.translate(tw, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, tw, th);
    ctx.restore();
    try {
      return canvas.toDataURL('image/jpeg', quality);
    } catch (e) {
      console.warn('Snapshot capture failed', e);
      return null;
    }
  }

  async function sendEvent(eventObj, endpoint) {
    if (!endpoint) {
      console.debug('Event not sent (no endpoint configured)', eventObj);
      return;
    }
    // keep it lightweight: send metadata + optional small snapshot (base64)
    const body = JSON.stringify(eventObj);
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    });
    if (!res.ok) {
      console.warn('Backend responded with non-ok status', res.status);
    }
    return res;
  }

  function getUserId() {
    // TODO: integrate with your auth/session token
    // placeholder: return null or read from a cookie/localStorage
    return (window.currentUserId || 'unknown_user');
  }

  // set up camera stream
  async function setupCamera(video) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera API not supported in this browser');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.play();
        resolve();
      };
    });
  }
}
