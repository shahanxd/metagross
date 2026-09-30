// METAGROSS stereo renderer (Three.js r186, WebGL2).
//
// Public API on window (driven synchronously by metagross/sim/render/bridge.py
// through Playwright page.evaluate; no requestAnimationFrame anywhere):
//   configureCameras(cfg)        intrinsics + T_body_cam + vehicle spec (from Python defaults)
//   loadScenario(json)           build terrain/objects/sky/lighting from a metagross.scenario/1 dict
//   renderStereo(state)          -> {left: b64 RGB HxWx3, right: b64 GRAY HxW, ms, exposure}
//   renderGT(state)              -> {depth: b64 float32 HxW (m along optical axis, 0 = sky), semantic: b64 uint8 HxW}
//   renderChase(state, w, h)     -> {rgb: b64 RGB hxwx3}
//   rendererInfo()               -> {renderer, vendor, ...}
//
// Frames: the Three.js scene is built directly in the WORLD frame (x east, y north,
// z up). Cameras are placed with explicit matrices, so Three's y-up default is
// never used for geometry (Object3D.DEFAULT_UP is set to +z for lights/lookAt).
// Pose convention: R = Rz(yaw) Ry(pitch) Rx(roll) (see camera_model.py).

import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { Terrain } from './terrain.js';
import { buildStaticObjects, makeDynamic, makeVehicle, LAYER_CHASE_ONLY } from './objects.js';
import { POST_VERT, POST_FRAG, METER_FRAG, DEPTH_VERT, DEPTH_FRAG, SEM_VERT, SEM_OBJ_FRAG, semTerrainFrag } from './shaders.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// ------------------------------------------------------------------ constants
const AE_TAU_S = 1.0;          // auto-exposure first-order time constant (s)
const AE_MAX_GAIN = 8.0;       // AE range relative to scenario exposure: [1/8, 8] (x2 time, x4 analog gain)
const AE_KEY = 0.30;           // metered mean HDR luminance mapped to the scenario's nominal exposure
const AE_CLAMP_LUM = 3.0;      // per-sample luminance clamp in the meter (keeps the sun disc from dominating)
const AE_SKY_WEIGHT = 0.4;     // metering weight of the top 30 % of the image
const NOISE_SIGMA_DN = 2.0;    // sensor read noise at unity gain (8-bit DN)
const VIGNETTE = 0.22;         // relative fall-off at the corner
const LENS_BLUR = 1.0;         // ~0.5 px sigma optical blur (0 = pinhole-sharp)
const EVENT_RAMP_S = 0.5;      // lighting event fade in/out (s)
const SHADOW_HALF_M = 22.0;    // half-size of the shadow frustum around the vehicle (m)
const DUST_FOG_GAIN = 8.0;     // fog density multiplier per unit dust level
const CHASE_BACK_M = 6.0, CHASE_UP_M = 3.0; // chase camera offset (m)
const SKY_GAIN = 0.5;          // Preetham sky radiance scale (keeps a blue sky under ACES at unity exposure)
const FOG_LAYER_M = 60.0;      // height of the ground fog/haze layer seen against the sky (m)
const AE_TIME_HEADROOM = 2.0;  // exposure-time increase available before analog gain (and noise) rises
const FLARE_GAIN = 2.0;        // flare sprite intensity per unit glare level (post-exposure linear units)
const VEIL_GAIN = 0.05;        // uniform veiling glare per unit glare level (post-exposure linear units)
const AE_GLARE_LOAD = 0.6;     // extra metered light per unit glare level (flare/veil seen by the AE meter)
// Playwright's transfer cost grows super-linearly with message size; ~350 k-char
// base64 slices fetched by separate evaluate calls measured fastest on this laptop.
const CHUNK_CHARS = 349524;    // multiple of 4 so every slice decodes on its own

const OPTS = Object.assign({ shadows: true, msaa: 0 }, window.MG_OPTIONS || {});

// ------------------------------------------------------------------ renderer
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: false });
renderer.setPixelRatio(1); renderer.setSize(16, 16, false);
renderer.toneMapping = THREE.NoToneMapping; renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.shadowMap.enabled = !!OPTS.shadows; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.shadowMap.autoUpdate = false;
const gl = renderer.getContext();

function rendererInfo() {
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return {
    renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    vendor: ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
    webgl2: renderer.capabilities.isWebGL2 !== false, three: THREE.REVISION,
    colorBufferFloat: !!gl.getExtension('EXT_color_buffer_float'),
    shadows: renderer.shadowMap.enabled, toBase64: !!Uint8Array.prototype.toBase64,
  };
}

function bytesToB64(u8) {
  if (u8.toBase64) return u8.toBase64();
  let s = ''; const CH = 0x8000;
  for (let i = 0; i < u8.length; i += CH) s += String.fromCharCode.apply(null, u8.subarray(i, i + CH));
  return btoa(s);
}

// ------------------------------------------------------------------ cameras
const cam = { W: 640, H: 400, fx: 440, fy: 440, cx: 319.5, cy: 199.5, B: 0.12, near: 0.05, far: 200, Tbc: new THREE.Matrix4(), veh: null, TbcArr: null };
const CV_TO_GL = new THREE.Matrix4().makeScale(1, -1, -1);
const camL = new THREE.PerspectiveCamera(); const camR = new THREE.PerspectiveCamera();
const chaseCam = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 400); chaseCam.layers.enable(LAYER_CHASE_ONLY);
for (const c of [camL, camR]) { c.matrixAutoUpdate = false; c.layers.set(0); }

function projectionFromIntrinsics(fx, fy, cx, cy, W, H, n, f) {
  // Exact OpenCV pinhole (pixel centres at integers, v down) -> GL clip space; see camera_model.projection_matrix_gl.
  const P = new THREE.Matrix4();
  P.set(2 * fx / W, 0, 1 - 2 * (cx + 0.5) / W, 0,
    0, 2 * fy / H, 2 * (cy + 0.5) / H - 1, 0,
    0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
    0, 0, -1, 0);
  return P;
}

function configureCameras(cfg) {
  Object.assign(cam, { W: cfg.width, H: cfg.height, fx: cfg.fx, fy: cfg.fy, cx: cfg.cx, cy: cfg.cy, B: cfg.baseline, near: cfg.near, far: cfg.far });
  cam.Tbc = new THREE.Matrix4().fromArray(cfg.T_body_cam).transpose(); // row-major in, column-major store
  cam.TbcArr = cfg.T_body_cam; cam.veh = cfg.vehicle || null;
  const P = projectionFromIntrinsics(cam.fx, cam.fy, cam.cx, cam.cy, cam.W, cam.H, cam.near, cam.far);
  for (const c of [camL, camR]) { c.projectionMatrix.copy(P); c.projectionMatrixInverse.copy(P).invert(); c.near = cam.near; c.far = cam.far; }
  allocTargets();
  if (world.vehicle) { world.scene.remove(world.vehicle); world.vehicle = null; }
  if (world.scene && cam.veh) { world.vehicle = makeVehicle(cam.veh, cam.TbcArr); world.scene.add(world.vehicle); }
  return { ok: true };
}

/** Body -> world matrix for pose [x,y,z,roll,pitch,yaw] with R = Rz(yaw) Ry(pitch) Rx(roll). */
function poseMatrix(pose) {
  const [x, y, z, r, p, yw] = pose;
  const cr = Math.cos(r), sr = Math.sin(r), cp = Math.cos(p), sp = Math.sin(p), cy = Math.cos(yw), sy = Math.sin(yw);
  const M = new THREE.Matrix4();
  M.set(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y,
    -sp, cp * sr, cp * cr, z,
    0, 0, 0, 1);
  return M;
}

/** OpenCV camera -> world for the left (eye=0) or right (eye=1) camera. */
function cvCamToWorld(pose, eye) {
  const M = poseMatrix(pose).multiply(cam.Tbc);
  if (eye === 1) M.multiply(new THREE.Matrix4().makeTranslation(cam.B, 0, 0));
  return M;
}
function placeStereo(pose) {
  for (const [c, eye] of [[camL, 0], [camR, 1]]) {
    c.matrix.copy(cvCamToWorld(pose, eye)).multiply(CV_TO_GL);
    c.matrixWorldNeedsUpdate = true; c.updateMatrixWorld(true);
  }
}

// ------------------------------------------------------------------ render targets
const T = {};
function mkRT(w, h, type, samples = 0) {
  const rt = new THREE.WebGLRenderTarget(w, h, { type, format: THREE.RGBAFormat, depthBuffer: true, stencilBuffer: false, samples, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false });
  rt.texture.colorSpace = THREE.NoColorSpace; return rt;
}
function allocTargets() {
  for (const k of Object.keys(T)) if (T[k] && T[k].dispose) T[k].dispose();
  T.hdrL = mkRT(cam.W, cam.H, THREE.HalfFloatType, OPTS.msaa); T.hdrR = mkRT(cam.W, cam.H, THREE.HalfFloatType, OPTS.msaa);
  T.out = mkRT(cam.W, cam.H, THREE.UnsignedByteType); T.gt = mkRT(cam.W, cam.H, THREE.UnsignedByteType);
  T.buf = new Uint8Array(cam.W * cam.H * 4); T.left = new Uint8Array(cam.W * cam.H * 3); T.right = new Uint8Array(cam.W * cam.H);
  T.packed = new Uint8Array(cam.W * cam.H * 4);
  T.packedYuv = new Uint8Array(cam.W * cam.H * 2 + 2 * (cam.W >> 1) * (cam.H >> 1));
  T.meter = new THREE.WebGLRenderTarget(16, 10, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  T.meterBuf = new Float32Array(16 * 10 * 4);
  T.chase = {};
}

// ------------------------------------------------------------------ post pass
const postMat = new THREE.ShaderMaterial({
  vertexShader: POST_VERT, fragmentShader: POST_FRAG, depthTest: false, depthWrite: false, blending: THREE.NoBlending, toneMapped: false,
  uniforms: {
    tL: { value: null }, tR: { value: null }, uMode: { value: 0 }, uSize: { value: new THREE.Vector2() }, uPP: { value: new THREE.Vector2() },
    uGainL: { value: 1 }, uGainR: { value: 1 }, uNoiseDN: { value: NOISE_SIGMA_DN }, uVignette: { value: VIGNETTE }, uBlur: { value: LENS_BLUR },
    uSun: { value: new THREE.Vector3() }, uVeil: { value: 0 }, uHaze: { value: new THREE.Vector3() }, uHazeAmt: { value: 0 }, uSeed: { value: 0 },
  },
});
const postScene = new THREE.Scene(); const postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const postQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), postMat); postQuad.frustumCulled = false; postScene.add(postQuad);
const meterMat = new THREE.ShaderMaterial({
  vertexShader: POST_VERT, fragmentShader: METER_FRAG, depthTest: false, depthWrite: false, blending: THREE.NoBlending,
  uniforms: { tSrc: { value: null }, uSrcSize: { value: new THREE.Vector2() }, uClampLum: { value: AE_CLAMP_LUM } },
});
const meterScene = new THREE.Scene(); const meterQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), meterMat); meterQuad.frustumCulled = false; meterScene.add(meterQuad);

/** Meter the left HDR frame (pre-exposure, pre-dim); result drives the NEXT frame's AE target. */
function meterExposure(hdr) {
  meterMat.uniforms.tSrc.value = hdr.texture; meterMat.uniforms.uSrcSize.value.set(cam.W, cam.H);
  renderer.setRenderTarget(T.meter); renderer.render(meterScene, postCam);
  renderer.readRenderTargetPixels(T.meter, 0, 0, 16, 10, T.meterBuf);
  let s = 0, ws = 0;
  for (let r = 0; r < 10; r++) for (let c = 0; c < 16; c++) {
    const w = r >= 7 ? AE_SKY_WEIGHT : 1.0; // GL rows: r = 7..9 are the top of the image (mostly sky)
    s += w * T.meterBuf[(r * 16 + c) * 4]; ws += w;
  }
  world.exposure.meter = s / ws;
}

// ------------------------------------------------------------------ world state
const world = { scene: null, terrain: null, sky: null, sun: null, hemi: null, env: null, dyn: [], vehicle: null, scenario: null, seed: 0, sunDir: new THREE.Vector3(0, 0, 1), fogBase: 0, fogColor: new THREE.Color(), dustColor: new THREE.Color(), events: [], exposure: { E: 1, t: null }, time: { value: 0 }, semMats: null, depthMat: null, baseExposure: 1 };

function patchSkyZUp(sky) {
  // three's Preetham Sky assumes y-up; this scene is z-up. Swap the axes in the shader text
  // and fail loudly if the upstream code changes so the patch no longer applies.
  const m = sky.material; let vs = m.vertexShader, fs = m.fragmentShader;
  const rep = (s, a, b) => { if (!s.includes(a)) throw new Error('Sky patch failed: ' + a); return s.split(a).join(b); };
  vs = rep(vs, 'sunIntensity( vSunDirection.y )', 'sunIntensity( vSunDirection.z )');
  vs = rep(vs, 'sunPosition.y / 450000.0', 'sunPosition.z / 450000.0');
  fs = rep(fs, 'max( 0.0, direction.y )', 'max( 0.0, direction.z )');
  fs = rep(fs, 'pow( 1.0 - vSunDirection.y, 5.0 )', 'pow( 1.0 - vSunDirection.z, 5.0 )');
  fs = rep(fs, 'acos( direction.y )', 'acos( direction.z )');
  fs = rep(fs, 'atan( direction.z, direction.x )', 'atan( direction.y, direction.x )');
  fs = rep(fs, 'direction.y > 0.0 && cloudCoverage', 'direction.z > 0.0 && cloudCoverage');
  fs = rep(fs, 'direction.xz / ( direction.y * elevation )', 'direction.xy / ( direction.z * elevation )');
  fs = rep(fs, 'smoothstep( 0.0, 0.03 + 0.06 * cloudElevation, direction.y )', 'smoothstep( 0.0, 0.03 + 0.06 * cloudElevation, direction.z )');
  fs = rep(fs, 'smoothstep( -0.08, 0.3, vSunDirection.y )', 'smoothstep( -0.08, 0.3, vSunDirection.z )');
  fs = rep(fs, 'uniform float time;', 'uniform float time;\nuniform float uSkyGain;\nuniform vec3 uDust;\nuniform float uDustAmt;');
  // Sky seen through a fog layer of height FOG_LAYER_M: path length H / sin(elevation), same
  // FogExp2 law as the terrain, so the horizon has no hard fog/sky edge (and below-horizon = fog).
  fs = rep(fs, 'gl_FragColor = vec4( texColor, 1.0 );', `float mgL = uFogH / max(direction.z, 1e-3);
			float mgF = 1.0 - exp(-(uFogDensity * mgL) * (uFogDensity * mgL));
			texColor = mix(texColor * uSkyGain, uFogCol, mgF);
			texColor = mix(texColor, uDust, uDustAmt);
			gl_FragColor = vec4( texColor, 1.0 );`);
  fs = rep(fs, 'uniform float uSkyGain;', 'uniform float uSkyGain;\nuniform vec3 uFogCol;\nuniform float uFogDensity;\nuniform float uFogH;');
  m.vertexShader = vs; m.fragmentShader = fs;
  m.uniforms.uSkyGain = { value: 1.0 }; m.uniforms.uDust = { value: new THREE.Vector3() }; m.uniforms.uDustAmt = { value: 0 };
  m.uniforms.uFogCol = { value: new THREE.Vector3(0.6, 0.65, 0.7) }; m.uniforms.uFogDensity = { value: 0 }; m.uniforms.uFogH = { value: FOG_LAYER_M };
  m.needsUpdate = true;
}

function srgbColor(r, g, b) { return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace); }

/** Mean linear sky radiance 2-10 deg above the horizon over 8 azimuths (sun disc hidden). */
function skyHorizonRadiance(sky) {
  const tmp = new THREE.Scene(); tmp.add(sky);
  const rt = new THREE.WebGLRenderTarget(8, 8, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: true });
  const c = new THREE.PerspectiveCamera(8, 1, 0.1, 1000); const buf = new Float32Array(8 * 8 * 4);
  const su = sky.material.uniforms; const disc = su.showSunDisc.value; su.showSunDisc.value = 0;
  const acc = new THREE.Color(0, 0, 0); let n = 0; const el = THREE.MathUtils.degToRad(6);
  sky.position.set(0, 0, 0); sky.updateMatrixWorld();
  for (let k = 0; k < 8; k++) {
    const a = k * Math.PI / 4;
    c.position.set(0, 0, 0); c.up.set(0, 0, 1); c.lookAt(Math.cos(a) * Math.cos(el), Math.sin(a) * Math.cos(el), Math.sin(el)); c.updateMatrixWorld();
    renderer.setRenderTarget(rt); renderer.render(tmp, c);
    renderer.readRenderTargetPixels(rt, 0, 0, 8, 8, buf);
    for (let i = 0; i < 64; i++) { acc.r += buf[i * 4]; acc.g += buf[i * 4 + 1]; acc.b += buf[i * 4 + 2]; n++; }
  }
  su.showSunDisc.value = disc; tmp.remove(sky); rt.dispose(); renderer.setRenderTarget(null);
  return acc.multiplyScalar(1 / n);
}

function loadScenario(sc) {
  const t0 = performance.now();
  disposeWorld();
  const scene = new THREE.Scene(); world.scene = scene; world.scenario = sc; world.seed = (sc.seed | 0) >>> 0;
  const L = sc.lighting || {};
  const elev = THREE.MathUtils.degToRad(L.sun_elev_deg ?? 35), az = THREE.MathUtils.degToRad(L.sun_azim_deg ?? 135);
  // sun_azim_deg is measured counter-clockwise from world +x (east), as documented in metagross/sim/scenario.py.
  world.sunDir.set(Math.cos(az) * Math.cos(elev), Math.sin(az) * Math.cos(elev), Math.sin(elev)).normalize();
  world.baseExposure = L.exposure ?? 1.0;
  world.events = (L.events || []).map((e) => ({ ...e }));

  // --- sky (z-up Preetham) + image-based environment
  const sky = new Sky(); patchSkyZUp(sky); sky.scale.setScalar(150); sky.frustumCulled = false;
  const su = sky.material.uniforms;
  su.turbidity.value = 2.2; su.rayleigh.value = 1.6; su.mieCoefficient.value = 0.003; su.mieDirectionalG.value = 0.85;
  su.sunPosition.value.copy(world.sunDir).multiplyScalar(1000); su.cloudCoverage.value = 0.35; su.cloudScale.value = 0.00025;
  su.uSkyGain.value = SKY_GAIN; su.time.value = (world.seed % 1000) * 10;
  world.sky = sky;
  // Fog colour = rendered sky radiance just above the horizon (mean over 8 azimuths), so
  // distant terrain fades into the actual sky at any sun elevation.
  world.fogColor.copy(skyHorizonRadiance(sky));
  su.uFogCol.value.set(world.fogColor.r, world.fogColor.g, world.fogColor.b);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene(); const envSky = new Sky(); patchSkyZUp(envSky); envSky.scale.setScalar(150);
  for (const k of Object.keys(su)) if (envSky.material.uniforms[k]) envSky.material.uniforms[k].value = su[k].value?.clone ? su[k].value.clone() : su[k].value;
  envSky.material.uniforms.showSunDisc.value = 0; envScene.add(envSky);
  world.env = pmrem.fromScene(envScene, 0.02, 0.1, 500); pmrem.dispose(); envSky.geometry.dispose(); envSky.material.dispose();
  scene.environment = world.env.texture; scene.environmentIntensity = 0.55;
  scene.add(sky);

  // --- lights: sun (warmer + weaker near the horizon) and hemisphere fill
  const lowSun = 1 - THREE.MathUtils.smoothstep(Math.sin(elev), 0.05, 0.5);
  const sunCol = new THREE.Color(1.0, 0.96, 0.88).lerp(new THREE.Color(1.0, 0.72, 0.45), lowSun);
  const sun = new THREE.DirectionalLight(sunCol, 3.2 * (0.35 + 0.65 * THREE.MathUtils.smoothstep(Math.sin(elev), 0.0, 0.35)));
  sun.castShadow = renderer.shadowMap.enabled;
  if (sun.castShadow) {
    sun.shadow.mapSize.set(2048, 2048); const sc2 = sun.shadow.camera;
    sc2.left = -SHADOW_HALF_M; sc2.right = SHADOW_HALF_M; sc2.top = SHADOW_HALF_M; sc2.bottom = -SHADOW_HALF_M; sc2.near = 1; sc2.far = 200;
    sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03; sun.shadow.radius = 3;
  }
  scene.add(sun); scene.add(sun.target); world.sun = sun;
  const hemi = new THREE.HemisphereLight(srgbColor(0.62, 0.72, 0.9), srgbColor(0.36, 0.31, 0.22), 0.45); hemi.position.set(0, 0, 1); scene.add(hemi); world.hemi = hemi;

  // --- fog (FogExp2 density in 1/m) and dust tint (dust scaled to the horizon luminance)
  const lum = (c) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  const tint = srgbColor(0.66, 0.56, 0.42);
  world.dustColor.copy(tint).multiplyScalar(lum(world.fogColor) / lum(tint));
  world.fogBase = L.fog_density ?? 0.004;
  scene.fog = new THREE.FogExp2(world.fogColor.clone(), world.fogBase);

  // --- terrain + objects
  const terr = new Terrain(sc.terrain, sc.hazards || []); world.terrain = terr;
  const terrMat = terr.makeMaterial(world.time);
  const tgroup = terr.buildMeshes(terrMat); scene.add(tgroup);
  scene.add(terr.buildApron(terrMat, world.seed));
  const statics = buildStaticObjects(sc.objects || [], (x, y) => terr.heightAt(x, y)); scene.add(statics);
  world.dyn = (sc.dynamic || []).map((d, i) => { const o = makeDynamic(d, i); scene.add(o); placeDynamic(o, d, i, null, 0); return o; });
  if (cam.veh) { world.vehicle = makeVehicle(cam.veh, cam.TbcArr); scene.add(world.vehicle); }

  // --- ground-truth materials
  world.depthMat = new THREE.ShaderMaterial({ vertexShader: DEPTH_VERT, fragmentShader: DEPTH_FRAG, side: THREE.DoubleSide });
  const semObj = new THREE.ShaderMaterial({ vertexShader: SEM_VERT, fragmentShader: SEM_OBJ_FRAG, uniforms: { uId: { value: sc.__sem?.object ?? 1 } }, side: THREE.DoubleSide });
  const matToSem = [0, 1, 2, 3, 4, 5].map((k) => (sc.__sem?.material_to_sem?.[k] ?? [3, 3, 4, 3, 2, 2][k]));
  const semTer = new THREE.ShaderMaterial({ vertexShader: SEM_VERT, fragmentShader: semTerrainFrag(matToSem), uniforms: { ...terr.gridUniforms } });
  world.semMats = { object: semObj, terrain: semTer, sky: sc.__sem?.sky ?? 0 };

  // Warm-up: run every pass once so shader compilation happens here, not in the first timed frame.
  const p0 = sc.start ? [sc.start.xy[0], sc.start.xy[1], terr.heightAt(sc.start.xy[0], sc.start.xy[1]), 0, 0, sc.start.yaw || 0] : [0, 0, 0, 0, 0, 0];
  world.exposure = { E: world.baseExposure, t: null };
  renderStereo({ pose: p0, t: 0 }); renderStereo({ pose: p0, t: 0 }, { chunked: true, yuv420: true });
  renderGT({ pose: p0, t: 0 }); if (cam.veh) renderChase({ pose: p0, t: 0 }, 64, 36);
  world.exposure = { E: world.baseExposure, t: null, meter: world.exposure.meter }; // keep the start-pose meter reading
  return { ok: true, ms: performance.now() - t0, terrain: terr.stats, objects: (sc.objects || []).length, dynamic: world.dyn.length };
}

function disposeWorld() {
  if (!world.scene) return;
  world.scene.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose()); });
  if (world.env) world.env.dispose();
  if (world.terrain) { world.terrain.texA.dispose(); world.terrain.texB.dispose(); world.terrain.texNrm.dispose(); }
  if (world.depthMat) world.depthMat.dispose();
  if (world.semMats) { world.semMats.object.dispose(); world.semMats.terrain.dispose(); }
  world.scene = null; world.vehicle = null; world.dyn = [];
}

// ------------------------------------------------------------------ dynamic actors
// state.dynamic: list aligned with scenario.dynamic (or entries carrying "id"), each
// {xy:[x,y] | xyz:[x,y,z], yaw?: rad, visible?: bool, phase?: rad}. When absent,
// actors stand at path[0]. z defaults to the terrain height.
function placeDynamic(obj, spec, i, st, t) {
  let x, y, z, yaw = 0, vis = true, phase = null;
  const path = spec.path || [[0, 0], [1, 0]];
  if (st) {
    const p = st.xyz || st.xy || st.pos || path[0]; x = p[0]; y = p[1]; z = p.length > 2 && st.xyz ? p[2] : null;
    yaw = st.yaw ?? Math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0]); vis = st.visible ?? true; phase = st.phase ?? null;
  } else { x = path[0][0]; y = path[0][1]; z = null; yaw = Math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0]); }
  if (z === null) z = world.terrain ? world.terrain.heightAt(x, y) : 0;
  obj.position.set(x, y, z); obj.rotation.set(0, 0, yaw); obj.visible = vis;
  if (obj.userData.animate) {
    if (phase === null) { const d = Math.hypot(x - path[0][0], y - path[0][1]); phase = (d / obj.userData.stride) * Math.PI * 2; }
    obj.userData.animate(phase);
  }
}
function applyDynamic(state) {
  const list = state.dynamic;
  world.dyn.forEach((o, i) => {
    let st = null;
    if (Array.isArray(list)) st = list.find((e) => e && e.id === i) || (list[i] && list[i].id === undefined ? list[i] : null);
    placeDynamic(o, world.scenario.dynamic[i], i, st, state.t || 0);
  });
}

// ------------------------------------------------------------------ lighting model
function envelope(t, t0, t1) {
  const r = Math.min(EVENT_RAMP_S, Math.max((t1 - t0) / 4, 1e-3));
  return THREE.MathUtils.smoothstep(t, t0, t0 + r) * (1 - THREE.MathUtils.smoothstep(t, t1 - r, t1));
}
/** Active lighting levels at time t. state.lighting may override {dim, glare, dust}. */
function lightingLevels(state) {
  const t = state.t || 0; let irr = 1, glare = 0, dust = 0;
  for (const e of world.events) {
    const k = envelope(t, e.t0, e.t1); if (k <= 0) continue;
    const g = e.gain ?? 1;
    if (e.type === 'dim') irr *= 1 + (g - 1) * k; else if (e.type === 'glare') glare = Math.max(glare, g * k); else if (e.type === 'dust') dust = Math.max(dust, g * k);
  }
  const o = state.lighting || {};
  if (typeof o.dim === 'number') irr = o.dim; if (typeof o.glare === 'number') glare = o.glare; if (typeof o.dust === 'number') dust = o.dust;
  return { irr: Math.max(irr, 1e-3), glare, dust };
}

/** Sun visibility from camera position: heightmap ray march (trees ignored). */
function sunVisible(p) {
  if (world.sunDir.z <= 0) return 0; const terr = world.terrain; if (!terr) return 1;
  for (let s = 0.5; s < 120; s *= 1.08) {
    const x = p.x + world.sunDir.x * s, y = p.y + world.sunDir.y * s, z = p.z + world.sunDir.z * s;
    if (terr.heightAt(x, y) > z) return 0;
  }
  return 1;
}

function updateFrameUniforms(state, camera, advanceAE) {
  const lv = lightingLevels(state);
  const camPos = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
  const fwd = new THREE.Vector3(0, 0, -1).transformDirection(camera.matrixWorld);
  const cosSun = fwd.dot(world.sunDir);
  const nearView = THREE.MathUtils.smoothstep(cosSun, Math.cos(THREE.MathUtils.degToRad(70)), Math.cos(THREE.MathUtils.degToRad(25)));
  const vis = sunVisible(camPos);
  const glareW = lv.glare * nearView * vis;
  // Auto-exposure: previous frame's metered HDR luminance x current dim factor, plus the
  // flare light a real meter would see when the sun is near the view (flare is added in post).
  const ex = world.exposure; const t = state.t || 0;
  const metered = (ex.meter ?? AE_KEY) * lv.irr * (1 + AE_GLARE_LOAD * glareW);
  const target = THREE.MathUtils.clamp(world.baseExposure * AE_KEY / Math.max(metered, 1e-5), world.baseExposure / AE_MAX_GAIN, world.baseExposure * AE_MAX_GAIN);
  if (advanceAE) {
    if (ex.t === null || t < ex.t || t - ex.t > 5) ex.E = target;
    else ex.E += (target - ex.E) * (1 - Math.exp(-(t - ex.t) / AE_TAU_S));
    ex.t = t;
  }
  const gain = ex.E * lv.irr;
  world.fogCurrent = world.fogBase * (1 + DUST_FOG_GAIN * lv.dust);
  world.scene.fog.density = world.fogCurrent;
  world.scene.fog.color.copy(world.fogColor).lerp(world.dustColor, Math.min(1, lv.dust));
  world.sky.material.uniforms.uDust.value.set(world.scene.fog.color.r, world.scene.fog.color.g, world.scene.fog.color.b);
  world.sky.material.uniforms.uDustAmt.value = Math.min(0.9, lv.dust * 0.8);
  world.sky.material.uniforms.uFogDensity.value = world.fogCurrent;
  world.time.value = t; world.sky.material.uniforms.time.value = (world.seed % 1000) * 10 + t;
  return { lv, gain, glareW, vis, cosSun, analogGain: Math.max(1, ex.E / (world.baseExposure * AE_TIME_HEADROOM)) };
}

function sunPixel(camera) {
  // Project the (infinitely distant) sun with the camera's view rotation + OpenCV intrinsics.
  const dv = world.sunDir.clone().transformDirection(camera.matrixWorldInverse); // GL cam: -z forward
  if (dv.z >= -1e-3) return null;
  const u = cam.fx * (dv.x / -dv.z) + cam.cx, v = cam.fy * (-dv.y / -dv.z) + cam.cy;
  return { u, v };
}

function positionShadow(pose, extraLayers) {
  if (!renderer.shadowMap.enabled) return;
  const yaw = pose[5]; const cx = pose[0] + Math.cos(yaw) * 10, cy = pose[1] + Math.sin(yaw) * 10;
  const cz = world.terrain ? world.terrain.heightAt(cx, cy) : 0;
  world.sun.target.position.set(cx, cy, cz); world.sun.position.set(cx, cy, cz).addScaledVector(world.sunDir, 80);
  world.sun.target.updateMatrixWorld(); world.sun.updateMatrixWorld();
  world.sun.shadow.camera.layers.set(0); if (extraLayers) world.sun.shadow.camera.layers.enable(LAYER_CHASE_ONLY);
  renderer.shadowMap.needsUpdate = true;
}

function renderHDR(camera, target) {
  world.sky.position.setFromMatrixPosition(camera.matrixWorld); world.sky.updateMatrixWorld();
  renderer.setRenderTarget(target); renderer.render(world.scene, camera);
}

function frameSeed(state, salt) { return ((world.seed * 2654435761) ^ Math.round((state.t || 0) * 1000) ^ ((state.seq | 0) * 40503) ^ salt) >>> 0; }

// ------------------------------------------------------------------ chunked transfer
let chunks = [];
/** Base64-encode `u8`, keep the slices, return {n, first}; fetch the rest with mgChunk(i). */
function stash(u8) {
  const s = bytesToB64(u8); chunks = [];
  for (let i = 0; i < s.length; i += CHUNK_CHARS) chunks.push(s.slice(i, i + CHUNK_CHARS));
  return { n: chunks.length, first: chunks[0] };
}
function mgChunk(i) { return chunks[i]; }

// ------------------------------------------------------------------ public API
/**
 * Render the stereo pair. Default return: {left: b64 RGB, right: b64 GRAY, ms, exposure}.
 * With opts.chunked: {n, first, ms, exposure}; payload = left RGB then right GRAY.
 */
function renderStereo(state, opts) {
  const t0 = performance.now();
  if (!world.scene) throw new Error('loadScenario() first');
  placeStereo(state.pose); applyDynamic(state);
  if (world.vehicle) world.vehicle.visible = false;
  const fu = updateFrameUniforms(state, camL, true);
  positionShadow(state.pose, false);
  renderHDR(camL, T.hdrL); renderer.shadowMap.needsUpdate = false; renderHDR(camR, T.hdrR);
  const sp = sunPixel(camL);
  const u = postMat.uniforms;
  u.tL.value = T.hdrL.texture; u.tR.value = T.hdrR.texture; u.uMode.value = 0; u.uSize.value.set(cam.W, cam.H); u.uPP.value.set(cam.cx, cam.cy);
  u.uGainL.value = fu.gain; u.uGainR.value = fu.gain * 0.985; // slight inter-camera gain mismatch
  u.uNoiseDN.value = NOISE_SIGMA_DN * Math.max(1, fu.analogGain); u.uVignette.value = VIGNETTE; u.uBlur.value = LENS_BLUR;
  u.uSun.value.set(sp ? sp.u : -1e4, sp ? sp.v : -1e4, sp ? FLARE_GAIN * fu.glareW : 0); u.uVeil.value = VEIL_GAIN * fu.glareW;
  u.uHaze.value.set(world.scene.fog.color.r, world.scene.fog.color.g, world.scene.fog.color.b); u.uHazeAmt.value = 0.25 * Math.min(1, fu.lv.dust) * fu.gain;
  u.uSeed.value = frameSeed(state, 0x51ED);
  u.uMode.value = (opts && opts.chunked && opts.yuv420) ? 2 : 0;
  renderer.setRenderTarget(T.out); renderer.render(postScene, postCam);
  const t1 = performance.now();
  renderer.readRenderTargetPixels(T.out, 0, 0, cam.W, cam.H, T.buf);
  const t2 = performance.now();
  meterExposure(T.hdrL);
  const n = cam.W * cam.H, buf = T.buf; let out;
  if (opts && opts.chunked && opts.yuv420) {
    // Y_left (n) | U (n/4) | V (n/4) | Y_right (n); chroma = 2x2 block mean.
    const P = T.packedYuv, W = cam.W, H = cam.H, W2 = W >> 1, n4 = (W2) * (H >> 1);
    for (let k = 0, q = 0; k < n; k++, q += 4) { P[k] = buf[q]; P[n + 2 * n4 + k] = buf[q + 3]; }
    for (let by = 0; by < (H >> 1); by++) {
      // readback row r holds image row r (flipped in the post pass)
      const r0 = (2 * by) * W * 4, r1 = r0 + W * 4;
      for (let bx = 0; bx < W2; bx++) {
        const a = r0 + bx * 8, b = r1 + bx * 8, o = by * W2 + bx;
        P[n + o] = (buf[a + 1] + buf[a + 5] + buf[b + 1] + buf[b + 5] + 2) >> 2;
        P[n + n4 + o] = (buf[a + 2] + buf[a + 6] + buf[b + 2] + buf[b + 6] + 2) >> 2;
      }
    }
    out = stash(P);
  } else if (opts && opts.chunked) {
    // one buffer: left RGB (H*W*3 bytes) followed by right GRAY (H*W bytes)
    const P = T.packed;
    for (let k = 0, q = 0, m = 0; k < n; k++, q += 4, m += 3) { P[m] = buf[q]; P[m + 1] = buf[q + 1]; P[m + 2] = buf[q + 2]; P[3 * n + k] = buf[q + 3]; }
    out = stash(P);
  } else {
    const Lb = T.left, Rb = T.right;
    for (let k = 0, q = 0, m = 0; k < n; k++, q += 4, m += 3) { Lb[m] = buf[q]; Lb[m + 1] = buf[q + 1]; Lb[m + 2] = buf[q + 2]; Rb[k] = buf[q + 3]; }
    out = { left: bytesToB64(Lb), right: bytesToB64(Rb) };
  }
  const t3 = performance.now();
  out.ms = { draw: t1 - t0, readback: t2 - t1, encode: t3 - t2, total: t3 - t0 };
  out.exposure = { E: world.exposure.E, meter: world.exposure.meter, irr: fu.lv.irr, glare: fu.glareW, dust: fu.lv.dust, noise_dn: u.uNoiseDN.value };
  return out;
}

/** Left-camera GT. Default: {depth: b64 float32, semantic: b64 uint8}; opts.chunked -> {n, first}. */
function renderGT(state, opts) {
  const t0 = performance.now();
  placeStereo(state.pose); applyDynamic(state);
  if (world.vehicle) world.vehicle.visible = false;
  const sky = world.sky; sky.visible = false; const fog = world.scene.fog; world.scene.fog = null; const bg = world.scene.background; world.scene.background = null;
  const oldClear = renderer.getClearColor(new THREE.Color()); const oldAlpha = renderer.getClearAlpha();
  renderer.setClearColor(0x000000, 0);
  // depth
  world.scene.overrideMaterial = world.depthMat;
  renderer.setRenderTarget(T.gt); renderer.clear(); renderer.render(world.scene, camL);
  world.scene.overrideMaterial = null;
  const depthBytes = new Uint8Array(cam.W * cam.H * 4);
  renderer.readRenderTargetPixels(T.gt, 0, 0, cam.W, cam.H, depthBytes);
  // semantic: swap materials by role
  const swapped = [];
  world.scene.traverse((o) => {
    if (!o.isMesh || !o.userData.sem) return;
    swapped.push([o, o.material]); o.material = o.userData.sem === 'terrain' ? world.semMats.terrain : world.semMats.object;
  });
  renderer.setClearColor(new THREE.Color(world.semMats.sky / 255, 0, 0), 1);
  renderer.setRenderTarget(T.gt); renderer.clear(); renderer.render(world.scene, camL);
  for (const [o, m] of swapped) o.material = m;
  const semRGBA = new Uint8Array(cam.W * cam.H * 4);
  renderer.readRenderTargetPixels(T.gt, 0, 0, cam.W, cam.H, semRGBA);
  sky.visible = true; world.scene.fog = fog; world.scene.background = bg; renderer.setClearColor(oldClear, oldAlpha);
  // GL rows are bottom-up: flip to OpenCV row order while repacking.
  // Packed layout: depth float32 LE (H*W*4 bytes) followed by semantic uint8 (H*W bytes).
  const W = cam.W, H = cam.H; const packed = new Uint8Array(W * H * 5);
  const depth = packed.subarray(0, W * H * 4), sem = packed.subarray(W * H * 4);
  for (let r = 0; r < H; r++) {
    const src = (H - 1 - r) * W; depth.set(depthBytes.subarray(src * 4, (src + W) * 4), r * W * 4);
    for (let c = 0; c < W; c++) sem[r * W + c] = semRGBA[(src + c) * 4];
  }
  if (opts && opts.chunked) return Object.assign(stash(packed), { ms: performance.now() - t0 });
  return { depth: bytesToB64(depth), semantic: bytesToB64(sem), ms: performance.now() - t0 };
}

// ---- optional chase-camera parameters (additive; video tooling only, never used by renderStereo/renderGT)
// state.chase = {back_m, up_m, yaw_smooth, lookahead_m}: camera `back_m` behind and `up_m` above the
// rover base along the WORLD heading `yaw_smooth` (rad; e.g. critically damped, planned in
// chase_replay.py, which also shortens back_m/up_m when a tree or rock would cut the sightline),
// looking at the point `lookahead_m` ahead. Any missing field falls back to the legacy fixed offset.
const CHASE_LOOK_M = 2.5, CHASE_LOOK_UP_M = 0.4, CHASE_TERRAIN_CLEAR_M = 1.0; // legacy look-at / ground clearance (m)
function chaseParams(state, yaw) {
  const c = state.chase || {}; const num = (v, d) => (Number.isFinite(v) ? v : d);
  return { back: num(c.back_m, CHASE_BACK_M), up: num(c.up_m, CHASE_UP_M), yaw: num(c.yaw_smooth, yaw), look: num(c.lookahead_m, CHASE_LOOK_M) };
}

function renderChase(state, w, h, opts) {
  const t0 = performance.now();
  const key = `${w}x${h}`;
  if (!T.chase[key]) T.chase = { [key]: { hdr: mkRT(w, h, THREE.HalfFloatType, 4), out: mkRT(w, h, THREE.UnsignedByteType), buf: new Uint8Array(w * h * 4), rgb: new Uint8Array(w * h * 3) } };
  const C = T.chase[key];
  applyDynamic(state);
  const pose = state.pose; const Mb = poseMatrix(pose);
  const cp = chaseParams(state, pose[5]); const yaw = cp.yaw; const base = new THREE.Vector3(pose[0], pose[1], pose[2]);
  const eye = base.clone().add(new THREE.Vector3(-Math.cos(yaw) * cp.back, -Math.sin(yaw) * cp.back, cp.up));
  if (world.terrain) eye.z = Math.max(eye.z, world.terrain.heightAt(eye.x, eye.y) + CHASE_TERRAIN_CLEAR_M);
  chaseCam.aspect = w / h; chaseCam.updateProjectionMatrix();
  chaseCam.position.copy(eye); chaseCam.up.set(0, 0, 1);
  chaseCam.lookAt(base.clone().add(new THREE.Vector3(Math.cos(yaw) * cp.look, Math.sin(yaw) * cp.look, CHASE_LOOK_UP_M))); chaseCam.updateMatrixWorld(true);
  if (world.vehicle) { world.vehicle.visible = true; world.vehicle.matrixAutoUpdate = false; world.vehicle.matrix.copy(Mb); world.vehicle.matrixWorldNeedsUpdate = true; }
  const fu = updateFrameUniforms(state, chaseCam, false);
  positionShadow(pose, true);
  renderHDR(chaseCam, C.hdr);
  if (world.vehicle) world.vehicle.visible = false;
  const u = postMat.uniforms;
  u.tL.value = C.hdr.texture; u.tR.value = C.hdr.texture; u.uMode.value = 1; u.uSize.value.set(w, h); u.uPP.value.set((w - 1) / 2, (h - 1) / 2);
  u.uGainL.value = fu.gain; u.uNoiseDN.value = 0.0; u.uVignette.value = VIGNETTE * 0.8; u.uBlur.value = 0.0;
  u.uSun.value.set(-1e4, -1e4, 0); u.uVeil.value = 0.0; u.uHazeAmt.value = 0.25 * Math.min(1, fu.lv.dust) * fu.gain; u.uSeed.value = 1;
  renderer.setRenderTarget(C.out); renderer.render(postScene, postCam);
  renderer.readRenderTargetPixels(C.out, 0, 0, w, h, C.buf);
  const n = w * h; for (let k = 0, q = 0, m = 0; k < n; k++, q += 4, m += 3) { C.rgb[m] = C.buf[q]; C.rgb[m + 1] = C.buf[q + 1]; C.rgb[m + 2] = C.buf[q + 2]; }
  if (opts && opts.chunked) return Object.assign(stash(C.rgb), { ms: performance.now() - t0 });
  return { rgb: bytesToB64(C.rgb), ms: performance.now() - t0 };
}

/** Debug helper for tests: project world points with the renderer's own matrices -> OpenCV px. */
function projectPoints(pose, pts, eye) {
  placeStereo(pose); const c = eye === 1 ? camR : camL; const v = new THREE.Vector3();
  return pts.map((p) => { v.set(p[0], p[1], p[2]).project(c); return [(v.x + 1) * cam.W / 2 - 0.5, (1 - v.y) * cam.H / 2 - 0.5]; });
}

allocTargets();
Object.assign(window, { configureCameras, loadScenario, renderStereo, renderGT, renderChase, rendererInfo, projectPoints, mgChunk, __mgReady: true });
