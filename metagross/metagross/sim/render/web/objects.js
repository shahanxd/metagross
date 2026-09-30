// METAGROSS renderer: static scene objects, dynamic actors and the chase-view
// vehicle model. All geometry is procedural and seeded (object "seed" field), so
// the same scenario always produces the same rocks/trees.
//
// World frame: x east, y north, z up (metres). Object anchors follow the scenario
// schema (metagross/contracts/scenario.py).

import * as THREE from 'three';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { NOISE_GLSL } from './shaders.js';

export const LAYER_CHASE_ONLY = 1; // vehicle model: visible to the chase camera only

// ------------------------------------------------------------ seeded helpers
export function mulberry32(seed) {
  let a = (seed >>> 0) || 0x9e3779b9;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function hash3(ix, iy, iz, seed) {
  let h = Math.imul(ix, 374761393) ^ Math.imul(iy, 668265263) ^ Math.imul(iz, 2147483647) ^ Math.imul(seed, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177); h ^= h >>> 16; return (h >>> 0) / 4294967296;
}
function vnoise3(x, y, z, seed) {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z); const fx = x - ix, fy = y - iy, fz = z - iz;
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy), w = fz * fz * (3 - 2 * fz);
  const L = (a, b, t) => a + (b - a) * t; const H = (a, b, c) => hash3(ix + a, iy + b, iz + c, seed);
  return L(L(L(H(0, 0, 0), H(1, 0, 0), u), L(H(0, 1, 0), H(1, 1, 0), u), v), L(L(H(0, 0, 1), H(1, 0, 1), u), L(H(0, 1, 1), H(1, 1, 1), u), v), w);
}
function fbm3(x, y, z, seed, oct = 4) {
  let s = 0, a = 0.5, n = 0; for (let i = 0; i < oct; i++) { s += a * vnoise3(x, y, z, seed + i * 17); n += a; x *= 2.1; y *= 2.1; z *= 2.1; a *= 0.5; } return s / n;
}

/** Unit icosphere displaced radially by seeded fbm; smooth normals. */
function blob(detail, seed, amp, freq) {
  let g = new THREE.IcosahedronGeometry(1, detail); g.deleteAttribute('uv'); g.deleteAttribute('normal');
  g = mergeVertices(g);
  const p = g.attributes.position;
  for (let k = 0; k < p.count; k++) {
    const x = p.getX(k), y = p.getY(k), z = p.getZ(k);
    const r = 1 + amp * (2 * fbm3(x * freq + 11.3, y * freq + 3.7, z * freq + 5.1, seed) - 1);
    p.setXYZ(k, x * r, y * r, z * r);
  }
  g.computeVertexNormals();
  return g;
}
function stripToPN(g) {
  const out = g.index ? g : mergeVertices(g);
  for (const k of Object.keys(out.attributes)) if (k !== 'position' && k !== 'normal') out.deleteAttribute(k);
  if (!out.attributes.normal) out.computeVertexNormals();
  return out;
}

// ------------------------------------------------------------ materials
// kind: 'rock' | 'bark' | 'leaf' | 'cloth' | 'box'
export function makeObjMaterial(kind, base) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, metalness: 0.0 });
  const bases = { rock: [0.46, 0.44, 0.41], bark: [0.30, 0.23, 0.17], leaf: [0.20, 0.30, 0.10], cloth: base || [0.3, 0.3, 0.35], box: base || [0.55, 0.42, 0.27] };
  const b = bases[kind];
  const code = {
    rock: `float n1 = mg_fbm3(wp * 1.3, 3); float n2 = mg_vnoise3(wp * 18.0); float n3 = mg_vnoise3(wp * 45.0);
           vec3 c = mix(uBase * 0.7, uBase * 1.15, n1) * (0.75 + 0.35 * n2 + 0.2 * n3);
           c = mix(c, mg_srgb(vec3(0.52, 0.54, 0.36)), smoothstep(0.62, 0.72, mg_fbm3(wp * 3.0 + 7.0, 2)) * 0.7 * smoothstep(0.0, 0.6, nW.z));
           c *= mix(0.55, 1.0, smoothstep(-0.2, 0.5, nW.z));
           outAlb = c; outRough = 0.85; outBump = n2 * 0.7 + n3 * 0.3; outAmp = 0.01;`,
    bark: `float n1 = mg_vnoise3(wp * vec3(38.0, 38.0, 3.0)); float n2 = mg_vnoise3(wp * 60.0);
           outAlb = uBase * (0.6 + 0.6 * n1) * (0.85 + 0.3 * n2); outRough = 0.95; outBump = n1; outAmp = 0.012;`,
    leaf: `float n1 = mg_vnoise3(wp * 2.2); float n2 = mg_vnoise3(wp * 26.0); float n3 = mg_vnoise3(wp * 70.0);
           vec3 c = mix(uBase, mg_srgb(vec3(0.34, 0.40, 0.13)), smoothstep(0.35, 0.8, n1));
           c = mix(c, mg_srgb(vec3(0.13, 0.19, 0.07)), smoothstep(0.55, 0.3, mg_vnoise3(wp * 6.0)) * 0.6);
           float gaps = smoothstep(0.25, 0.1, n2 * 0.7 + n3 * 0.3);
           c *= (0.4 + 1.0 * (0.6 * n2 + 0.4 * n3)) * (1.0 - 0.7 * gaps);
           c *= mix(0.45, 1.0, smoothstep(-0.7, 0.5, nW.z));
           outAlb = c; outRough = 0.8; outBump = n2 * 0.6 + n3 * 0.4; outAmp = 0.035;`,
    cloth: `float n1 = mg_vnoise3(wp * 40.0); outAlb = uBase * (0.85 + 0.3 * n1); outRough = 0.9; outBump = n1; outAmp = 0.002;`,
    box: `float n1 = mg_vnoise3(wp * 25.0); float n2 = mg_vnoise3(wp * 3.0); outAlb = uBase * (0.8 + 0.25 * n1 + 0.2 * n2); outRough = 0.8; outBump = n1; outAmp = 0.003;`,
  }[kind];
  const uBase = { value: new THREE.Vector3(...b.map((v) => Math.pow(v, 2.2))) };
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uBase = uBase;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vMgWorld;\nvarying vec3 vMgNW;')
      .replace('#include <project_vertex>', '#include <project_vertex>\nvMgWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvMgNW = normalize(mat3(modelMatrix) * objectNormal);');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>\nvarying vec3 vMgWorld;\nvarying vec3 vMgNW;\nuniform vec3 uBase;\n${NOISE_GLSL}`)
      .replace('#include <map_fragment>', `vec3 wp = vMgWorld; vec3 nW = normalize(vMgNW);
          vec3 outAlb; float outRough; float outBump; float outAmp;
          ${code}
          diffuseColor.rgb = outAlb;`)
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = outRough;')
      .replace('#include <normal_fragment_maps>', 'normal = mg_perturb(-vViewPosition, normal, outBump, outAmp);');
  };
  m.customProgramCacheKey = () => 'mg_obj_' + kind;
  return m;
}

// ------------------------------------------------------------ static objects
/**
 * Build merged meshes for all static objects (one draw call per material).
 * @param {object[]} objects scenario.objects
 * @param {(x:number,y:number)=>number} heightAt terrain height (m)
 */
export function buildStaticObjects(objects, heightAt) {
  const rocks = [], bark = [], leaves = [];
  const M = new THREE.Matrix4(), Q = new THREE.Quaternion(), S = new THREE.Vector3(), T = new THREE.Vector3(), E = new THREE.Euler();
  for (const o of objects || []) {
    const seed = (o.seed ?? 0) | 0; const rnd = mulberry32(seed * 7919 + 13);
    if (o.type === 'rock') {
      const r = o.radius, sq = o.squash ?? 0.7;
      const g = blob(3, seed, 0.28, 1.2);
      E.set((rnd() - 0.5) * 0.3, (rnd() - 0.5) * 0.3, rnd() * Math.PI * 2); Q.setFromEuler(E);
      M.compose(T.set(o.xyz[0], o.xyz[1], o.xyz[2]), Q, S.set(r * (0.9 + 0.3 * rnd()), r * (0.9 + 0.2 * rnd()), r * sq));
      g.applyMatrix4(M); rocks.push(stripToPN(g));
    } else if (o.type === 'tree') {
      const [x, y] = o.xy; const z = heightAt(x, y) - 0.1; const H = o.height, tr = o.trunk_r, cr = o.canopy_r;
      const trunkH = Math.max(H - 1.2 * cr, 0.4 * H);
      let t = new THREE.CylinderGeometry(tr * 0.65, tr * 1.2, trunkH + 0.1, 10, 4); t.rotateX(Math.PI / 2); t.translate(0, 0, (trunkH + 0.1) / 2);
      const lean = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler((rnd() - 0.5) * 0.08, (rnd() - 0.5) * 0.08, 0));
      t.applyMatrix4(lean); t.translate(x, y, z); bark.push(stripToPN(t));
      // Canopy: a core blob plus clustered lobes on an irregular ellipsoid shell, sagging lower lobes.
      const nb = 9 + Math.floor(rnd() * 5); const top = new THREE.Vector3(0, 0, H - cr * 1.0).applyMatrix4(lean);
      const core = blob(2, seed * 31 + 99, 0.3, 1.4); core.scale(cr * 0.75, cr * 0.75, cr * 0.65); core.translate(x + top.x, y + top.y, z + top.z);
      leaves.push(stripToPN(core));
      for (let k = 0; k < nb; k++) {
        const a = rnd() * Math.PI * 2, el = (rnd() - 0.35) * 1.3;
        const rr = cr * (0.55 + 0.25 * rnd()) * Math.cos(el), dz = cr * 0.6 * Math.sin(el) - 0.1 * cr;
        const br = cr * (0.3 + 0.22 * rnd());
        const g = blob(2, seed * 31 + k, 0.4, 1.8);
        g.scale(br, br, br * 0.8); g.translate(x + top.x + rr * Math.cos(a), y + top.y + rr * Math.sin(a), z + top.z + dz);
        leaves.push(stripToPN(g));
      }
    } else if (o.type === 'bush') {
      const [x, y] = o.xy; const z = heightAt(x, y); const R = o.radius, H = o.height;
      const nb = 3 + Math.floor(rnd() * 3);
      for (let k = 0; k < nb; k++) {
        const a = rnd() * Math.PI * 2, rr = R * 0.45 * rnd(); const br = R * (0.5 + 0.25 * rnd());
        const g = blob(2, seed * 37 + k, 0.4, 1.8);
        g.scale(br, br, H * 0.55); g.translate(x + rr * Math.cos(a), y + rr * Math.sin(a), z + H * 0.42);
        leaves.push(stripToPN(g));
      }
    } else if (o.type === 'log') {
      const [x, y] = o.xy; const r = o.radius, L = o.length;
      const g = new THREE.CylinderGeometry(r, r * 1.05, L, 14, 3); g.rotateZ(Math.PI / 2); g.rotateZ(o.yaw ?? 0);
      g.translate(x, y, heightAt(x, y) + r * 0.8); bark.push(stripToPN(g));
    }
  }
  const group = new THREE.Group(); group.name = 'static_objects';
  const add = (list, mat, name) => {
    if (!list.length) return;
    const mesh = new THREE.Mesh(mergeGeometries(list, false), mat); mesh.name = name;
    mesh.castShadow = true; mesh.receiveShadow = true; mesh.userData.sem = 'object'; mesh.matrixAutoUpdate = false;
    group.add(mesh);
  };
  add(rocks, makeObjMaterial('rock'), 'rocks'); add(bark, makeObjMaterial('bark'), 'bark'); add(leaves, makeObjMaterial('leaf'), 'foliage');
  return group;
}

// ------------------------------------------------------------ dynamic actors
function capsuleZ(r, len) { const g = new THREE.CapsuleGeometry(r, len, 4, 10); g.rotateX(Math.PI / 2); return g; }

/** Articulated capsule walker, `size` = [sx, sy, sz] (sz = standing height, m). Faces local +x. */
function makeWalker(size, seed) {
  const rnd = mulberry32(seed + 1); const H = size[2] || 1.7;
  const shirt = makeObjMaterial('cloth', [0.2 + 0.5 * rnd(), 0.2 + 0.3 * rnd(), 0.2 + 0.3 * rnd()]);
  const pants = makeObjMaterial('cloth', [0.12, 0.13, 0.16]); const skin = makeObjMaterial('cloth', [0.55, 0.40, 0.30]);
  const root = new THREE.Group();
  const torso = new THREE.Mesh(capsuleZ(0.16 * H / 1.7, 0.36 * H), shirt); torso.position.z = 0.70 * H; root.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.11 * H / 1.7, 14, 10), skin); head.position.z = 0.93 * H; root.add(head);
  const limbs = {};
  const limb = (name, mat, px, py, pz, r, len) => {
    const pivot = new THREE.Group(); pivot.position.set(px, py, pz);
    const m = new THREE.Mesh(capsuleZ(r, len), mat); m.position.z = -len / 2 - r; pivot.add(m); root.add(pivot); limbs[name] = pivot;
  };
  limb('legL', pants, 0, 0.09 * H / 1.7, 0.50 * H, 0.07 * H / 1.7, 0.40 * H); limb('legR', pants, 0, -0.09 * H / 1.7, 0.50 * H, 0.07 * H / 1.7, 0.40 * H);
  limb('armL', shirt, 0, 0.21 * H / 1.7, 0.84 * H, 0.05 * H / 1.7, 0.30 * H); limb('armR', shirt, 0, -0.21 * H / 1.7, 0.84 * H, 0.05 * H / 1.7, 0.30 * H);
  root.userData.animate = (phase) => {
    const s = Math.sin(phase) * 0.45;
    limbs.legL.rotation.y = s; limbs.legR.rotation.y = -s; limbs.armL.rotation.y = -0.8 * s; limbs.armR.rotation.y = 0.8 * s;
  };
  root.userData.stride = 1.4; // m per gait cycle
  return root;
}

export function makeDynamic(d, index) {
  let obj;
  const size = d.size || [0.5, 0.5, 1.7];
  if (d.type === 'walker') obj = makeWalker(size, index * 101 + 7);
  else if (d.type === 'boulder') {
    const r = (size[0] + size[1] + size[2]) / 6; const g = blob(3, index * 13 + 5, 0.22, 1.2); g.scale(size[0] / 2, size[1] / 2, size[2] / 2);
    obj = new THREE.Group(); const m = new THREE.Mesh(g, makeObjMaterial('rock')); m.position.z = size[2] / 2 * 0.9; obj.add(m); obj.userData.radius = r;
  } else {
    obj = new THREE.Group(); const m = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), makeObjMaterial('box')); m.position.z = size[2] / 2; obj.add(m);
  }
  obj.traverse((c) => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; c.userData.sem = 'object'; } });
  obj.name = `dynamic_${index}`; obj.userData.spec = d;
  return obj;
}

// ------------------------------------------------------------ chase-view vehicle
/** Simple skid-steer model in body frame (x fwd, y left, z up), for the chase camera only. */
export function makeVehicle(veh, camT) {
  const g = new THREE.Group(); g.name = 'vehicle';
  const L = veh.length_m, W = veh.width_m, r = veh.wheel_radius_m, tw = veh.track_width_m;
  const body = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(0.24, 0.27, 0.17, THREE.SRGBColorSpace), roughness: 0.6, metalness: 0.2 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.9 });
  const metal = new THREE.MeshStandardMaterial({ color: 0x8a8d90, roughness: 0.4, metalness: 0.6 });
  const b = new THREE.Mesh(new THREE.BoxGeometry(L, W - 0.16, 0.22), body); b.position.z = r + 0.08; g.add(b);
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.1, 18), dark); w.position.set(sx * (L / 2 - r), sy * tw / 2, r); g.add(w);
  }
  const cx = camT[3], cz = camT[11];
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, cz - r - 0.19, 8), metal); mast.rotation.x = Math.PI / 2; mast.position.set(cx - 0.05, 0, (cz + r + 0.19) / 2 - 0.03); g.add(mast);
  const bar = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.2, 0.04), dark); bar.position.set(cx, 0, cz); g.add(bar);
  g.traverse((c) => { c.layers.set(LAYER_CHASE_ONLY); if (c.isMesh) c.castShadow = true; });
  return g;
}
