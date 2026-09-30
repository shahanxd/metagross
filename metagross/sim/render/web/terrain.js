// METAGROSS renderer: terrain mesh + material from the scenario heightmap.
//
// Heightmap: float32 row-major, rows along +y, cols along +x, res 0.05 m; cell
// (i, j) centre = origin + (j*res, i*res). Vertices sit on cell centres, so the
// rendered surface interpolates the heightmap exactly at every kept vertex.
//
// LOD: the grid is split into 64x64-cell chunks. Chunks within DITCH_KEEP_M of a
// ditch/crest hazard keep every vertex (0.05 m) so ditch lips and walls stay crisp;
// other chunks use the coarsest step in {8,4,2,1} whose bilinear reconstruction
// error stays below LOD_TOL_M. Shading always uses the full-resolution normal
// texture, so decimated chunks still shade at 0.05 m. Skirts hide LOD cracks.

import * as THREE from 'three';
import { NOISE_GLSL, TERRAIN_COMMON_GLSL, TERRAIN_SURFACE_GLSL } from './shaders.js';

const CHUNK = 64;            // cells per chunk side
const LOD_STEPS = [8, 4, 2]; // candidate vertex strides (cells), coarsest first
const LOD_TOL_M = 0.015;     // max vertical reconstruction error for a decimated chunk (m)
const DITCH_KEEP_M = 15.0;   // full-res radius around ditch / crest polylines (m)
const SKIRT_M = 0.15;        // skirt depth (m) >> LOD_TOL_M
const CAVITY_RADIUS_M = 1.0; // radius of the local-mean filter for cavity / AO (m)
const CAVITY_SCALE = 1.5;    // cavity metres -> texture units around 0.5
const APRON_STEP_CELLS = 10; // apron perimeter sampling (cells)
const APRON_RINGS_M = [0.5, 3, 10, 25, 50, 90, 140, 185]; // ring distances beyond the grid edge (m), < far clip
const APRON_HILL_M = 9.0;    // max height of horizon hills (m)

export function b64ToBytes(b64) {
  if (Uint8Array.fromBase64) return Uint8Array.fromBase64(b64);
  const s = atob(b64); const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return u;
}

function segDist(px, py, ax, ay, bx, by) {
  const vx = bx - ax, vy = by - ay; const L2 = vx * vx + vy * vy;
  let t = L2 > 0 ? ((px - ax) * vx + (py - ay) * vy) / L2 : 0; t = Math.max(0, Math.min(1, t));
  const dx = ax + t * vx - px, dy = ay + t * vy - py; return Math.hypot(dx, dy);
}

export class Terrain {
  /** @param {object} ter scenario.terrain  @param {object[]} hazards scenario.hazards */
  constructor(ter, hazards) {
    this.ny = ter.shape[0]; this.nx = ter.shape[1];
    this.res = ter.res_m; this.x0 = ter.origin_xy[0]; this.y0 = ter.origin_xy[1];
    const hb = b64ToBytes(ter.height_b64);
    if (hb.byteLength !== this.nx * this.ny * 4) throw new Error(`height_b64 has ${hb.byteLength} bytes, expected ${this.nx * this.ny * 4}`);
    this.h = new Float32Array(hb.buffer, hb.byteOffset, this.nx * this.ny);
    this.mat = b64ToBytes(ter.material_b64);
    if (this.mat.length !== this.nx * this.ny) throw new Error('material_b64 size mismatch');
    this.hazardLines = [];
    for (const hz of hazards || []) {
      if ((hz.type === 'ditch' || hz.type === 'crest') && hz.polyline) this.hazardLines.push({ pts: hz.polyline, pad: (hz.width || 0) * 0.5 });
    }
    this.stats = { chunks: 0, fullResChunks: 0, vertices: 0, triangles: 0 };
    this._buildTextures();
  }

  /** Bilinear terrain height at world (x, y), clamped to the grid (m). */
  heightAt(x, y) {
    const fx = Math.min(Math.max((x - this.x0) / this.res, 0), this.nx - 1.001);
    const fy = Math.min(Math.max((y - this.y0) / this.res, 0), this.ny - 1.001);
    const j = Math.floor(fx), i = Math.floor(fy); const u = fx - j, v = fy - i; const nx = this.nx, h = this.h;
    const a = h[i * nx + j], b = h[i * nx + j + 1], c = h[(i + 1) * nx + j], d = h[(i + 1) * nx + j + 1];
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
  }

  _buildTextures() {
    const { nx, ny, h, res } = this; const n = nx * ny;
    // Normals from central differences + cavity = h - local mean (integral image box filter).
    const nrm = new Uint8Array(n * 4);
    const I = new Float64Array((nx + 1) * (ny + 1));
    for (let i = 0; i < ny; i++) {
      let row = 0;
      for (let j = 0; j < nx; j++) { row += h[i * nx + j]; I[(i + 1) * (nx + 1) + j + 1] = I[i * (nx + 1) + j + 1] + row; }
    }
    const R = Math.max(1, Math.round(CAVITY_RADIUS_M / res));
    for (let i = 0; i < ny; i++) {
      const im = Math.max(i - 1, 0), ip = Math.min(i + 1, ny - 1);
      const r0 = Math.max(i - R, 0), r1 = Math.min(i + R, ny - 1);
      for (let j = 0; j < nx; j++) {
        const jm = Math.max(j - 1, 0), jp = Math.min(j + 1, nx - 1);
        const dzdx = (h[i * nx + jp] - h[i * nx + jm]) / ((jp - jm) * res);
        const dzdy = (h[ip * nx + j] - h[im * nx + j]) / ((ip - im) * res);
        const inv = 1 / Math.sqrt(dzdx * dzdx + dzdy * dzdy + 1);
        const c0 = Math.max(j - R, 0), c1 = Math.min(j + R, nx - 1);
        const sum = I[(r1 + 1) * (nx + 1) + c1 + 1] - I[r0 * (nx + 1) + c1 + 1] - I[(r1 + 1) * (nx + 1) + c0] + I[r0 * (nx + 1) + c0];
        const cav = h[i * nx + j] - sum / ((r1 - r0 + 1) * (c1 - c0 + 1));
        const k = (i * nx + j) * 4;
        nrm[k] = Math.round((-dzdx * inv * 0.5 + 0.5) * 255);
        nrm[k + 1] = Math.round((-dzdy * inv * 0.5 + 0.5) * 255);
        nrm[k + 2] = Math.round((inv * 0.5 + 0.5) * 255);
        nrm[k + 3] = Math.round(Math.min(Math.max(0.5 + cav * CAVITY_SCALE, 0), 1) * 255);
      }
    }
    this.nrm = nrm;
    const A = new Uint8Array(n * 4), B = new Uint8Array(n * 4);
    for (let k = 0; k < n; k++) {
      const m = this.mat[k];
      if (m < 4) A[k * 4 + m] = 255; else if (m < 6) B[k * 4 + (m - 4)] = 255; else A[k * 4] = 255;
    }
    const mk = (data) => {
      const t = new THREE.DataTexture(data, nx, ny, THREE.RGBAFormat, THREE.UnsignedByteType);
      t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearFilter; t.generateMipmaps = false;
      t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.flipY = false; t.colorSpace = THREE.NoColorSpace; t.needsUpdate = true;
      return t;
    };
    this.texNrm = mk(nrm); this.texA = mk(A); this.texB = mk(B);
    this.gridUniforms = {
      uMatA: { value: this.texA }, uMatB: { value: this.texB }, uNrm: { value: this.texNrm },
      uGridOrigin: { value: new THREE.Vector2(this.x0, this.y0) }, uGridRes: { value: this.res },
      uGridSize: { value: new THREE.Vector2(nx, ny) },
    };
  }

  _nearHazard(xa, ya, xb, yb) {
    const cx = 0.5 * (xa + xb), cy = 0.5 * (ya + yb); const halfDiag = 0.5 * Math.hypot(xb - xa, yb - ya);
    for (const L of this.hazardLines) {
      const p = L.pts;
      for (let k = 0; k + 1 < p.length; k++) if (segDist(cx, cy, p[k][0], p[k][1], p[k + 1][0], p[k + 1][1]) <= DITCH_KEEP_M + L.pad + halfDiag) return true;
      if (p.length === 1 && Math.hypot(p[0][0] - cx, p[0][1] - cy) <= DITCH_KEEP_M + halfDiag) return true;
    }
    return false;
  }

  static _lattice(a, b, s) { const out = []; for (let k = a; k < b; k += s) out.push(k); out.push(b); return out; }

  _lodError(rows, cols, i0, i1, j0, j1) {
    const { h, nx } = this; let err = 0; let ri = 0;
    for (let i = i0; i <= i1; i++) {
      while (ri + 1 < rows.length - 1 && rows[ri + 1] <= i) ri++;
      const ia = rows[ri], ib = rows[Math.min(ri + 1, rows.length - 1)]; const v = ib > ia ? (i - ia) / (ib - ia) : 0;
      let ci = 0;
      for (let j = j0; j <= j1; j++) {
        while (ci + 1 < cols.length - 1 && cols[ci + 1] <= j) ci++;
        const ja = cols[ci], jb = cols[Math.min(ci + 1, cols.length - 1)]; const u = jb > ja ? (j - ja) / (jb - ja) : 0;
        const z = (h[ia * nx + ja] * (1 - u) + h[ia * nx + jb] * u) * (1 - v) + (h[ib * nx + ja] * (1 - u) + h[ib * nx + jb] * u) * v;
        const e = Math.abs(z - h[i * nx + j]); if (e > err) { err = e; if (err > LOD_TOL_M) return err; }
      }
    }
    return err;
  }

  _chunkGeometry(rows, cols) {
    const { h, nx, nrm, res, x0, y0 } = this; const nr = rows.length, nc = cols.length;
    const nGrid = nr * nc; const nSkirt = 2 * (nr + nc);
    const pos = new Float32Array((nGrid + nSkirt) * 3); const nor = new Float32Array((nGrid + nSkirt) * 3);
    const put = (k, i, j, dz) => {
      pos[k * 3] = x0 + j * res; pos[k * 3 + 1] = y0 + i * res; pos[k * 3 + 2] = h[i * nx + j] - dz;
      const q = (i * nx + j) * 4; nor[k * 3] = nrm[q] / 127.5 - 1; nor[k * 3 + 1] = nrm[q + 1] / 127.5 - 1; nor[k * 3 + 2] = nrm[q + 2] / 127.5 - 1;
    };
    for (let a = 0; a < nr; a++) for (let b = 0; b < nc; b++) put(a * nc + b, rows[a], cols[b], 0);
    const idx = [];
    for (let a = 0; a + 1 < nr; a++) for (let b = 0; b + 1 < nc; b++) {
      const p = a * nc + b, q = p + 1, r = p + nc, s = r + 1;
      idx.push(p, q, s, p, s, r); // CCW seen from +z
    }
    // Skirts: duplicate each border vertex SKIRT_M lower and stitch a vertical strip.
    let k = nGrid;
    const border = (list) => {
      const start = k;
      for (const [a, b] of list) { put(k, rows[a], cols[b], SKIRT_M); k++; }
      for (let m = 0; m + 1 < list.length; m++) {
        const t0 = list[m][0] * nc + list[m][1], t1 = list[m + 1][0] * nc + list[m + 1][1];
        const s0 = start + m, s1 = start + m + 1;
        idx.push(t0, s0, s1, t0, s1, t1, t0, s1, s0, t0, t1, s1); // both windings: skirt visible from either side
      }
    };
    border(Array.from({ length: nc }, (_, b) => [0, b]));
    border(Array.from({ length: nc }, (_, b) => [nr - 1, b]));
    border(Array.from({ length: nr }, (_, a) => [a, 0]));
    border(Array.from({ length: nr }, (_, a) => [a, nc - 1]));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setIndex(idx);
    g.computeBoundingSphere(); g.computeBoundingBox();
    this.stats.vertices += nGrid + nSkirt; this.stats.triangles += idx.length / 3;
    return g;
  }

  /** Build chunk meshes sharing `material`. Returns a THREE.Group. */
  buildMeshes(material) {
    const group = new THREE.Group(); group.name = 'terrain';
    const { nx, ny, res, x0, y0 } = this;
    for (let i0 = 0; i0 < ny - 1; i0 += CHUNK) {
      const i1 = Math.min(i0 + CHUNK, ny - 1);
      for (let j0 = 0; j0 < nx - 1; j0 += CHUNK) {
        const j1 = Math.min(j0 + CHUNK, nx - 1);
        let rows = Terrain._lattice(i0, i1, 1), cols = Terrain._lattice(j0, j1, 1);
        const full = this._nearHazard(x0 + j0 * res, y0 + i0 * res, x0 + j1 * res, y0 + i1 * res);
        if (!full) {
          for (const s of LOD_STEPS) {
            const r = Terrain._lattice(i0, i1, s), c = Terrain._lattice(j0, j1, s);
            if (this._lodError(r, c, i0, i1, j0, j1) <= LOD_TOL_M) { rows = r; cols = c; break; }
          }
        } else this.stats.fullResChunks++;
        const mesh = new THREE.Mesh(this._chunkGeometry(rows, cols), material);
        mesh.receiveShadow = true; mesh.castShadow = false; mesh.userData.sem = 'terrain';
        mesh.matrixAutoUpdate = false;
        group.add(mesh); this.stats.chunks++;
      }
    }
    return group;
  }

  /**
   * Apron: a coarse ring mesh from the grid boundary out to APRON_RINGS_M, rising into
   * low hills so the scenario edge and the far clip plane are never visible. It is
   * scenery only (the world/referee terrain ends at the grid boundary).
   */
  buildApron(material, seed) {
    const { nx, ny, res, x0, y0, h } = this; const S = APRON_STEP_CELLS;
    const per = [];
    for (let j = 0; j < nx - 1; j += S) per.push([0, j]);
    for (let i = 0; i < ny - 1; i += S) per.push([i, nx - 1]);
    for (let j = nx - 1; j > 0; j -= S) per.push([ny - 1, j]);
    for (let i = ny - 1; i > 0; i -= S) per.push([i, 0]);
    const cx = x0 + 0.5 * (nx - 1) * res, cy = y0 + 0.5 * (ny - 1) * res;
    let hMean = 0; for (const [i, j] of per) hMean += h[i * nx + j]; hMean /= per.length;
    const R = APRON_RINGS_M; const nP = per.length, nR = R.length + 1;
    const pos = new Float32Array(nP * nR * 3);
    const rnd = (a, k) => 0.5 + 0.5 * Math.sin(a * k + seed * 0.37 + k * 1.7);
    for (let p = 0; p < nP; p++) {
      const [i, j] = per[p]; const bx = x0 + j * res, by = y0 + i * res, bz = h[i * nx + j] - 0.005;
      let dx = bx - cx, dy = by - cy; const dl = Math.hypot(dx, dy) || 1; dx /= dl; dy /= dl;
      const a = Math.atan2(dy, dx);
      const hill = 0.45 * rnd(a, 3) + 0.3 * rnd(a, 7) + 0.25 * rnd(a, 13);
      for (let r = 0; r < nR; r++) {
        const d = r === 0 ? 0 : R[r - 1]; const k = (p * nR + r) * 3;
        const blend = Math.min(1, d / 20);
        const rise = APRON_HILL_M * hill * THREE.MathUtils.smoothstep(d, 30, 140);
        pos[k] = bx + dx * d; pos[k + 1] = by + dy * d; pos[k + 2] = (1 - blend) * bz + blend * hMean + rise;
      }
    }
    const idx = [];
    for (let p = 0; p < nP; p++) {
      const q = (p + 1) % nP;
      for (let r = 0; r + 1 < nR; r++) {
        const a = p * nR + r, b = q * nR + r, c = q * nR + r + 1, d = p * nR + r + 1;
        idx.push(a, d, c, a, c, b); // perimeter is CCW seen from +z, so these face up
      }
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setIndex(idx);
    g.computeVertexNormals(); g.computeBoundingSphere();
    const mesh = new THREE.Mesh(g, material); mesh.receiveShadow = true; mesh.userData.sem = 'terrain'; mesh.name = 'apron'; mesh.matrixAutoUpdate = false;
    return mesh;
  }

  /** Terrain visual material: MeshStandardMaterial with the procedural surface patched in. */
  makeMaterial(timeUniform) {
    const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1.0, metalness: 0.0 });
    const grid = this.gridUniforms;
    m.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, grid); sh.uniforms.uTime = timeUniform;
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec3 vMgWorld;\nvarying vec3 vMgNW;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvMgWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvMgNW = normalize(mat3(modelMatrix) * objectNormal);');
      sh.fragmentShader = sh.fragmentShader
        .replace('#include <common>', `#include <common>\nvarying vec3 vMgWorld;\nvarying vec3 vMgNW;\n${NOISE_GLSL}\n${TERRAIN_COMMON_GLSL}\n${TERRAIN_SURFACE_GLSL}`)
        .replace('#include <map_fragment>', `
          vec3 mgDx = dFdx(vMgWorld), mgDy = dFdy(vMgWorld);
          // footprint (m/px): horizontal image axis dominates stereo; vertical is down-weighted
          float mgFp = max(length(mgDx), 0.35 * length(mgDy));
          MgSurf mgS = mg_terrainSurface(vMgWorld, mgFp, vMgNW);
          diffuseColor.rgb = mgS.albedo;`)
        .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = mgS.rough;')
        .replace('#include <normal_fragment_maps>', `
          normal = normalize((viewMatrix * vec4(mgS.nW, 0.0)).xyz);
          float mgWater = texture(uMatB, mg_terrainUV(mg_warp(vMgWorld.xy))).g;
          float mgRip = mg_vnoise(vMgWorld.xy * 7.0 + vec2(uTime * 0.6, uTime * 0.4)) + 0.5 * mg_vnoise(vMgWorld.xy * 17.0 - uTime * 0.9);
          float mgH = mix(mgS.bump, mgRip, mgWater);
          normal = mg_perturb(-vViewPosition, normal, mgH, mix(mgS.bumpAmp, 0.02, mgWater));`);
    };
    m.customProgramCacheKey = () => 'mg_terrain_v1';
    return m;
  }
}
