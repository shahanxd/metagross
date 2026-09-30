// METAGROSS renderer: GLSL building blocks shared by the visual, semantic and
// post passes. Everything is procedural (no texture downloads) and hash-based,
// so a given world position always produces the same albedo on every GPU.
//
// Conventions: world frame x east, y north, z up (metres). All colours below are
// authored in sRGB and converted to linear with mg_srgb() before lighting.

export const NOISE_GLSL = /* glsl */ `
// ---------------------------------------------------------------- hashes/noise
float mg_hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 mg_hash22(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
float mg_hash13(vec3 p3) { p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
vec3 mg_srgb(vec3 c) { return pow(c, vec3(2.2)); }

float mg_vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(mg_hash12(i), mg_hash12(i + vec2(1.0, 0.0)), u.x),
             mix(mg_hash12(i + vec2(0.0, 1.0)), mg_hash12(i + vec2(1.0, 1.0)), u.x), u.y);
}
float mg_vnoise3(vec3 p) {
  vec3 i = floor(p), f = fract(p); vec3 u = f * f * (3.0 - 2.0 * f);
  float a = mix(mg_hash13(i), mg_hash13(i + vec3(1, 0, 0)), u.x);
  float b = mix(mg_hash13(i + vec3(0, 1, 0)), mg_hash13(i + vec3(1, 1, 0)), u.x);
  float c = mix(mg_hash13(i + vec3(0, 0, 1)), mg_hash13(i + vec3(1, 0, 1)), u.x);
  float d = mix(mg_hash13(i + vec3(0, 1, 1)), mg_hash13(i + vec3(1, 1, 1)), u.x);
  return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}
// Band-limit weight for a feature of 'cyc' cycles per pixel: 1 below ~0.3 cyc/px,
// fading to 0 by Nyquist, so sub-pixel detail averages out instead of aliasing
// differently in the two eyes (which would corrupt stereo matching).
float mg_bl(float cyc) { return 1.0 - smoothstep(0.3, 0.55, cyc); }
// Band-limited fbm; fp = footprint in metres/pixel; returns ~[0,1], mean 0.5.
float mg_fbm(vec2 p, float freq, float fp, int oct) {
  float s = 0.0, a = 0.5, n = 0.0; vec2 q = p * freq; float c = fp * freq;
  for (int i = 0; i < 6; i++) {
    if (i >= oct) break;
    s += a * mix(0.5, mg_vnoise(q), mg_bl(c)); n += a;
    q = q * 2.03 + vec2(1.7, 9.2); c *= 2.03; a *= 0.5;
  }
  return s / n;
}
float mg_fbm3(vec3 p, int oct) {
  float s = 0.0, a = 0.5, n = 0.0;
  for (int i = 0; i < 5; i++) { if (i >= oct) break; s += a * mg_vnoise3(p); n += a; p = p * 2.07 + 3.1; a *= 0.5; }
  return s / n;
}
// Cellular noise: x = distance to nearest feature point (cell units),
// y,z = two random numbers attached to that cell (stone identity).
vec3 mg_voronoi(vec2 p) {
  vec2 i = floor(p), f = fract(p); float d1 = 8.0; vec2 id = vec2(0.0);
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec2 g = vec2(float(x), float(y)); vec2 r = g + mg_hash22(i + g) - f; float d = dot(r, r);
    if (d < d1) { d1 = d; id = i + g; }
  }
  return vec3(sqrt(d1), mg_hash12(id), mg_hash12(id + 7.31));
}
// Screen-space derivative bump (Mikkelsen 2010), view-space inputs.
vec3 mg_perturb(vec3 surf_pos, vec3 surf_norm, float h, float amp) {
  vec2 dHdxy = vec2(dFdx(h), dFdy(h)) * amp;
  vec3 vSigmaX = dFdx(surf_pos), vSigmaY = dFdy(surf_pos);
  vec3 R1 = cross(vSigmaY, surf_norm), R2 = cross(surf_norm, vSigmaX);
  float fDet = dot(vSigmaX, R1);
  vec3 vGrad = sign(fDet) * (dHdxy.x * R1 + dHdxy.y * R2);
  return normalize(abs(fDet) * surf_norm - vGrad);
}
`;

// Terrain lookups: material weight textures and normal/cavity texture, all
// sampled at world (x, y). The warp perturbs material boundaries so the 5 cm
// material grid never shows as blocks; the semantic GT pass uses the SAME warp so
// labels match the rendered appearance.
export const TERRAIN_COMMON_GLSL = /* glsl */ `
uniform sampler2D uMatA;      // rgba = grass, dirt, gravel_trail, rocky_ground (one-hot, linear filtered)
uniform sampler2D uMatB;      // r = mud, g = water
uniform sampler2D uNrm;       // rgb = world normal*0.5+0.5, a = cavity (0.5 = flush)
uniform vec2 uGridOrigin;     // world xy of cell (0,0) centre (m)
uniform float uGridRes;       // m per cell
uniform vec2 uGridSize;       // (nx, ny)
vec2 mg_terrainUV(vec2 xy) { return ((xy - uGridOrigin) / uGridRes + 0.5) / uGridSize; }
vec2 mg_warp(vec2 xy) {
  return xy + 0.09 * (vec2(mg_vnoise(xy * 2.1), mg_vnoise(xy * 2.1 + 17.3)) - 0.5)
            + 0.03 * (vec2(mg_vnoise(xy * 9.0 + 3.1), mg_vnoise(xy * 9.0 + 8.7)) - 0.5);
}
bool mg_inGrid(vec2 xy) { vec2 uv = mg_terrainUV(xy); return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0))); }
void mg_weights(vec2 xy, out float w[6]) {
  vec2 q = mg_warp(xy);
  if (!mg_inGrid(q)) {
    // Apron outside the scenario grid: grass with dry-soil patches.
    float d = smoothstep(0.55, 0.75, mg_vnoise(xy * 0.15));
    w[0] = 1.0 - d; w[1] = d; w[2] = 0.0; w[3] = 0.0; w[4] = 0.0; w[5] = 0.0; return;
  }
  vec2 uv = mg_terrainUV(q);
  vec4 a = texture(uMatA, uv); vec4 b = texture(uMatB, uv);
  w[0] = a.r; w[1] = a.g; w[2] = a.b; w[3] = a.a; w[4] = b.r; w[5] = b.g;
}
`;

// Visual terrain surface model. Called from the patched MeshStandardMaterial.
export const TERRAIN_SURFACE_GLSL = /* glsl */ `
uniform float uTime;
struct MgSurf { vec3 albedo; float rough; float bump; float bumpAmp; vec3 nW; float wet; };

MgSurf mg_terrainSurface(vec3 wp, float fp, vec3 geoNW) {
  vec2 p = wp.xy;
  float w[6]; mg_weights(p, w);
  // Contrast-sharpen blends with a noisy height-blend so transitions look organic.
  float hb = mg_vnoise(p * 6.0);
  float ws = 0.0;
  for (int k = 0; k < 6; k++) { float x = clamp(w[k] * (0.75 + 0.5 * hb), 0.0, 1.0); w[k] = x * x * x; ws += w[k]; }
  for (int k = 0; k < 6; k++) w[k] /= max(ws, 1e-4);

  vec4 nc = texture(uNrm, mg_terrainUV(p));
  bool inGrid = mg_inGrid(p);
  vec3 nW = inGrid ? normalize(nc.rgb * 2.0 - 1.0) : normalize(geoNW);
  float cav = inGrid ? (nc.a - 0.5) / 1.5 : 0.0; // metres relative to 1 m-radius local mean
  float slope = 1.0 - nW.z;                     // 0 flat .. 1 vertical

  // Shared noise features (band-limited by the pixel footprint fp, m/px).
  float macro = mg_fbm(p, 0.12, fp, 3);                      // 8 m colour patches
  float mid = mg_fbm(p, 1.1, fp, 3);                         // ~1 m
  float fine = mix(0.5, mg_vnoise(p * 23.0), mg_bl(fp * 23.0));   // ~4 cm
  float fine2 = mix(0.5, mg_vnoise(p * 57.0 + 5.0), mg_bl(fp * 57.0)); // ~2 cm
  vec3 vor = mg_voronoi(p * 9.0);                            // ~11 cm stones
  float bvor = mg_bl(fp * 9.0);
  vec3 vor2 = mg_voronoi(p * 24.0 + 3.7);                    // ~4 cm gravel
  float bvor2 = mg_bl(fp * 24.0);

  vec3 alb = vec3(0.0); float rough = 0.0; float bump = 0.0; float wet = 0.0;

  // grass: olive/green with dry patches, dense blade-scale speckle
  if (w[0] > 0.004) {
    vec3 g = mix(mg_srgb(vec3(0.30, 0.40, 0.13)), mg_srgb(vec3(0.50, 0.49, 0.26)), smoothstep(0.42, 0.72, macro));
    g = mix(g, mg_srgb(vec3(0.20, 0.30, 0.10)), smoothstep(0.55, 0.3, mid) * 0.6);
    float blades = 0.45 + 1.1 * (0.6 * fine2 + 0.4 * fine);
    g *= blades;
    alb += w[0] * g; rough += w[0] * 0.92; bump += w[0] * (fine2 * 0.7 + fine * 0.3);
  }
  // dirt: brown soil with sparse small stones and grain
  if (w[1] > 0.004) {
    vec3 d = mix(mg_srgb(vec3(0.42, 0.32, 0.22)), mg_srgb(vec3(0.31, 0.23, 0.16)), mid);
    d *= 0.78 + 0.44 * (0.6 * fine2 + 0.4 * fine);
    float peb = (1.0 - smoothstep(0.12, 0.2, vor.x)) * step(0.72, vor.y) * bvor;
    d = mix(d, mg_srgb(vec3(0.45, 0.40, 0.34)) * (0.75 + 0.4 * vor.z), peb * 0.8);
    alb += w[1] * d; rough += w[1] * 0.95; bump += w[1] * (peb * 0.8 + fine2 * 0.3);
  }
  // gravel trail: compacted mixed grey/tan stones in a fines matrix
  if (w[2] > 0.004) {
    float stone = mix(0.5, 1.0 - smoothstep(0.3, 0.6, vor2.x), bvor2);
    vec3 tint = mix(mg_srgb(vec3(0.47, 0.44, 0.40)), mg_srgb(vec3(0.50, 0.42, 0.33)), vor2.z);
    vec3 s = tint * mix(1.0, 0.7 + 0.55 * vor2.y, bvor2);
    s = mix(mg_srgb(vec3(0.33, 0.29, 0.24)) * (0.8 + 0.4 * fine2), s, 0.3 + 0.55 * stone);
    s *= 0.82 + 0.3 * macro + 0.12 * (mid - 0.5);
    float big = (1.0 - smoothstep(0.16, 0.26, vor.x)) * step(0.78, vor.y) * bvor;
    s = mix(s, mg_srgb(vec3(0.44, 0.42, 0.40)) * (0.75 + 0.4 * vor.z), big * 0.9);
    alb += w[2] * s; rough += w[2] * 0.85; bump += w[2] * (stone * 0.8 + big);
  }
  // rocky ground: weathered grey rock, ridged cracks, lichen
  if (w[3] > 0.004) {
    float crack = 1.0 - smoothstep(0.0, 0.05, abs(mg_fbm(p, 1.7, fp, 3) - 0.5));
    vec3 r = mix(mg_srgb(vec3(0.47, 0.45, 0.42)), mg_srgb(vec3(0.38, 0.35, 0.31)), mid);
    r *= 0.75 + 0.5 * fine;
    r = mix(r, mg_srgb(vec3(0.20, 0.18, 0.16)), crack * 0.7);
    r = mix(r, mg_srgb(vec3(0.55, 0.56, 0.38)), smoothstep(0.62, 0.7, mg_fbm(p, 3.0, fp, 2)) * 0.6);
    alb += w[3] * r; rough += w[3] * 0.8; bump += w[3] * (0.6 * fine + 0.5 * (1.0 - crack));
  }
  // mud: dark, wet, low roughness, tyre-track-like smears
  if (w[4] > 0.004) {
    vec3 m = mix(mg_srgb(vec3(0.24, 0.18, 0.12)), mg_srgb(vec3(0.17, 0.13, 0.09)), mid);
    m *= 0.8 + 0.4 * fine;
    alb += w[4] * m; rough += w[4] * mix(0.25, 0.55, fine); bump += w[4] * fine * 0.4; wet += w[4];
  }
  // water: dark, near-mirror; ripple normal is added in the material patch
  if (w[5] > 0.004) {
    alb += w[5] * mg_srgb(vec3(0.05, 0.07, 0.07)); rough += w[5] * 0.05; wet += w[5];
  }

  // Exposed soil on steep faces (ditch walls, banks): dark, horizontally layered.
  float steep = smoothstep(0.22, 0.45, slope) * (1.0 - w[5]);
  if (steep > 0.0) {
    float layers = mg_vnoise(vec2(p.x * 0.7 + p.y * 0.7, wp.z * 22.0));
    vec3 soil = mix(mg_srgb(vec3(0.27, 0.20, 0.14)), mg_srgb(vec3(0.36, 0.28, 0.19)), layers);
    soil *= 0.75 + 0.5 * mix(0.5, mg_vnoise3(wp * 30.0), mg_bl(fp * 30.0));
    float roots = (1.0 - smoothstep(0.0, 0.04, abs(mg_vnoise(p * 4.0 + wp.z * 3.0) - 0.5))) * 0.5;
    soil = mix(soil, mg_srgb(vec3(0.12, 0.09, 0.07)), roots);
    alb = mix(alb, soil, steep); rough = mix(rough, 0.9, steep); bump = mix(bump, layers * 0.6 + fine2 * 0.4, steep);
  }
  // Depressions (ditch floors, hollows) hold moisture: darker, damper soil.
  float damp = smoothstep(-0.06, -0.22, cav) * (1.0 - w[5]);
  alb = mix(alb, alb * vec3(0.55, 0.52, 0.5), damp);
  rough = mix(rough, 0.65, damp * 0.5); wet = max(wet, damp * 0.4);
  // Cavity ambient occlusion (cheap stand-in for sky visibility).
  alb *= mix(0.6, 1.0, smoothstep(-0.3, 0.0, cav));

  MgSurf s; s.albedo = alb; s.rough = clamp(rough, 0.04, 1.0); s.bump = bump;
  s.bumpAmp = mix(0.012, 0.004, w[5] + w[4]); s.nW = nW; s.wet = wet;
  return s;
}
`;

// Post pass: exposure, optical vignette + blur, glare, ACES, sRGB, sensor noise.
// Output layout (mode 0): one RGBA8 texel per pixel = (left R, G, B, right GRAY)
// with rows written top-first so a single readPixels returns the images in
// OpenCV row order (no CPU flip).
export const POST_VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

export const POST_FRAG = /* glsl */ `
precision highp float; precision highp int;
uniform sampler2D tL;
uniform sampler2D tR;
uniform int uMode;            // 0 = stereo pack (L rgb + R gray), 1 = single colour image
uniform vec2 uSize;           // output size (px)
uniform vec2 uPP;             // principal point (px, OpenCV)
uniform float uGainL;         // linear gain (exposure * irradiance) applied before tone mapping
uniform float uGainR;
uniform float uNoiseDN;       // gaussian sensor noise sigma, 8-bit DN
uniform float uVignette;      // relative fall-off at the image corner
uniform float uBlur;          // 0 = pinhole-sharp, 1 = ~0.5 px sigma lens blur
uniform vec3 uSun;            // (u, v, weight): sun position in px (OpenCV) and flare weight
uniform float uVeil;          // additive veiling glare (linear, post-exposure)
uniform vec3 uHaze;           // additive haze colour (linear) for dust veil
uniform float uHazeAmt;
uniform uint uSeed;

uint mg_pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float mg_u01(uint h) { return (float(h >> 8u) + 0.5) * (1.0 / 16777216.0); }
vec2 mg_gauss2(uint h) {
  float u1 = mg_u01(h), u2 = mg_u01(mg_pcg(h ^ 0x9E3779B9u));
  float r = sqrt(-2.0 * log(u1)); return r * vec2(cos(6.2831853 * u2), sin(6.2831853 * u2));
}
vec3 RRTAndODTFit(vec3 v) { vec3 a = v * (v + 0.0245786) - 0.000090537; vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081; return a / b; }
vec3 mg_aces(vec3 color) {
  const mat3 IM = mat3(vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777));
  const mat3 OM = mat3(vec3(1.60475, -0.10208, -0.00327), vec3(-0.53108, 1.10813, -0.07276), vec3(-0.07367, -0.00605, 1.07602));
  color *= 1.0 / 0.6; color = IM * color; color = RRTAndODTFit(color); color = OM * color;
  return clamp(color, 0.0, 1.0);
}
vec3 mg_oetf(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c)); }

vec3 mg_fetch(sampler2D t, ivec2 p, ivec2 sz) { return texelFetch(t, clamp(p, ivec2(0), sz - 1), 0).rgb; }
vec3 mg_lens(sampler2D t, ivec2 p, ivec2 sz) {
  vec3 c = mg_fetch(t, p, sz);
  if (uBlur <= 0.0) return c;
  vec3 e = mg_fetch(t, p + ivec2(1, 0), sz) + mg_fetch(t, p - ivec2(1, 0), sz) + mg_fetch(t, p + ivec2(0, 1), sz) + mg_fetch(t, p - ivec2(0, 1), sz);
  vec3 d = mg_fetch(t, p + ivec2(1, 1), sz) + mg_fetch(t, p - ivec2(1, 1), sz) + mg_fetch(t, p + ivec2(1, -1), sz) + mg_fetch(t, p + ivec2(-1, 1), sz);
  vec3 b = 0.5 * c + 0.1 * e + 0.025 * d;
  return mix(c, b, uBlur);
}
// Lens flare: glow + 6-point star around the sun, three ghosts mirrored through the centre.
vec3 mg_flare(vec2 px) {
  if (uSun.z <= 0.0) return vec3(0.0);
  float s = uSize.x;
  vec2 d = (px - uSun.xy) / s; float r = length(d);
  float glow = 1.6 * exp(-r * 30.0) + 0.25 * exp(-r * 7.0);
  float ang = atan(d.y, d.x);
  float star = pow(abs(cos(ang * 3.0)), 80.0) * exp(-r * 14.0) * 0.6;
  vec3 col = vec3(1.0, 0.93, 0.8) * (glow + star);
  vec2 c = uPP; vec2 axis = uSun.xy - c;
  vec3 gcol[3]; gcol[0] = vec3(0.5, 0.8, 0.4); gcol[1] = vec3(0.4, 0.5, 1.0); gcol[2] = vec3(1.0, 0.6, 0.3);
  float gpos[3]; gpos[0] = -0.45; gpos[1] = -0.9; gpos[2] = 0.35;
  float grad[3]; grad[0] = 0.035; grad[1] = 0.07; grad[2] = 0.02;
  for (int k = 0; k < 3; k++) {
    vec2 g = c + axis * gpos[k];
    float gr = length(px - g) / s;
    col += gcol[k] * 0.05 * (1.0 - smoothstep(grad[k] * 0.6, grad[k], gr));
  }
  return col * uSun.z;
}

void main() {
  ivec2 sz = ivec2(uSize);
  ivec2 op = ivec2(gl_FragCoord.xy);
  // Output row i (first row returned by readPixels) holds OpenCV image row v = i;
  // in the GL source texture that row is H-1-i.
  ivec2 sp = ivec2(op.x, sz.y - 1 - op.y);
  vec2 px = vec2(float(op.x), float(op.y));      // OpenCV pixel coordinates (u, v)

  vec2 dv = (px - uPP) / (0.5 * length(uSize));
  float vig = 1.0 - uVignette * dot(dv, dv);
  vec3 flare = mg_flare(px) + vec3(uVeil) + uHaze * uHazeAmt;

  uint base = mg_pcg(uint(op.x) + uint(op.y) * 8192u + mg_pcg(uSeed));
  vec3 cl = mg_lens(tL, sp, sz) * uGainL * vig + flare;
  vec3 sl = mg_oetf(mg_aces(cl)) * 255.0;
  vec2 n1 = mg_gauss2(base), n2 = mg_gauss2(mg_pcg(base + 0x68E31DA4u));
  if (uMode == 1) {
    vec3 o = sl + uNoiseDN * vec3(n1, n2.x);
    gl_FragColor = vec4(clamp(o, 0.0, 255.0) / 255.0, 1.0);
    return;
  }
  vec3 cr = mg_lens(tR, sp, sz) * uGainR * vig + flare;
  vec3 sr = mg_oetf(mg_aces(cr)) * 255.0;
  float gray = dot(sr, vec3(0.299, 0.587, 0.114));   // OpenCV RGB2GRAY weights
  vec3 ol = clamp(sl + uNoiseDN * vec3(n1, n2.x), 0.0, 255.0);
  float og = clamp(gray + uNoiseDN * n2.y, 0.0, 255.0);
  if (uMode == 2) {
    // Camera-style YUV (full-range BT.601 / JFIF) of the left image; chroma is
    // 2x2-averaged on the CPU into 4:2:0. Right stays luma (= same weights).
    float Y = dot(ol, vec3(0.299, 0.587, 0.114));
    float U = 128.0 - 0.168736 * ol.r - 0.331264 * ol.g + 0.5 * ol.b;
    float V = 128.0 + 0.5 * ol.r - 0.418688 * ol.g - 0.081312 * ol.b;
    gl_FragColor = vec4(clamp(vec3(Y, U, V), 0.0, 255.0) / 255.0, og / 255.0);
    return;
  }
  gl_FragColor = vec4(ol / 255.0, og / 255.0);
}
`;

// Auto-exposure meter: 16x10 grid of mean HDR luminance (each output texel averages
// an 8x8 sub-grid of its image tile), clamped per sample so the sun disc cannot
// dominate. Read back after the frame and used for the NEXT frame, like a real AE.
export const METER_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D tSrc;
uniform vec2 uSrcSize;
uniform float uClampLum;
void main() {
  vec2 tile = uSrcSize / vec2(16.0, 10.0);
  vec2 o = floor(gl_FragCoord.xy) * tile;
  float s = 0.0;
  for (int j = 0; j < 8; j++) for (int i = 0; i < 8; i++) {
    vec2 p = o + (vec2(float(i), float(j)) + 0.5) * tile / 8.0;
    vec3 c = texelFetch(tSrc, ivec2(p), 0).rgb;
    s += min(dot(c, vec3(0.2126, 0.7152, 0.0722)), uClampLum);
  }
  gl_FragColor = vec4(s / 64.0, 0.0, 0.0, 1.0);
}
`;

// Ground-truth passes -------------------------------------------------------
// Depth: view-space z (metres along the optical axis) written as the 4 raw
// bytes of a little-endian float32 into an RGBA8 target; cleared to 0 = sky.
export const DEPTH_VERT = /* glsl */ `
varying float vDepth;
void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vDepth = -mv.z; gl_Position = projectionMatrix * mv; }
`;
export const DEPTH_FRAG = /* glsl */ `
precision highp float; precision highp int;
varying float vDepth;
void main() {
  uint b = floatBitsToUint(vDepth);
  gl_FragColor = vec4(float(b & 255u), float((b >> 8u) & 255u), float((b >> 16u) & 255u), float(b >> 24u)) / 255.0;
}
`;
export const SEM_VERT = /* glsl */ `
varying vec3 vMgWorld;
void main() { vec4 w = modelMatrix * vec4(position, 1.0); vMgWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }
`;
export const SEM_OBJ_FRAG = /* glsl */ `
uniform float uId;
void main() { gl_FragColor = vec4(uId / 255.0, 0.0, 0.0, 1.0); }
`;
export function semTerrainFrag(matToSem) {
  // matToSem: array of 6 ints, material id -> 5-class semantic id.
  const lut = matToSem.map((v) => `${v.toFixed(1)}`).join(', ');
  return /* glsl */ `
${NOISE_GLSL}
${TERRAIN_COMMON_GLSL}
varying vec3 vMgWorld;
void main() {
  float w[6]; mg_weights(vMgWorld.xy, w);
  float lut[6] = float[6](${lut});
  int best = 0; float bw = -1.0;
  for (int k = 0; k < 6; k++) { if (w[k] > bw) { bw = w[k]; best = k; } }
  gl_FragColor = vec4(lut[best] / 255.0, 0.0, 0.0, 1.0);
}
`;
}
