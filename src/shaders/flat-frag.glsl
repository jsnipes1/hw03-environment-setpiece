#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

// Concept: Super Mario Galaxy scene, including...
  // Lumas (combined SDFs, twist operations, subsurface scattering, animated)
  // Flying/floating star bits (animated; use sawtooth function to regenerate the same ones)
    // Needs some sort of transmissive shader to achieve glassy look
    // Make from two intersected tetrahedra
  // Small, distinct planets (use low-octave noise to distort normals)
  // Glowing asteroids? (subsurface scattering)
  // Galactic background (recursive noise)

// Requirements:
  // Animated environment elements (star bits)
  // 3 uses of noise (planet terrain, asteroid glow, background)
  // Remap [0, 1] to a set of colors (done)
  // Toolbox functions (sawtooth wave for star bits)
  // Approximated environmental lighting using 3-4 dir. lights (done) + ambient (TODO)
  // Soft shadows (done)

  // Ray-based specular reflection (star bits, planet features, ...)
  // SDF blending (Lumas)

  // Technique mastery: Make deliberate choices in scene composition!

/////// RAYS ///////
// Properties of each ray cast into the scene
struct Ray {
  vec3 origin;
  vec3 direction;
};

// March t units along ray r
vec3 pointOnRay(Ray r, float t) {
  return r.origin + t * r.direction;
}
////////////////////

///////// NOISE ///////////
// From Mariano's github
float hash3D(vec3 x) {
	float i = dot(x, vec3(123.4031, 46.5244876, 91.106168));
	return fract(sin(i * 7.13) * 268573.103291);
}

// 3D noise
float noise(vec3 p) {
  vec3 bCorner = floor(p);
  vec3 inCell = fract(p);

  float bLL = hash3D(bCorner);
  float bUL = hash3D(bCorner + vec3(0.0, 0.0, 1.0));
  float bLR = hash3D(bCorner + vec3(0.0, 1.0, 0.0));
  float bUR = hash3D(bCorner + vec3(0.0, 1.0, 1.0));
  float b0 = mix(bLL, bUL, inCell.z);
  float b1 = mix(bLR, bUR, inCell.z);
  float b = mix(b0, b1, inCell.y);

  vec3 fCorner = bCorner + vec3(1.0, 0.0, 0.0);
  float fLL = hash3D(fCorner);
  float fUL = hash3D(fCorner + vec3(0.0, 0.0, 1.0));
  float fLR = hash3D(fCorner + vec3(0.0, 1.0, 0.0));
  float fUR = hash3D(fCorner + vec3(0.0, 1.0, 1.0));
  float f0 = mix(fLL, fUL, inCell.z);
  float f1 = mix(fLR, fUR, inCell.z);
  float f = mix(f0, f1, inCell.y);

  return mix(b, f, inCell.x);
}

// 3-octave FBM
float fbm(vec3 q) {
  float acc = 0.0;
  float freqScale = 2.0;
  float invScale = 1.0 / freqScale;
  float freq = 1.0;
  float amp = 1.0;

  for (int i = 0; i < 3; ++i) {
    freq *= freqScale;
    amp *= invScale;
    acc += noise(q * freq) * amp;
  }
  return acc;
}

// Recursive FBM pattern
float pattern(in vec3 p) {
  vec3 q = vec3(fbm(p + vec3(0.0)),
                fbm(p + vec3(5.2, 1.3, 2.8)),
                fbm(p + vec3(1.2, 3.4, 1.2)));

  return fbm(p + 4.0 * q);
}

// Cosine color palette from IQ
vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}
///////////////////////////

//////// Toolbox Functions from IQ ////////
// Used to display multiple SDFs in one scene
float opUnion(float d1, float d2) {  
  return min(d1, d2);
}

float opSmoothUnion(float d1, float d2, float k) {
  float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return mix(d2, d1, h) - k * h * (1.0 - h);
}

float opSubtraction(float d1, float d2) {
  return max(-d1,d2);
}

float opIntersection(float d1, float d2) {
  return max(d1,d2);
}

// The below functions are modified so they can be applied to individual objects
vec3 opTwist(vec3 p) {
  const float k = 1.0; // Number of rotations
  float c = cos(k * p.y);
  float s = sin(k * p.y);
  mat2 m = mat2(c, -s, s, c); // Rotation matrix about y axis
  return vec3(m * p.xz, p.y);
}

float opDisplacement(float sdf, vec3 p) {
  float dr = fbm(p); // Displace the point by some amount
  return sdf + dr;
}

vec3 opRep(vec3 p, vec3 c) {
  return mod(p, c) - 0.5 * c;
}
///////////////////////////////////////////

//////// SDFs ////////
// SDF for a sphere centered at c
  // Source: http://www.michaelwalczyk.com/blog/2017/5/25/ray-marching
float sphereSDF(vec3 p, vec3 c, float r) {
  return length(p - c) - r;
}

// SDF for a cube centered at the origin
float cubeSDF(vec3 p, vec3 b) {
  vec3 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float sceneSDF(vec3 p) {
  // All objects in the scene
  float scene[2];
  scene[0] = sphereSDF(p, vec3(2.6, 0.5, 0.0), 1.0);
  scene[1] = cubeSDF(opTwist(p), vec3(1.0));

  float minDist = 100000000.0;
  for (int i = 0; i < scene.length(); ++i) {
    if (scene[i] < minDist) {
      minDist = scene[i];
    }
  }

  return minDist;
}
//////////////////////

// Compute a surface normal via gradients
  // Source: http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
vec3 surfaceNormal(vec3 p) {
  float e = 0.001;
  vec3 n;
  n.x = sceneSDF(vec3(p.x + e, p.y, p.z)) - sceneSDF(vec3(p.x - e, p.y, p.z));
  n.y = sceneSDF(vec3(p.x, p.y + e, p.z)) - sceneSDF(vec3(p.x, p.y - e, p.z));
  n.z = sceneSDF(vec3(p.x, p.y, p.z + e)) - sceneSDF(vec3(p.x, p.y, p.z - e));
  return normalize(n);
}

////////// SHADOWING //////////
// From IQ; higher k => harder shadows
float softShadow(Ray r, float mint, float maxt, float k) {
  float res = 1.0;

  for(float t = mint; t < maxt; ) {
    float h = sceneSDF(pointOnRay(r, t));
    if (h < 0.001) {
        return 0.0;
    }
    res = min(res, k * h / t);
    t += h;
  }

  return res;
}

// Five-tap AO
float ambientOcclusion(vec3 p, vec3 n, float k) {
  float sum = 0.0;
  float delta = 0.05;
  for (int i = 1; i <= 5; ++i) {
    sum += (float(i) * delta - sceneSDF(p + n * float(i) * delta)) / pow(2.0, float(i));
  }

  return 1.0 - k * sum;
}
///////////////////////////////

//// REFLECTION MODELS ////
// From Emily's Shadertoy example
vec4 lambert(vec4 lights[3], vec3 lightColors[3], vec3 p, vec3 baseColor, Ray r) {
  vec3 sumColor = vec3(0.0);
  vec3 nHat = surfaceNormal(p);

  for (int i = 0; i < 3; ++i) {
    vec3 lHat = normalize(lights[i].xyz - p);
    Ray lightRay = Ray(p, lHat);
    vec3 lamb = baseColor * clamp(dot(nHat, lHat), 0.0, 1.0) * lights[i].w * lightColors[i];
    sumColor += lamb * vec3(softShadow(lightRay, 0.1, 10.0, 4.0));
  }

  // Return average color
  sumColor /= 3.0;
  return vec4(sumColor, 1.0);
}

// TODO: For star bits; maybe use Fresnel?
vec4 glass(vec4 lights[3], vec3 lightColors[3], vec3 p, vec3 baseColor, Ray r) {
  return vec4(1.0);
}

// TODO: For Lumas
vec4 subsurfaceScatter(vec4 lights[3], vec3 lightColors[3], vec3 p, vec3 baseColor, Ray r) {
  return vec4(1.0);
}
///////////////////////////

// Galactic background
vec4 galaxy(Ray r, float time) {
  float t = pattern(r.direction * time);
  vec3 a = vec3(-0.262, -0.582, -0.102);
  vec3 b = vec3(0.5);
  vec3 c = vec3(1.0, 1.0, 0.428);
  vec3 d = vec3(0.428, 0.333, 0.667);
  return vec4(palette(t, a, b, c, d), 1.0);
}

vec4 raymarch(Ray r, const float start, const int maxIterations, float t) {
  float depth = start;

  for (int i = 0; i < maxIterations; ++i) {
    vec3 p = pointOnRay(r, depth);

    // Find closest SDF
    float toShape = sceneSDF(p);

    // Set up three-point lighting (w component is the intensity term)
    vec4 lights[3];
    vec3 lightColors[3];

    // Taken from Emily's Shadertoy example; to be tweaked
    lights[0] = vec4(6.0, 3.0, 5.0, 2.0); // key light
    lights[1] = vec4(-6.0, 3.0, 5.0, 1.5); // fill light
	  lights[2] = vec4(6.0, 5.0, -1.75, 4.0); // back light
    
    lightColors[0] = vec3(0.9, 0.5, 0.9);
    lightColors[1] = vec3(0.4, 0.7, 1.0);
    lightColors[2] = vec3(1.0, 1.0, 0.2);

    // We're inside the shape, so the ray hit it; return color of shape + shading
    if (abs(toShape) <= 0.01) {
      // Lambertian shading
      vec3 base = vec3(1.0);
      return lambert(lights, lightColors, p, base, r) * vec4(vec3(ambientOcclusion(p, surfaceNormal(p), 1.0)), 1.0);
    }

    // We've yet to hit anything; continue marching
    depth += toShape;
  }

  // We miss all objects; return the background
  return galaxy(r, t * 0.00003);
}

void main() {
  vec3 eyeToRef = u_Ref - u_Eye;
  float len = length(eyeToRef);
  vec3 uLook = normalize(eyeToRef);
  vec3 uRight = cross(uLook, u_Up);

  float vFOV = 90.0;
  float tanAlpha = tan(vFOV * 0.5);
  float aspect = u_Dimensions.x / u_Dimensions.y;
  vec3 V = u_Up * len * tanAlpha;
  vec3 H = uRight * len * aspect * tanAlpha;
  
  vec3 p = u_Ref + fs_Pos.x * H + fs_Pos.y * V;
  vec3 dir = normalize(p - u_Eye);

  Ray r = Ray(u_Eye, dir);

  out_Col = raymarch(r, 0.001, 256, u_Time);
}
