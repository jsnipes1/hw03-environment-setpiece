#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

// Properties of each ray cast into the scene
struct Ray {
  vec3 origin;
  vec3 direction;
};

// March t units along ray r
vec3 pointOnRay(Ray r, float t) {
  return r.origin + t * r.direction;
}

// SDF for a sphere centered at c
  // Source: http://www.michaelwalczyk.com/blog/2017/5/25/ray-marching
float sphereSDF(vec3 p, vec3 c, float r) {
  return length(p - c) - r;
}

// Compute a surface normal via gradients
  // Source: http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
vec3 surfaceNormal(vec3 p, vec3 c, float r) {
  float e = 0.001;
  vec3 n;
  n.x = sphereSDF(vec3(p.x + e, p.y, p.z), c, r) - sphereSDF(vec3(p.x - e, p.y, p.z), c, r);
  n.y = sphereSDF(vec3(p.x, p.y + e, p.z), c, r) - sphereSDF(vec3(p.x, p.y - e, p.z), c, r);
  n.z = sphereSDF(vec3(p.x, p.y, p.z + e), c, r) - sphereSDF(vec3(p.x, p.y, p.z - e), c, r);
  return normalize(n);
}

//////// From IQ ////////
// float pattern(in vec3 p) {
//   vec3 q = vec3(fbm(p + vec3(0.0)),
//                 fbm(p + vec3(5.2, 1.3, 2.8)),
//                 fbm(p + vec3(1.2, 3.4, 1.2)));

//   return fbm(p + 4.0 * q);
// }

vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

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

// Slightly adapted to avoid recomputation; higher k => harder shadows
float softShadow(Ray r, float mint, float maxt, float k) {
  float res = 1.0;
  float ph = 10000000.0;

  for(float t = mint; t < maxt; ) {
      float h = sphereSDF(pointOnRay(r, t), vec3(0.0), 1.0);
      if (h < 0.0001) {
          return 0.0;
      }
      float y = h * h / (2.0 * ph);
      float d = sqrt(h * h - y * y);
      res = min(res, k * d / max(0.0, t - y));
      ph = h;
      t += h;
  }

  return res;
}
////////////////////////

//// REFLECTION MODELS ////
// From Emily's Shadertoy code
vec4 lambert(vec4 lights[3], vec3 lightColors[3], vec3 p, vec3 baseColor, Ray r, int shapeID) {
  vec3 sumColor = vec3(0.0);
  vec3 nHat; // Generalize surfaceNormal() and compute here

  // Sphere
  if (shapeID == 0) {
    nHat = surfaceNormal(p, vec3(0.0), 1.0);
  }
  // Add else ifs and so on as you add shapes

  for (int i = 0; i < 3; ++i) {
    vec3 lHat = normalize(lights[i].xyz - p);
    vec3 lamb = baseColor * clamp(dot(nHat, lHat), 0.0, 1.0) * lights[i].w * lightColors[i];
    sumColor += lamb * softShadow(r, 0.1, 1.0, 2.0);
  }

  // Return average color
  sumColor /= 3.0;
  return vec4(sumColor, 1.0);
}
///////////////////////////

vec4 raymarch(Ray r, const float start, const int maxIterations, float t) {
  float depth = start;

  for (int i = 0; i < maxIterations; ++i) {
    vec3 p = pointOnRay(r, depth);

    // Find closest SDF (when there are multiple in the scene)

    // Euclidean distance to the shape
    float toShape = sphereSDF(p, vec3(0.0), 1.0); // For initial testing

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
      // Lambertian shading (applied to all shapes)
      vec3 base = vec3(1.0);
      return lambert(lights, lightColors, p, base, r, 0);
    }

    // We've yet to hit anything; continue marching
    depth += toShape;
  }

  // We miss all objects; return a black background for testing
  return vec4(0.0, 0.0, 0.0, 1.0);
}

void main() {
  // Basic ray casting (from 560 slides)
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
