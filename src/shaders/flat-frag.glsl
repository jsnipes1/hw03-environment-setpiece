#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

// Concept: Super Mario Galaxy scene, including...
  // Lumas (done)
  // Flying/floating star bits (animated; use sawtooth function to regenerate the same ones)
    // Make from octahedron intersected with cube
  // Small, distinct planets (use low-octave noise to distort normals)
  // Glowing asteroids? (subsurface scattering)
  // Galactic background (done)

// Requirements:
  // Animated environment elements (star bits)
  // 3 uses of noise (planet terrain, luma glow, background)
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

// Multi-octave FBM
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

float opSmoothIntersection(float d1, float d2, float k) {
  float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return mix(d2, d1, h) + k * h * (1.0 - h);
}

// The below functions are modified so they can be applied to individual objects
vec3 opTwist(vec3 p, const float k) {
  float c = cos(k * p.y);
  float s = sin(k * p.y);
  mat3 m = mat3(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
  return m * p;
}

// Displace the point by some amount to create a bumpy surface
float opDisplacement(vec3 p, float sdf) {
  float dr = sin(p.x) * cos(2.0 * sin(p.y)) * cos(p.z) * 0.5;
  return opSmoothUnion(sdf, sdf + dr, 0.5);
}

vec3 opRep(vec3 p, vec3 c) {
  return mod(p, c) - 0.5 * c;
}

vec3 opTranslate(vec3 p, vec3 t) {
  return p - t;
}

// Clockwise rotations about each axis
vec3 opRotateX(vec3 p, float theta) {
  float c = cos(theta * 0.01745329251);
  float s = sin(theta * 0.01745329251);
  mat3 r = mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
  return r * p;
}

vec3 opRotateY(vec3 p, float theta) {
  float c = cos(theta * 0.01745329251);
  float s = sin(theta * 0.01745329251);
  mat3 r = mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
  return r * p;
}

vec3 opRotateZ(vec3 p, float theta) {
  float c = cos(theta * 0.01745329251);
  float s = sin(theta * 0.01745329251);
  mat3 r = mat3(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0);
  return r * p;
}

// Symmetry across X axis
vec3 opSymX(vec3 p) {
  return vec3(abs(p.x), p.y, p.z);
}
///////////////////////////////////////////

//////// SDFs ////////
struct SceneObject {
  int reflectionModel;
  float sdf;
  vec3 baseColor;
};

// SDF for a sphere centered at c
  // Source: http://www.michaelwalczyk.com/blog/2017/5/25/ray-marching
float sphereSDF(vec3 p, vec3 c, float r) {
  return length(p - c) - r;
}

// SDF for a rounded cone shape
float roundConeSDF(vec3 p, float r1, float r2, float h) {
  vec2 q = vec2(length(p.xz), p.y);
  float b = (r1 - r2) / h;
  float a = sqrt(1.0 - b * b);
  float k = dot(q, vec2(-b, a));

  if (k < 0.0) {
    return length(q) - r1;
  }
  if (k > a * h) {
    return length(q - vec2(0.0, h)) - r2;
  }

  return dot(q, vec2(a, b)) - r1;
}

float vertCapsuleSDF(vec3 p, float h, float r) {
  vec3 q = p;
  q.y -= clamp(q.y, 0.0, h);
  return length(q) - r;
}

// The eyes of the Luma need to be treated separately since they
// use a different reflection model and have a different base color
float lumaEyesSDF(vec3 p, vec3 c) {
  return vertCapsuleSDF(opTranslate(opSymX(p - c), vec3(0.3, 0.1, -0.85)), 0.25, 0.15);
}

// Make the luma's main body
float lumaSDF(vec3 p, vec3 c) {
  float body = sphereSDF(p, c, 1.0);
  float legs = roundConeSDF(opTranslate(opSymX(p - c), vec3(0.46, -1.15, 0.0)), 0.1, 0.26, 0.5);
  float bodyLegs = opSmoothUnion(body, legs, 0.15);

  float arms = roundConeSDF(opTranslate(opRotateZ(opSymX(p - c + vec3(0.0, -1.25, 0.0)), 60.0), vec3(1.5, 0.0, 0.0)), 0.3, 0.1, 0.6);
  float armsBodyLegs = opSmoothUnion(bodyLegs, arms, 0.08);

  vec3 twist1 = opTwist(opTranslate(p - c, vec3(0.0, 0.7, 0.0)), 0.8);
  float swirl1 = roundConeSDF(twist1, 0.4, 0.1, 0.75);
  float addTwist = opSmoothUnion(armsBodyLegs, swirl1, 0.3);

  vec3 twist2 = opTwist(opRotateZ(opTranslate(p - c, vec3(-0.49, 1.365, 0.0)), -92.0), 1.6);
  float swirl2 = roundConeSDF(twist2, 0.07, 0.05, 0.25);
  
  return opSmoothUnion(addTwist, swirl2, 0.075);
}

float octahedronSDF(vec3 p, vec3 c, float s) {
  vec3 q = abs(p - c);
  float m = q.x + q.y + q.z - s;
  vec3 r;
  if (3.0 * q.x < m) {
    r = q.xyz;
  }
  else if (3.0 * q.y < m) {
    r = q.yzx;
  }
  else if (3.0 * q.z < m) {
    r = q.zxy;
  }
  else {
    return m * 0.57735027;
  }

  float k = clamp(0.5 * (r.z - r.y + s), 0.0, s);
  return length(vec3(r.x, r.y - s + k, r.z - k));
}

float cubeSDF(vec3 p, vec3 c, vec3 b) {
  vec3 d = abs(p - c) - b;
  return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// Starbits are much simpler
float starbitSDF(vec3 p, vec3 c) {
  // Subtract off some value to round the edges for a more cartoony feel
  float oct = octahedronSDF(p, c, 2.0) - 0.3;
  float cube = cubeSDF(p, c, vec3(0.97)) - 0.3;
  return opSmoothUnion(oct, cube, 0.25);
}

// TODO
float waterPlanetSDF(vec3 p, vec3 c) {
  float mainPlanet = opDisplacement(p, sphereSDF(p, c, 3.0));
  return mainPlanet;
}

SceneObject sceneSDF(vec3 p) {
  // All objects in the scene
  SceneObject scene[1];

  // One luma requires two scene objects due to the different reflection model in the eyes
  // scene[0] = SceneObject(3, lumaSDF(p, vec3(2.0, 1.0, 0.0)), vec3(1.0));
  // scene[1] = SceneObject(1, lumaEyesSDF(p, vec3(2.0, 1.0, 0.0)), vec3(0.3));

  // scene[0] = SceneObject(2, starbitSDF(p, vec3(0.0)), vec3(0.0, 1.0, 0.3));
  scene[0] = SceneObject(1, waterPlanetSDF(p, vec3(-6.0, 1.0, -2.0)), vec3(0.619, 0.427, 0.101));

  float minDist = 100000000.0;
  int closest = 0;
  for (int i = 0; i < scene.length(); ++i) {
    if (scene[i].sdf < minDist) {
      minDist = scene[i].sdf;
      closest = i;
    }
  }
  return scene[closest];
}
//////////////////////

// Compute a surface normal via gradients
  // Source: http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
vec3 surfaceNormal(vec3 p) {
  float e = 0.001;
  vec3 n;
  n.x = sceneSDF(vec3(p.x + e, p.y, p.z)).sdf - sceneSDF(vec3(p.x - e, p.y, p.z)).sdf;
  n.y = sceneSDF(vec3(p.x, p.y + e, p.z)).sdf - sceneSDF(vec3(p.x, p.y - e, p.z)).sdf;
  n.z = sceneSDF(vec3(p.x, p.y, p.z + e)).sdf - sceneSDF(vec3(p.x, p.y, p.z - e)).sdf;
  return normalize(n);
}

////////// SHADOWING //////////
// From IQ; higher k => harder shadows
float softShadow(Ray r, float mint, float maxt, float k) {
  float res = 1.0;

  for(float t = mint; t < maxt; ) {
    float h = sceneSDF(pointOnRay(r, t)).sdf;
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
  for (int i = 0; i <= 5; ++i) {
    sum += (float(i) * delta - sceneSDF(p + n * float(i) * delta).sdf) / pow(2.0, float(i));
  }

  return 1.0 - k * sum;
}
///////////////////////////////

/////// BACKGROUND ////////
float starFBM(vec3 q) {
  float acc = 0.0;
  float amp = 1.0;
  float maxAmp = 0.0;

  for (int i = 0; i < 4; ++i) {
    maxAmp += amp;
    acc += noise(q) * amp;
    amp *= 0.5;
    q *= 2.0;
  }
  return 1.2 * acc / maxAmp;
}

vec4 galaxy(vec3 p) {
  // Based on Joe's galaxy/nebula shader
  float star1 = starFBM(p * 5.7);
  float star2 = starFBM(p + vec3(1.27, 6.298, 4.243));
  float star3 = starFBM(p + vec3(0.23, 0.45, 0.67) * 5.0 + 0.005 * sin(0.05 * u_Time * fbm(p)));
  float starTotal = star1 * star2 * star3 * 3.0;

  float falloff = 0.55;
  float noiseThreshold = 1.9;

  starTotal = clamp(starTotal - noiseThreshold + falloff, 0.0, 1.0);

  float weight = starTotal / (7.0 * falloff);
  return vec4(18.0 * weight * vec3(star1 * 0.6, star2 * 0.4, star3 * 0.4), 1.0);
}

// vec4 nebula(Ray r) {
//   float t = pattern(r.direction * u_Time * 0.0003);
//   vec3 a = vec3(-0.452, -0.082, -0.082);
//   vec3 b = vec3(0.5);
//   vec3 c = vec3(1.0, 0.878, 0.558);
//   vec3 d = vec3(-0.982, 0.348, 0.667);
//   return vec4(palette(t, a, b, c, d), 0.0) + galaxy(pointOnRay(r, 400.0));
// }
///////////////////////////

//// REFLECTION MODELS ////
// From Emily's Shadertoy example
vec4 lambert(vec4 lights[3], vec3 lightColors[3], vec3 p, vec3 baseColor, Ray r) {
  vec3 sumColor = vec3(0.0);
  vec3 nHat = surfaceNormal(p);

  for (int i = 0; i < 3; ++i) {
    vec3 lHat = normalize(lights[i].xyz - p);
    Ray lightRay = Ray(p, lHat);
    vec3 lamb = baseColor * clamp(dot(nHat, lHat), 0.0, 1.0) * lights[i].w * lightColors[i];
    sumColor += lamb * vec3(softShadow(lightRay, 0.1, 10.0, 8.0));
  }

  // Return average color
  sumColor /= 3.0;
  return vec4(sumColor, 1.0);
}

// Is it necessarily true that light color == specular color?
vec4 blinnPhong(vec4 lights[3], vec3 lightColors[3], vec3 p, vec3 baseColor, Ray r, float s) {
  vec3 sumColor = vec3(0.0);
  vec3 nHat = surfaceNormal(p);

  for (int i = 0; i < 3; ++i) {
    vec3 lHat = normalize(lights[i].xyz - p);
    vec3 h = normalize(lHat - r.direction);
    float angle = max(dot(h, nHat), 0.0);
    float spec = pow(angle, s);
    Ray lightRay = Ray(p, lHat);
    sumColor += lightColors[i] * vec3(spec) * lights[i].w * vec3(softShadow(lightRay, 0.1, 10.0, 8.0));
  }

  sumColor /= 3.0;
  return vec4(sumColor, 0.0);
}

// From class slides/GDC talk
vec4 subsurfaceScatter(vec4 lights[3], vec3 lightColors[3], vec3 p, vec3 baseColor, Ray view, float thinness) {
  // Tunable parameters
  float distort = 0.2;
  float glow = 4.0; // Similar to Blinn-Phong's specular power; higher => tighter highlight
  float scale = 10.0; // Higher => larger lit area
  float ambient = 0.0;

  vec3 totalCol = vec3(0.0);
  for (int i = 0; i < 3; ++i) {
    vec3 lHat = normalize(lights[i].xyz - p);
    vec3 normal = surfaceNormal(p);
    vec3 scatterDir = lHat + normal * distort;
    float lightReachingEye = pow(clamp(dot(-view.direction, -scatterDir), 0.0, 1.0), glow) * scale;
    float totalLight = thinness * (lightReachingEye + ambient);
    totalCol += baseColor * lightColors[i] * totalLight;
  }

  totalCol /= 3.0;
  return vec4(totalCol, 0.0);
}

// For starbits; based on Adam's Fresnel shader
vec4 glossy(vec4 lights[3], vec3 lightColors[3], vec3 p, vec3 baseColor, Ray r) {
  vec3 nHat = surfaceNormal(p);
  vec3 vHat = normalize(u_Eye - p);

  float fresnel = 1.0 - max(0.0, dot(vHat, nHat));
  fresnel = 0.25 + 0.75 * fresnel;

  vec3 newCol = mix(baseColor, galaxy(reflect(r.direction, nHat)).xyz, fresnel);
  return vec4(8.0 * (blinnPhong(lights, lightColors, p, newCol, r, 80.0).xyz +  vec3(0.08)) - 3.0 * (lambert(lights, lightColors, p, newCol, r).xyz), 0.0);
}
///////////////////////////

vec4 raymarch(Ray r, const float start, const int maxIterations, float t) {
  float depth = start;

  for (int i = 0; i < maxIterations; ++i) {
    vec3 p = pointOnRay(r, depth);

    // Find closest SDF
    SceneObject shape = sceneSDF(p);

    // Set up three-point lighting (w component is the intensity term)
    vec4 lights[3];
    vec3 lightColors[3];

    // Taken from Emily's Shadertoy example; to be tweaked
    lights[0] = vec4(6.0, 3.0, 5.0, 3.0); // key light
    lights[1] = vec4(-6.0, 3.0, 5.0, 2.5); // fill light
	  lights[2] = vec4(6.0, 5.0, -1.75, 4.0); // back light
    
    lightColors[0] = vec3(1.0);//vec3(0.9, 0.5, 0.9);
    lightColors[1] = vec3(1.0);//vec3(0.4, 0.7, 1.0);
    lightColors[2] = vec3(1.0);//vec3(1.0, 1.0, 0.2);

    // We're inside the shape, so the ray hit it; return color of shape + shading
    if (abs(shape.sdf) <= 0.01) {
      vec4 ao = vec4(vec3(ambientOcclusion(p, surfaceNormal(p), 1.0)), 1.0);
      vec4 color = lambert(lights, lightColors, p, shape.baseColor, r) * ao;
      switch (shape.reflectionModel) {
        // Lambertian
        case 0:
          break;
        // Blinn-Phong
        case 1:
          color += blinnPhong(lights, lightColors, p, shape.baseColor, r, 25.0) * ao;
          break;
        // Glossy/iridescent material
        case 2:
          color += glossy(lights, lightColors, p, shape.baseColor, r) * ao;
          break;
        // Subsurface scattering
        case 3:
          // Thinness is computed as ambient occlusion inside the object
          float thin = ambientOcclusion(p, -surfaceNormal(p), 1.0);
          color += subsurfaceScatter(lights, lightColors, p, shape.baseColor, r, thin) * ao;
          break;
      }
      // Gamma correction
      return pow(color, vec4(1.5/2.2));
    }

    // We've yet to hit anything; continue marching
    depth += shape.sdf;
  }

  // We miss all objects; return the background
  return galaxy(vec3(fs_Pos, 1.0) * (0.1 * fbm(vec3(0.000012 * u_Time)) + 1.0) * 90.0);
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
