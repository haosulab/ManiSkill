struct PointLight {
  vec4 position;
  vec4 emission;
};

struct DirectionalLight {
  vec4 direction;
  vec4 emission;
};

struct SpotLight {
  vec4 position;
  vec4 direction;
  vec4 emission;
};


float diffuse(float NoL) {
  return NoL / 3.141592653589793f;
}

vec3 ggx(float NoL, float NoV, float NoH, float VoH, float roughness, vec3 fresnel) {
  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;

  float k = (alpha + 2 * roughness + 1.0) / 8.0;

  float FMi = ((-5.55473) * VoH - 6.98316) * VoH;
  vec3 frac = (fresnel + (1 - fresnel) * pow(2.0, FMi)) * alpha2;
  float nom0 = NoH * NoH * (alpha2 - 1) + 1;
  float nom1 = NoV * (1 - k) + k;
  float nom2 = NoL * (1 - k) + k;
  float nom = clamp((4 * 3.141592653589793f * nom0 * nom0 * nom1 * nom2), 1e-6, 4 * 3.141592653589793f);
  vec3 spec = frac / nom;

  return spec * NoL;
}

vec3 computeDirectionalLight(vec3 direction, vec3 emission, vec3 normal, vec3 camDir, vec3 diffuseAlbedo, float roughness, vec3 fresnel) {
  vec3 lightDir = -direction;

  vec3 H = lightDir + camDir;
  float H2 = dot(H, H);
  H = H2 < 1e-6 ? vec3(0) : normalize(H);
  float NoH = clamp(dot(normal, H), 1e-6, 1);
  float VoH = clamp(dot(camDir, H), 1e-6, 1);
  float NoL = clamp(dot(normal, lightDir), 0, 1);
  float NoV = clamp(dot(normal, camDir), 1e-6, 1);

  vec3 color = diffuseAlbedo * emission * diffuse(NoL);
  color += emission * ggx(NoL, NoV, NoH, VoH, roughness, fresnel);
  return color;
}

vec3 computePointLight(vec3 emission, vec3 l, vec3 normal, vec3 camDir, vec3 diffuseAlbedo, float roughness, vec3 fresnel) {
  float d = max(length(l), 0.0001);

  if (length(l) == 0) {
    return vec3(0.f);
  }

  vec3 lightDir = normalize(l);

  vec3 H = lightDir + camDir;
  float H2 = dot(H, H);
  H = H2 < 1e-6 ? vec3(0) : normalize(H);
  float NoH = clamp(dot(normal, H), 1e-6, 1);
  float VoH = clamp(dot(camDir, H), 1e-6, 1);
  float NoL = clamp(dot(normal, lightDir), 0, 1);
  float NoV = clamp(dot(normal, camDir), 1e-6, 1);

  vec3 color = diffuseAlbedo * emission * diffuse(NoL) / d / d;
  color += emission * ggx(NoL, NoV, NoH, VoH, roughness, fresnel) / d / d;
  return color;
}

vec3 computeSpotLight(float fov1, float fov2, vec3 centerDir, vec3 emission, vec3 l, vec3 normal, vec3 camDir, vec3 diffuseAlbedo, float roughness, vec3 fresnel) {
  float d = max(length(l), 0.0001);

  if (length(l) == 0) {
    return vec3(0.f);
  }

  vec3 lightDir = normalize(l);

  float cf1 = cos(fov1/2) + 1e-6;
  float cf2 = cos(fov2/2);

  float visibility = clamp((dot(-lightDir, centerDir) - cf2) / (cf1 - cf2), 0, 1);

  vec3 H = lightDir + camDir;
  float H2 = dot(H, H);
  H = H2 < 1e-6 ? vec3(0) : normalize(H);
  float NoH = clamp(dot(normal, H), 1e-6, 1);
  float VoH = clamp(dot(camDir, H), 1e-6, 1);
  float NoL = clamp(dot(normal, lightDir), 0, 1);
  float NoV = clamp(dot(normal, camDir), 1e-6, 1);

  vec3 color = diffuseAlbedo * emission * diffuse(NoL) / d / d;
  color += emission * ggx(NoL, NoV, NoH, VoH, roughness, fresnel) / d / d;
  return visibility * color;
}

vec3 computeSpotLight2(float fov, vec3 centerDir, vec3 emission, vec3 l, vec3 normal, vec3 camDir, vec3 diffuseAlbedo, float roughness, vec3 fresnel) {
  float d = max(length(l), 0.0001);

  if (length(l) == 0) {
    return vec3(0.f);
  }

  vec3 lightDir = normalize(l);

  if (dot(-lightDir, centerDir) <= 0) {
    return vec3(0.f);
  }

  vec3 H = lightDir + camDir;
  float H2 = dot(H, H);
  H = H2 < 1e-6 ? vec3(0) : normalize(H);
  float NoH = clamp(dot(normal, H), 1e-6, 1);
  float VoH = clamp(dot(camDir, H), 1e-6, 1);
  float NoL = clamp(dot(normal, lightDir), 0, 1);
  float NoV = clamp(dot(normal, camDir), 1e-6, 1);

  vec3 color = diffuseAlbedo * emission * diffuse(NoL) / d / d;
  color += emission * ggx(NoL, NoV, NoH, VoH, roughness, fresnel) / d / d;
  return color;
}
