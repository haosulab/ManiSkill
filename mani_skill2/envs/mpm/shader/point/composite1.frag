#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outPoint;

layout(set = 0, binding = 0) uniform sampler2D samplerSmoothedDepthLinear;
layout(set = 0, binding = 1) uniform sampler2D samplerPointColor;

// IBL
layout(set = 0, binding = 2) uniform samplerCube samplerEnvironment;
layout(set = 0, binding = 3) uniform sampler2D samplerBRDFLUT;

// layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 3;
// layout (constant_id = 1) const int NUM_POINT_LIGHTS = 10;
// layout (constant_id = 2) const int NUM_SPOT_LIGHTS = 10;

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 3;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 10;
layout (constant_id = 2) const int NUM_DIRECTIONAL_LIGHT_SHADOWS = 1;
layout (constant_id = 3) const int NUM_POINT_LIGHT_SHADOWS = 3;
layout (constant_id = 4) const int NUM_TEXTURED_LIGHT_SHADOWS = 1;
layout (constant_id = 5) const int NUM_SPOT_LIGHT_SHADOWS = 10;
layout (constant_id = 6) const int NUM_SPOT_LIGHTS = 10;


layout(set = 1, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
  mat4 prevViewMatrix;
  mat4 prevViewMatrixInverse;
  float width;
  float height;
} cameraBuffer;

#include "../common/lights.glsl"
#include "../common/shadow.glsl"
layout(set = 2, binding = 0) uniform SceneBuffer {
  vec4 ambientLight;
  DirectionalLight directionalLights[3];
  SpotLight spotLights[10];
  PointLight pointLights[10];
  SpotLight texturedLights[1];
} sceneBuffer;

struct LightBuffer {
  mat4 viewMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrix;
  mat4 projectionMatrixInverse;
  int width;
  int height;
};

layout(set = 2, binding = 1) uniform ShadowBuffer {
  LightBuffer directionalLightBuffers[3];
  LightBuffer spotLightBuffers[10];
  LightBuffer pointLightBuffers[60];
  LightBuffer texturedLightBuffers[1];
} shadowBuffer;

layout(set = 2, binding = 2) uniform samplerCube samplerPointLightDepths[3];
layout(set = 2, binding = 3) uniform sampler2D samplerDirectionalLightDepths[1];
layout(set = 2, binding = 4) uniform sampler2D samplerTexturedLightDepths[1];
layout(set = 2, binding = 5) uniform sampler2D samplerSpotLightDepths[10];
layout(set = 2, binding = 6) uniform sampler2D samplerTexturedLightTextures[1];

vec3 getCameraSpacePosition(vec2 uv, float z) {
  float depth = -cameraBuffer.projectionMatrix[2][2] - cameraBuffer.projectionMatrix[3][2] / z;
  vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
  vec4 csPosition = cameraBuffer.projectionMatrixInverse * ndc;
  return vec3(csPosition.xy / csPosition.w, z);
}

vec4 world2camera(vec4 pos) {
  return cameraBuffer.viewMatrix * pos;
}


vec3 diffuseIBL(vec3 albedo, vec3 N) {
  N = vec3(-N.y, N.z, -N.x);
  vec3 color = textureLod(samplerEnvironment, N, 5).rgb;
  return color * albedo;
}

vec3 specularIBL(vec3 fresnel, float roughness, vec3 N, vec3 V) {
  float dotNV = max(dot(N, V), 0);
  vec3 R = 2 * dot(N, V) * N - V;
  R = vec3(-R.y, R.z, -R.x);
  vec3 color = textureLod(samplerEnvironment, R, roughness * 5).rgb;
  vec2 envBRDF = texture(samplerBRDFLUT, vec2(roughness, dotNV)).xy;
  return color * (fresnel * envBRDF.x + envBRDF.y);
}

void main() {
  vec2 res = vec2(cameraBuffer.width, cameraBuffer.height);

  vec2 uv = inUV;
  float z = texture(samplerSmoothedDepthLinear, uv).x;
  if (z >= 0.0) {
    discard;
  }
  vec3 csp = getCameraSpacePosition(uv, z);

  uv = inUV + vec2(-1 / cameraBuffer.width, 0);
  vec3 cspnx = getCameraSpacePosition(uv, texture(samplerSmoothedDepthLinear, uv).x);

  uv = inUV + vec2(1 / cameraBuffer.width, 0);
  vec3 csppx = getCameraSpacePosition(uv, texture(samplerSmoothedDepthLinear, uv).x);

  uv = inUV + vec2(0, -1 / cameraBuffer.height);
  vec3 cspny = getCameraSpacePosition(uv, texture(samplerSmoothedDepthLinear, uv).x);

  uv = inUV + vec2(0, 1 / cameraBuffer.height);
  vec3 csppy = getCameraSpacePosition(uv, texture(samplerSmoothedDepthLinear, uv).x);

  vec3 pdx = csppx - csp;
  vec3 ndx = csp - cspnx;

  vec3 pdy = csppy - csp;
  vec3 ndy = csp - cspny;

  vec3 dx = abs(pdx.z) < abs(ndx.z) ? pdx: ndx;
  vec3 dy = abs(pdy.z) < abs(ndy.z) ? pdy: ndy;

  vec3 normal = normalize(cross(dy, dx));
  vec3 albedo = texture(samplerPointColor, inUV).rgb;

  // lighting
  vec4 csPosition = vec4(csp, 1.0);
  vec3 camDir = -normalize(csPosition.xyz);
  vec4 ndc = cameraBuffer.projectionMatrix * csPosition;
  ndc /= ndc.w;

  float roughness = 0.5;
  vec3 fresnel = vec3(0.2);

  vec3 color = vec3(0.0);


  // point light
  for (int i = 0; i < NUM_POINT_LIGHT_SHADOWS; ++i) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    mat4 shadowProj = shadowBuffer.pointLightBuffers[6 * i].projectionMatrix;

    vec3 l = pos - csPosition.xyz;
    vec3 wsl = vec3(cameraBuffer.viewMatrixInverse * vec4(l, 0));

    vec3 v = abs(wsl);
    vec4 p = shadowProj * vec4(0, 0, -max(max(v.x, v.y), v.z), 1);
    float pixelDepth = p.z / p.w;
    float shadowDepth = texture(samplerPointLightDepths[i], wsl).x;

    float visibility = step(pixelDepth - shadowDepth, 0);
    color += visibility * computePointLight(
        sceneBuffer.pointLights[i].emission.rgb,
        l, normal, camDir, albedo, roughness, fresnel);
  }

  for (int i = NUM_POINT_LIGHT_SHADOWS; i < NUM_POINT_LIGHTS; i++) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    color += computePointLight(
        sceneBuffer.pointLights[i].emission.rgb,
        l, normal, camDir, albedo, roughness, fresnel);
  }

  // directional light
  for (int i = 0; i < NUM_DIRECTIONAL_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.directionalLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.directionalLightBuffers[i].projectionMatrix;

    vec3 lightDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerDirectionalLightDepths[i], 0).x;
    float visibility = ShadowMapPCF(
        samplerDirectionalLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);

    color += visibility * computeDirectionalLight(
        lightDir,
        sceneBuffer.directionalLights[i].emission.rgb,
        normal, camDir, albedo, roughness, fresnel);
  }

  for (int i = NUM_DIRECTIONAL_LIGHT_SHADOWS; i < NUM_DIRECTIONAL_LIGHTS; ++i) {
    color += computeDirectionalLight(
        mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz,
        sceneBuffer.directionalLights[i].emission.rgb,
        normal, camDir, albedo, roughness, fresnel);
  }

  // spot light
  for (int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.spotLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.spotLightBuffers[i].projectionMatrix;

    vec3 pos = world2camera(vec4(sceneBuffer.spotLights[i].position.xyz, 1.f)).xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.spotLights[i].direction.xyz;
    vec3 l = pos - csPosition.xyz;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerSpotLightDepths[i], 0).x;
    float visibility = ShadowMapPCF(
        samplerSpotLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);

    color += visibility * computeSpotLight(
        sceneBuffer.spotLights[i].emission.a,
        sceneBuffer.spotLights[i].direction.a,
        centerDir,
        sceneBuffer.spotLights[i].emission.rgb,
        l, normal, camDir, albedo, roughness, fresnel);
  }

  for (int i = NUM_SPOT_LIGHT_SHADOWS; i < NUM_SPOT_LIGHTS; ++i) {
    vec3 pos = world2camera(vec4(sceneBuffer.spotLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.spotLights[i].direction.xyz;
    color += computeSpotLight(
        sceneBuffer.spotLights[i].emission.a,
        sceneBuffer.spotLights[i].direction.a,
        centerDir,
        sceneBuffer.spotLights[i].emission.rgb,
        l, normal, camDir, albedo, roughness, fresnel);
  }

  // textured light
  for (int i = 0; i < NUM_TEXTURED_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.texturedLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.texturedLightBuffers[i].projectionMatrix;

    vec3 pos = world2camera(vec4(sceneBuffer.texturedLights[i].position.xyz, 1.f)).xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.texturedLights[i].direction.xyz;
    vec3 l = pos - csPosition.xyz;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerTexturedLightDepths[i], 0).x;
    float visibility = ShadowMapPCF(
        samplerTexturedLightDepths[i], shadowMapCoord.xyz, resolution, 1 / resolution, 1);
    visibility *= texture(samplerTexturedLightTextures[i], shadowMapCoord.xy).x;

    color += visibility * computeSpotLight(
        sceneBuffer.texturedLights[i].emission.a,
        sceneBuffer.texturedLights[i].direction.a,
        centerDir,
        sceneBuffer.texturedLights[i].emission.rgb,
        l, normal, camDir, albedo, roughness, fresnel);
  }

  // environmental light
  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;
  color += diffuseIBL(albedo, wnormal);
  color += specularIBL(fresnel, roughness,
                       wnormal,
                       mat3(cameraBuffer.viewMatrixInverse) * camDir);

  color += sceneBuffer.ambientLight.rgb * albedo;

  outPoint.xyz = color;
}
